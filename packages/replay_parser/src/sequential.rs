//! Sequential behavioral-cloning extraction from replays.
//!
//! Processes a replay turn-by-turn, emitting one observation+action sample per
//! builder bot per turn. Within each turn, events are processed incrementally
//! so that each bot's observation reflects state changes from earlier bots.

use std::collections::HashMap;

use crate::protobuf::*;
use crate::replay::{compute_core_centers, seed_core_entities};
use crate::types::*;

// Observation layout matching bots/tensor_inference/main.py
const BC_GRID_SIZE: usize = 5;
const BC_GRID_HALF: i32 = 2;
const BC_TILE_FEATURES: usize = 21;
const BC_GLOBAL_FEATURES: usize = 19;
pub const BC_OBS_SIZE: usize =
    BC_GRID_SIZE * BC_GRID_SIZE * BC_TILE_FEATURES + BC_GLOBAL_FEATURES;
const BC_GLOBAL_OBS_START: usize = BC_GRID_SIZE * BC_GRID_SIZE * BC_TILE_FEATURES;
pub const BC_ACTION_SIZE: usize = 5;

pub struct BCSamples {
    pub obs: Vec<f32>,
    pub actions: Vec<u8>,
    pub turns: Vec<u32>,
    pub bot_ids: Vec<i32>,
    pub n_samples: usize,
    pub obs_size: usize,
    pub winner: Option<u8>,
    pub map_w: usize,
    pub map_h: usize,
}

// ── Action encoding helpers ──────────────────────────────────────────────

/// Map EntityKind -> obs one-hot index (0-9), matching BLD_IDX in main.py.
fn bld_idx_for_obs(kind: EntityKind) -> usize {
    match kind {
        EntityKind::Harvester => 1,
        EntityKind::Conveyor | EntityKind::ArmouredConveyor => 2,
        EntityKind::Splitter => 3,
        EntityKind::Road => 4,
        EntityKind::Barrier => 5,
        EntityKind::Bridge => 6,
        EntityKind::Foundry => 7,
        EntityKind::Gunner | EntityKind::Sentinel | EntityKind::Breach | EntityKind::Launcher => 8,
        EntityKind::Core => 9,
        _ => 0,
    }
}

/// Map EntityKind -> build action index (0-12), matching BUILD_ACTIONS in main.py.
fn build_action_idx(kind: EntityKind) -> u8 {
    match kind {
        EntityKind::Harvester => 1,
        EntityKind::Conveyor => 2,
        EntityKind::Road => 3,
        EntityKind::Gunner => 4,
        EntityKind::Sentinel => 5,
        EntityKind::Breach => 6,
        EntityKind::Splitter => 7,
        EntityKind::ArmouredConveyor => 8,
        EntityKind::Barrier => 9,
        EntityKind::Foundry => 10,
        EntityKind::Launcher => 11,
        EntityKind::Bridge => 12,
        _ => 0,
    }
}

/// Bridge target (dx, dy) -> BRIDGE_OFFSETS index.
/// Uses a 7x7 lookup table (indexed by dx+3, dy+3) instead of looping.
fn bridge_target_to_idx(dx: i32, dy: i32) -> u8 {
    // Precomputed: for each (dx, dy) where 0 < dx²+dy² ≤ 9, the sequential index.
    // 255 = invalid. Row-major [dy+3][dx+3].
    const LUT: [[u8; 7]; 7] = {
        let mut table = [[255u8; 7]; 7];
        let mut idx: u8 = 0;
        let mut oy: i32 = -3;
        while oy <= 3 {
            let mut ox: i32 = -3;
            while ox <= 3 {
                let d2 = ox * ox + oy * oy;
                if d2 > 0 && d2 <= 9 {
                    table[(oy + 3) as usize][(ox + 3) as usize] = idx;
                    idx += 1;
                }
                ox += 1;
            }
            oy += 1;
        }
        table
    };
    if dx < -3 || dx > 3 || dy < -3 || dy > 3 {
        return 0;
    }
    let v = LUT[(dy + 3) as usize][(dx + 3) as usize];
    if v == 255 { 0 } else { v }
}

/// (dx, dy) -> DIR_MAP_9 index.  Layout: NW N NE / W C E / SW S SE
#[inline]
fn dxdy_to_dir9(dx: i32, dy: i32) -> u8 {
    let cx = (dx + 1).clamp(0, 2);
    let cy = (dy + 1).clamp(0, 2);
    (cy * 3 + cx) as u8
}

/// Protobuf Direction enum -> CONV_DIR_4 index (N=0, W=1, E=2, S=3).
#[inline]
fn proto_dir_to_conv4(dir: u8) -> u8 {
    match dir {
        1 => 0, // NORTH
        7 => 1, // WEST
        3 => 2, // EAST
        5 => 3, // SOUTH
        _ => 0,
    }
}

/// Check if a tile is impassable (wall or non-road building present).
fn is_tile_impassable(
    wx: i32,
    wy: i32,
    env_grid: &[u8],
    map_w: usize,
    map_h: usize,
    pos_buildings: &HashMap<(i32, i32), i32>,
    entities: &HashMap<i32, Entity>,
) -> bool {
    if wx < 0 || wy < 0 || wx as usize >= map_w || wy as usize >= map_h {
        return true;
    }
    let env = env_grid[wy as usize * map_w + wx as usize];
    if env == 1 {
        return true;
    }
    if let Some(&eid) = pos_buildings.get(&(wx, wy)) {
        if let Some(ent) = entities.get(&eid) {
            if ent.kind != EntityKind::Road && ent.kind != EntityKind::Marker {
                return true;
            }
        }
    }
    false
}

// ── Observation encoding ─────────────────────────────────────────────────

/// Encode the 5x5 local observation for a builder bot.
/// Must match _encode_state_into in bots/tensor_inference/main.py exactly.
#[allow(clippy::too_many_arguments)]
fn encode_local_obs(
    obs: &mut [f32],
    bot: &Entity,
    entities: &HashMap<i32, Entity>,
    env_grid: &[u8],
    map_w: usize,
    map_h: usize,
    team: u8,
    player_res: &[PlayerRes; 2],
    core_center: &[(i32, i32); 2],
    core_hp: &[i32; 2],
    turn_idx: usize,
    spawn_round: i32,
    prev_own_ti_collected: i32,
    prev_own_ax_collected: i32,
    pos_buildings: &HashMap<(i32, i32), i32>,
    pos_bots: &HashMap<(i32, i32), i32>,
    scale_milli: i32,
) {
    let bx = bot.x;
    let by = bot.y;
    let team_idx = team as usize;
    let enemy_idx = 1 - team_idx;

    // 5x5 local grid
    let mut n_local_builders = 0i32;
    for ly in 0..BC_GRID_SIZE as i32 {
        for lx in 0..BC_GRID_SIZE as i32 {
            let wx = bx + lx - BC_GRID_HALF;
            let wy = by + ly - BC_GRID_HALF;
            let base = (ly as usize * BC_GRID_SIZE + lx as usize) * BC_TILE_FEATURES;

            if wx < 0 || wy < 0 || wx as usize >= map_w || wy as usize >= map_h {
                continue;
            }

            let env = env_grid[wy as usize * map_w + wx as usize];
            obs[base] = if env == 1 { 1.0 } else { 0.0 };
            obs[base + 1] = if env == 2 { 1.0 } else { 0.0 };
            obs[base + 2] = if env == 3 { 1.0 } else { 0.0 };
            obs[base + 3] = if is_tile_impassable(
                wx, wy, env_grid, map_w, map_h, pos_buildings, entities,
            ) {
                1.0
            } else {
                0.0
            };
            obs[base + 4] = 1.0; // default: building type = none

            if let Some(&eid) = pos_buildings.get(&(wx, wy)) {
                if let Some(ent) = entities.get(&eid) {
                    let bidx = bld_idx_for_obs(ent.kind);
                    if bidx > 0 {
                        obs[base + 4] = 0.0;
                        obs[base + 4 + bidx] = 1.0;
                    }
                    obs[base + 14] = if ent.team == team { 1.0 } else { -1.0 };
                    let d = ent.direction as usize;
                    if d < 9 {
                        let (dx, dy) = DIR_DELTA[d];
                        obs[base + 15] = dx;
                        obs[base + 16] = dy;
                    }
                    if ent.kind.is_storing() && ent.stored > 0 {
                        obs[base + 18] = if ent.stored == 1 { 1.0 } else { -1.0 };
                    }
                }
            }

            if let Some(&eid) = pos_bots.get(&(wx, wy)) {
                if let Some(ent) = entities.get(&eid) {
                    obs[base + 19] = 1.0;
                    obs[base + 20] = if ent.team == team { 1.0 } else { -1.0 };
                    if ent.team == team {
                        n_local_builders += 1;
                    }
                }
            }
        }
    }

    // Global features (15)
    let g = BC_GLOBAL_OBS_START;
    let own = &player_res[team_idx];
    obs[g] = turn_idx as f32 / 2000.0;
    let cc = core_center[team_idx];
    obs[g + 1] = (bx - cc.0) as f32 / map_w.max(1) as f32;
    obs[g + 2] = (by - cc.1) as f32 / map_h.max(1) as f32;
    obs[g + 3] = own.titanium as f32 / 500.0;
    obs[g + 4] = own.axionite as f32 / 500.0;
    let scale_pct = (1000 + scale_milli) as f32 / 10.0;
    obs[g + 5] = scale_pct / 500.0;

<<<<<<< Updated upstream
    let cc = core_center[team_idx];
    obs[g + 6] = (cc.0 - bx) as f32;
    obs[g + 7] = (cc.1 - by) as f32;
=======
    let cdx = cc.0 - bx;
    let cdy = cc.1 - by;
    let cmax = cdx.abs().max(cdy.abs()).max(1);
    obs[g + 6] = cdx as f32 / cmax as f32;
    obs[g + 7] = cdy as f32 / cmax as f32;
>>>>>>> Stashed changes

    let ec = core_center[enemy_idx];
    obs[g + 8] = (ec.0 - bx) as f32;
    obs[g + 9] = (ec.1 - by) as f32;

    obs[g + 10] = n_local_builders as f32 / 10.0;
    obs[g + 11] = (own.titanium_collected - prev_own_ti_collected) as f32 / 500.0;
    obs[g + 12] = (own.axionite_collected - prev_own_ax_collected) as f32 / 500.0;
    obs[g + 13] = (core_hp[team_idx] - core_hp[enemy_idx]) as f32 / 500.0;
    obs[g + 14] = spawn_round as f32 / 2000.0;
<<<<<<< Updated upstream
    obs[g + 15] = 1.0; // core_known (always true in replay data)
    obs[g + 16] = 1.0; // ecore_known (always true in replay data)
    obs[g + 17] = map_w as f32 / 50.0;
    obs[g + 18] = map_h as f32 / 50.0;
=======

    // Positional embedding (8) — core-relative to match bot encoding
    sinusoidal_pos_embed(
        bx - cc.0,
        by - cc.1,
        &mut obs[g + BC_GLOBAL_FEATURES..g + BC_GLOBAL_FEATURES + BC_POS_EMBED_DIM],
    );
>>>>>>> Stashed changes
}

// ── Per-turn pending action ──────────────────────────────────────────────

struct PendingBotAction {
    build_type: u8,
    build_dir: u8,
    conv_dir: u8,
    bridge_target: u8,
    move_dir: u8,
    has_action: bool,
}

impl Default for PendingBotAction {
    fn default() -> Self {
        Self {
            build_type: 0,
            build_dir: 4,
            conv_dir: 0,
            bridge_target: 0,
            move_dir: 4,
            has_action: false,
        }
    }
}

// ── Incremental state mutation helpers ──────────────────────────────────

/// Apply a single entity placement to the world state.
#[inline]
fn apply_place(
    ent: Entity,
    entities: &mut HashMap<i32, Entity>,
    pos_buildings: &mut HashMap<(i32, i32), i32>,
    pos_bots: &mut HashMap<(i32, i32), i32>,
    harv_positions: &mut HashMap<(i32, i32), i32>,
    scale_milli: &mut i32,
    team: u8,
) {
    let eid = ent.id;
    if ent.kind == EntityKind::BuilderBot {
        pos_bots.insert((ent.x, ent.y), eid);
    } else if ent.kind != EntityKind::Marker {
        if ent.kind == EntityKind::Harvester {
            harv_positions.insert((ent.x, ent.y), eid);
        }
        pos_buildings.insert((ent.x, ent.y), eid);
        if ent.team == team {
            *scale_milli += ent.kind.scale_milli();
        }
    }
    entities.insert(eid, ent);
}

/// Apply a single entity removal to the world state.
#[inline]
fn apply_remove(
    id: i32,
    entities: &mut HashMap<i32, Entity>,
    pos_buildings: &mut HashMap<(i32, i32), i32>,
    pos_bots: &mut HashMap<(i32, i32), i32>,
    harv_positions: &mut HashMap<(i32, i32), i32>,
    scale_milli: &mut i32,
    team: u8,
) {
    if let Some(ent) = entities.remove(&id) {
        if ent.kind == EntityKind::Harvester {
            harv_positions.remove(&(ent.x, ent.y));
        }
        if ent.kind == EntityKind::BuilderBot {
            pos_bots.remove(&(ent.x, ent.y));
        } else if ent.kind != EntityKind::Marker && ent.kind != EntityKind::Core {
            pos_buildings.remove(&(ent.x, ent.y));
            if ent.team == team {
                *scale_milli -= ent.kind.scale_milli();
            }
        }
    }
}

/// Apply a bot move to the world state.
#[inline]
fn apply_move(
    id: i32,
    to_x: i32,
    to_y: i32,
    entities: &mut HashMap<i32, Entity>,
    pos_bots: &mut HashMap<(i32, i32), i32>,
    harv_positions: &mut HashMap<(i32, i32), i32>,
) {
    if let Some(ent) = entities.get_mut(&id) {
        let old_pos = (ent.x, ent.y);
        ent.x = to_x;
        ent.y = to_y;
        if ent.kind == EntityKind::Harvester {
            harv_positions.remove(&old_pos);
            harv_positions.insert((to_x, to_y), id);
        }
        if ent.kind == EntityKind::BuilderBot {
            pos_bots.remove(&old_pos);
            pos_bots.insert((to_x, to_y), id);
        }
    }
}

/// Apply an HP change to the world state.
#[inline]
fn apply_hp(id: i32, delta: i32, entities: &mut HashMap<i32, Entity>, core_hp: &mut [i32; 2]) {
    if let Some(ent) = entities.get_mut(&id) {
        ent.hp += delta;
        if ent.kind == EntityKind::Core {
            core_hp[ent.team as usize] = ent.hp;
        }
    }
}

/// Apply a cooldown change.
#[inline]
fn apply_cooldown(id: i32, field: u8, value: i32, entities: &mut HashMap<i32, Entity>) {
    if let Some(ent) = entities.get_mut(&id) {
        if field == 0 {
            ent.action_cooldown = value;
        } else {
            ent.move_cooldown = value;
        }
    }
}

/// Apply resource distribution.
fn apply_distribute(
    moves: &[(Pos, Pos)],
    entities: &mut HashMap<i32, Entity>,
    harv_positions: &HashMap<(i32, i32), i32>,
    pos_buildings: &HashMap<(i32, i32), i32>,
) {
    for (from, to) in moves {
        if let Some(&eid) = harv_positions.get(&(from.x, from.y)) {
            if let Some(ent) = entities.get_mut(&eid) {
                ent.cooldown = 3;
            }
        }
        let mut moved_res = 1u8;
        if let Some(&src_id) = pos_buildings.get(&(from.x, from.y)) {
            if let Some(src) = entities.get_mut(&src_id) {
                if src.kind.is_storing() {
                    moved_res = if src.stored > 0 { src.stored } else { 1 };
                    if src.kind != EntityKind::Harvester {
                        src.stored = 0;
                    }
                }
            }
        }
        if let Some(&dst_id) = pos_buildings.get(&(to.x, to.y)) {
            if let Some(dst) = entities.get_mut(&dst_id) {
                if dst.kind.is_storing() {
                    dst.stored = moved_res;
                }
            }
        }
    }
}

// ── Deferred events that apply at turn end (not per-bot) ────────────────

struct DeferredTurnEnd {
    distributes: Vec<Vec<(Pos, Pos)>>,
    player_res: Option<(Option<PlayerRes>, Option<PlayerRes>)>,
}

// ── Buffered per-bot-group mutations ────────────────────────────────────

struct BotGroupMutations {
    moves: Vec<(i32, i32, i32)>,    // (id, to_x, to_y)
    places: Vec<Entity>,
    removes: Vec<i32>,
    hp_changes: Vec<(i32, i32)>,    // (id, delta)
    cooldowns: Vec<(i32, u8, i32)>, // (id, field, value)
}

impl BotGroupMutations {
    fn new() -> Self {
        Self {
            moves: Vec::new(),
            places: Vec::new(),
            removes: Vec::new(),
            hp_changes: Vec::new(),
            cooldowns: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.moves.clear();
        self.places.clear();
        self.removes.clear();
        self.hp_changes.clear();
        self.cooldowns.clear();
    }

    fn is_empty(&self) -> bool {
        self.moves.is_empty()
            && self.places.is_empty()
            && self.removes.is_empty()
            && self.hp_changes.is_empty()
            && self.cooldowns.is_empty()
    }
}

/// Flush buffered mutations into the world state.
#[allow(clippy::too_many_arguments)]
fn flush_mutations(
    buf: &mut BotGroupMutations,
    entities: &mut HashMap<i32, Entity>,
    pos_buildings: &mut HashMap<(i32, i32), i32>,
    pos_bots: &mut HashMap<(i32, i32), i32>,
    harv_positions: &mut HashMap<(i32, i32), i32>,
    scale_milli: &mut i32,
    core_hp: &mut [i32; 2],
    team: u8,
) {
    for &(id, to_x, to_y) in &buf.moves {
        apply_move(id, to_x, to_y, entities, pos_bots, harv_positions);
    }
    for ent in buf.places.drain(..) {
        apply_place(
            ent,
            entities,
            pos_buildings,
            pos_bots,
            harv_positions,
            scale_milli,
            team,
        );
    }
    for &id in &buf.removes {
        apply_remove(
            id,
            entities,
            pos_buildings,
            pos_bots,
            harv_positions,
            scale_milli,
            team,
        );
    }
    for &(id, delta) in &buf.hp_changes {
        apply_hp(id, delta, entities, core_hp);
    }
    for &(id, field, value) in &buf.cooldowns {
        apply_cooldown(id, field, value, entities);
    }
    buf.clear();
}

// ── Main sequential processor ────────────────────────────────────────────

/// Process replay sequentially, emitting one merged BC sample per bot per turn.
///
/// Within each turn, state mutations are applied incrementally between bot
/// groups so that each bot's observation reflects changes from earlier bots.
/// Bot group boundaries are detected via SetActionCooldown events — the game
/// engine emits events in per-bot groups:
///   [BotOutput] SetActionCooldown PlaceEntity [MoveBuilderBot SetMoveCooldown]
pub fn process_replay_sequential(data: &[u8], team: u8) -> BCSamples {
    let top = decode_fields(data, 0, data.len());
    let map = get_submsg(&top, 1, data)
        .map(decode_map)
        .unwrap_or(MapData {
            width: 0,
            height: 0,
            env_grid: vec![],
            cores: vec![],
        });

    let map_w = map.width;
    let map_h = map.height;
    let winner = get_varint(&top, 4).map(|v| v as u8);

    let mut entities: HashMap<i32, Entity> = HashMap::new();
    let mut harv_positions: HashMap<(i32, i32), i32> = HashMap::new();
    let mut core_hp: [i32; 2] = [500, 500];

    let core_center = compute_core_centers(&map.cores);
    seed_core_entities(&map.cores, &mut entities);

    let mut player_res: [PlayerRes; 2] = [PlayerRes::default(), PlayerRes::default()];
    let mut spawn_rounds: HashMap<i32, i32> = HashMap::new();
    let mut prev_own_ti_collected: i32 = 0;
    let mut prev_own_ax_collected: i32 = 0;
    let team_idx = team as usize;

    let turns = get_repeated_submsg(&top, 3, data);

    let est_samples = turns.len() * 4;
    let mut bc_obs: Vec<f32> = Vec::with_capacity(est_samples * BC_OBS_SIZE);
    let mut bc_actions: Vec<u8> = Vec::with_capacity(est_samples * BC_ACTION_SIZE);
    let mut bc_turns: Vec<u32> = Vec::with_capacity(est_samples);
    let mut bc_bot_ids: Vec<i32> = Vec::with_capacity(est_samples);
    let mut n_samples: usize = 0;

    // Maintain position-indexed lookups incrementally across turns
    let mut pos_buildings: HashMap<(i32, i32), i32> = HashMap::new();
    let mut pos_bots: HashMap<(i32, i32), i32> = HashMap::new();
    let mut scale_milli: i32 = 0;
    for ent in entities.values() {
        match ent.kind {
            EntityKind::BuilderBot => {
                pos_bots.insert((ent.x, ent.y), ent.id);
            }
            EntityKind::Marker => {}
            EntityKind::Core => {
                pos_buildings.insert((ent.x, ent.y), ent.id);
            }
            _ => {
                pos_buildings.insert((ent.x, ent.y), ent.id);
                if ent.team == team {
                    scale_milli += ent.kind.scale_milli();
                }
            }
        }
    }

    // Reusable buffer for per-bot-group mutations
    let mut group_buf = BotGroupMutations::new();

    for (turn_idx, turn_data) in turns.iter().enumerate() {
        for ent in entities.values_mut() {
            if ent.kind == EntityKind::Harvester && ent.cooldown > 0 {
                ent.cooldown -= 1;
            }
        }

        let turn_fields = decode_fields(turn_data, 0, turn_data.len());
        let updates = get_repeated_submsg(&turn_fields, 1, turn_data);

        // Per-bot state within this turn
        let mut current_bot_id: Option<i32> = None;
        let mut current_bot_action = PendingBotAction::default();
        let mut current_bot_obs_start: Option<usize> = None;
        let mut turn_end = DeferredTurnEnd {
            distributes: Vec::new(),
            player_res: None,
        };

        // Process events sequentially, flushing mutations between bot groups
        for update_data in &updates {
            if let Some(event) = decode_update(update_data) {
                match event {
                    UpdateEvent::SetActionCooldown { id, value } => {
                        // Check if this starts a new friendly builder group
                        let is_friendly_builder = entities
                            .get(&id)
                            .map(|e| e.team == team && e.kind == EntityKind::BuilderBot)
                            .unwrap_or(false);

                        if is_friendly_builder {
                            // Emit previous bot's sample if any
                            if let Some(bot_id) = current_bot_id.take() {
                                if current_bot_action.has_action {
                                    bc_actions.push(current_bot_action.build_type);
                                    bc_actions.push(if current_bot_action.build_type > 0 {
                                        current_bot_action.build_dir
                                    } else {
                                        4
                                    });
                                    bc_actions.push(current_bot_action.conv_dir);
                                    bc_actions.push(current_bot_action.bridge_target);
                                    bc_actions.push(current_bot_action.move_dir);
                                    bc_turns.push(turn_idx as u32);
                                    bc_bot_ids.push(bot_id);
                                    n_samples += 1;
                                } else if let Some(start) = current_bot_obs_start.take() {
                                    // No action — discard the pre-allocated obs
                                    bc_obs.truncate(start);
                                }
                            }

                            // Flush previous bot group's mutations so new bot
                            // sees up-to-date state
                            if !group_buf.is_empty() {
                                flush_mutations(
                                    &mut group_buf,
                                    &mut entities,
                                    &mut pos_buildings,
                                    &mut pos_bots,
                                    &mut harv_positions,
                                    &mut scale_milli,
                                    &mut core_hp,
                                    team,
                                );
                            }

                            // Encode this bot's observation from current state
                            let sr = spawn_rounds.get(&id).copied().unwrap_or(0);
                            let obs_start = bc_obs.len();
                            bc_obs.resize(obs_start + BC_OBS_SIZE, 0.0);
                            if let Some(bot) = entities.get(&id) {
                                encode_local_obs(
                                    &mut bc_obs[obs_start..],
                                    bot,
                                    &entities,
                                    &map.env_grid,
                                    map_w,
                                    map_h,
                                    team,
                                    &player_res,
                                    &core_center,
                                    &core_hp,
                                    turn_idx,
                                    sr,
                                    prev_own_ti,
                                    prev_own_ax,
                                    &pos_buildings,
                                    &pos_bots,
                                    scale_milli,
                                );
                            }

                            current_bot_id = Some(id);
                            current_bot_action = PendingBotAction::default();
                            current_bot_obs_start = Some(obs_start);
                        }

                        // Buffer the cooldown mutation
                        group_buf.cooldowns.push((id, 0, value));
                    }

                    UpdateEvent::MoveBuilderBot { id, to } => {
                        // Extract bot info before any mutable borrows
                        let bot_info = entities.get(&id).map(|b| (b.kind, b.team, b.x, b.y));

                        if let Some((EntityKind::BuilderBot, t, bx, by)) = bot_info {
                            if t == team {
                                if current_bot_id == Some(id) {
                                    let dx = to.x - bx;
                                    let dy = to.y - by;
                                    current_bot_action.move_dir = dxdy_to_dir9(dx, dy);
                                    current_bot_action.has_action = true;
                                } else {
                                    // Builder that had no SetActionCooldown —
                                    // move-only action. Flush previous group,
                                    // encode obs, start new group.
                                    if let Some(prev_id) = current_bot_id.take() {
                                        if current_bot_action.has_action {
                                            bc_actions.push(current_bot_action.build_type);
                                            bc_actions.push(if current_bot_action.build_type > 0 {
                                                current_bot_action.build_dir
                                            } else {
                                                4
                                            });
                                            bc_actions.push(current_bot_action.conv_dir);
                                            bc_actions.push(current_bot_action.bridge_target);
                                            bc_actions.push(current_bot_action.move_dir);
                                            bc_turns.push(turn_idx as u32);
                                            bc_bot_ids.push(prev_id);
                                            n_samples += 1;
                                        } else if let Some(start) = current_bot_obs_start.take() {
                                            bc_obs.truncate(start);
                                        }
                                    }

                                    if !group_buf.is_empty() {
                                        flush_mutations(
                                            &mut group_buf,
                                            &mut entities,
                                            &mut pos_buildings,
                                            &mut pos_bots,
                                            &mut harv_positions,
                                            &mut scale_milli,
                                            &mut core_hp,
                                            team,
                                        );
                                    }

                                    let dx = to.x - bx;
                                    let dy = to.y - by;

                                    let sr = spawn_rounds.get(&id).copied().unwrap_or(0);
                                    let obs_start = bc_obs.len();
                                    bc_obs.resize(obs_start + BC_OBS_SIZE, 0.0);
                                    if let Some(bot) = entities.get(&id) {
                                        encode_local_obs(
                                            &mut bc_obs[obs_start..],
                                            bot,
                                            &entities,
                                            &map.env_grid,
                                            map_w,
                                            map_h,
                                            team,
                                            &player_res,
                                            &core_center,
                                            &core_hp,
                                            turn_idx,
                                            sr,
                                            prev_own_ti,
                                            prev_own_ax,
                                            &pos_buildings,
                                            &pos_bots,
                                            scale_milli,
                                        );
                                    }

                                    current_bot_id = Some(id);
                                    current_bot_action = PendingBotAction::default();
                                    current_bot_action.move_dir = dxdy_to_dir9(dx, dy);
                                    current_bot_action.has_action = true;
                                    current_bot_obs_start = Some(obs_start);
                                }
                            }
                        }
                        group_buf.moves.push((id, to.x, to.y));
                    }

                    UpdateEvent::PlaceEntity(ent) => {
                        if ent.kind == EntityKind::BuilderBot {
                            spawn_rounds.insert(ent.id, turn_idx as i32);
                        } else if ent.team == team
                            && ent.kind != EntityKind::Marker
                            && build_action_idx(ent.kind) > 0
                        {
                            // Friendly building — attribute to current bot
                            if let (Some(_bid), Some(builder)) =
                                (current_bot_id, entities.get(&current_bot_id.unwrap_or(-1)))
                            {
                                let bdx = ent.x - builder.x;
                                let bdy = ent.y - builder.y;
                                current_bot_action.build_type = build_action_idx(ent.kind);
                                current_bot_action.build_dir = dxdy_to_dir9(bdx, bdy);
                                current_bot_action.conv_dir = if matches!(
                                    ent.kind,
                                    EntityKind::Conveyor
                                        | EntityKind::Splitter
                                        | EntityKind::ArmouredConveyor
                                ) {
                                    proto_dir_to_conv4(ent.direction)
                                } else {
                                    0
                                };
                                current_bot_action.bridge_target =
                                    if ent.kind == EntityKind::Bridge {
                                        bridge_target_to_idx(
                                            ent.target_x - ent.x,
                                            ent.target_y - ent.y,
                                        )
                                    } else {
                                        0
                                    };
                                current_bot_action.has_action = true;
                            }
                        }
                        group_buf.places.push(ent);
                    }

                    UpdateEvent::RemoveEntity { id } => {
                        group_buf.removes.push(id);
                    }
                    UpdateEvent::UpdateHp { id, delta } => {
                        group_buf.hp_changes.push((id, delta));
                    }
                    UpdateEvent::SetMoveCooldown { id, value } => {
                        group_buf.cooldowns.push((id, 1, value));
                    }
                    UpdateEvent::DistributeResources(moves) => {
                        turn_end.distributes.push(moves);
                    }
                    UpdateEvent::UpdatePlayers { a, b } => {
                        turn_end.player_res = Some((a, b));
                    }
                }
            }
        }

<<<<<<< Updated upstream
        // Phase 2: Emit samples from pre-turn state
        for (&bot_id, pa) in &pending {
            if !pa.has_action {
                continue;
            }
            if let Some(bot) = entities.get(&bot_id) {
                let sr = spawn_rounds.get(&bot_id).copied().unwrap_or(0);
                let obs_start = bc_obs.len();
                bc_obs.resize(obs_start + BC_OBS_SIZE, 0.0);
                encode_local_obs(
                    &mut bc_obs[obs_start..],
                    bot,
                    &entities,
                    &map.env_grid,
                    map_w,
                    map_h,
                    team,
                    &player_res,
                    &core_center,
                    &core_hp,
                    turn_idx,
                    sr,
                    prev_own_ti_collected,
                    prev_own_ax_collected,
                    &pos_buildings,
                    &pos_bots,
                    n_team_buildings,
                );
                bc_actions.push(pa.build_type);
                bc_actions.push(if pa.build_type > 0 { pa.build_dir } else { 4 });
                bc_actions.push(pa.conv_dir);
                bc_actions.push(pa.bridge_target);
                bc_actions.push(pa.move_dir);
=======
        // Emit final bot's sample
        if let Some(bot_id) = current_bot_id.take() {
            if current_bot_action.has_action {
                bc_actions.push(current_bot_action.build_type);
                bc_actions.push(if current_bot_action.build_type > 0 {
                    current_bot_action.build_dir
                } else {
                    4
                });
                bc_actions.push(current_bot_action.conv_dir);
                bc_actions.push(current_bot_action.bridge_target);
                bc_actions.push(current_bot_action.move_dir);
>>>>>>> Stashed changes
                bc_turns.push(turn_idx as u32);
                bc_bot_ids.push(bot_id);
                n_samples += 1;
            } else if let Some(start) = current_bot_obs_start.take() {
                bc_obs.truncate(start);
            }
        }

        // Flush remaining bot group mutations
        if !group_buf.is_empty() {
            flush_mutations(
                &mut group_buf,
                &mut entities,
                &mut pos_buildings,
                &mut pos_bots,
                &mut harv_positions,
                &mut scale_milli,
                &mut core_hp,
                team,
            );
        }

        // Apply turn-end events
        for moves in &turn_end.distributes {
            apply_distribute(moves, &mut entities, &harv_positions, &pos_buildings);
        }
        if let Some((a, b)) = turn_end.player_res {
            if let Some(res) = a {
                player_res[0] = res;
            }
            if let Some(res) = b {
                player_res[1] = res;
            }
        }

        prev_own_ti_collected = player_res[team_idx].titanium_collected;
        prev_own_ax_collected = player_res[team_idx].axionite_collected;
    }

    BCSamples {
        obs: bc_obs,
        actions: bc_actions,
        turns: bc_turns,
        bot_ids: bc_bot_ids,
        n_samples,
        obs_size: BC_OBS_SIZE,
        winner,
        map_w,
        map_h,
    }
}
