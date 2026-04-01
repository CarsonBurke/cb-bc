//! Sequential behavioral-cloning extraction from replays.
//!
//! Processes a replay turn-by-turn, emitting one observation+action sample per
//! builder bot per turn. Observations are encoded from pre-turn state; actions
//! are merged from all events in the turn.

use std::collections::HashMap;

use crate::protobuf::*;
use crate::replay::{compute_core_centers, seed_core_entities};
use crate::types::*;

// Observation layout matching bots/tensor_inference/main.py
const BC_GRID_SIZE: usize = 5;
const BC_GRID_HALF: i32 = 2;
const BC_TILE_FEATURES: usize = 21;
const BC_GLOBAL_FEATURES: usize = 17;
const BC_POS_EMBED_DIM: usize = 8;
pub const BC_OBS_SIZE: usize =
    BC_GRID_SIZE * BC_GRID_SIZE * BC_TILE_FEATURES + BC_GLOBAL_FEATURES + BC_POS_EMBED_DIM;
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

/// 8-dim sinusoidal position embedding matching tensor_lib/embed.py.
fn sinusoidal_pos_embed(x: i32, y: i32, out: &mut [f32]) {
    const BASE: f64 = 10000.0;
    const D_MODEL: usize = 8;
    const QUARTER: usize = D_MODEL / 4;
    let half = QUARTER * 2;

    for i in 0..QUARTER {
        let freq = 1.0 / BASE.powf(2.0 * i as f64 / D_MODEL as f64);
        let ax = x as f64 * freq;
        let ay = y as f64 * freq;
        out[2 * i] = ax.sin() as f32;
        out[2 * i + 1] = ax.cos() as f32;
        out[half + 2 * i] = ay.sin() as f32;
        out[half + 2 * i + 1] = ay.cos() as f32;
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
    prev_own_ti: f32,
    prev_own_ax: f32,
    pos_buildings: &HashMap<(i32, i32), i32>,
    pos_bots: &HashMap<(i32, i32), i32>,
    n_team_buildings: i32,
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
    obs[g + 1] = bx as f32 / map_w.max(1) as f32;
    obs[g + 2] = by as f32 / map_h.max(1) as f32;
    obs[g + 3] = own.titanium as f32 / 500.0;
    obs[g + 4] = own.axionite as f32 / 500.0;
    let scale_pct = 100.0 + 5.0 * n_team_buildings as f32;
    obs[g + 5] = scale_pct / 500.0;

    let cc = core_center[team_idx];
    obs[g + 6] = (cc.0 - bx) as f32;
    obs[g + 7] = (cc.1 - by) as f32;

    let ec = core_center[enemy_idx];
    obs[g + 8] = (ec.0 - bx) as f32;
    obs[g + 9] = (ec.1 - by) as f32;

    obs[g + 10] = n_local_builders as f32 / 10.0;
    obs[g + 11] = (own.titanium as f32 - prev_own_ti) / 500.0;
    obs[g + 12] = (own.axionite as f32 - prev_own_ax) / 500.0;
    obs[g + 13] = (core_hp[team_idx] - core_hp[enemy_idx]) as f32 / 500.0;
    obs[g + 14] = spawn_round as f32 / 2000.0;
    obs[g + 15] = 1.0; // core_known (always true in replay data)
    obs[g + 16] = 1.0; // ecore_known (always true in replay data)

    // Positional embedding (8)
    sinusoidal_pos_embed(
        bx,
        by,
        &mut obs[g + BC_GLOBAL_FEATURES..g + BC_GLOBAL_FEATURES + BC_POS_EMBED_DIM],
    );
}

// ── Per-turn pending action ──────────────────────────────────────────────

#[derive(Default)]
struct PendingBotAction {
    build_type: u8,
    build_dir: u8,
    conv_dir: u8,
    bridge_target: u8,
    move_dir: u8,
    has_action: bool,
}

struct DeferredMove {
    id: i32,
    to_x: i32,
    to_y: i32,
}
struct DeferredPlace {
    ent: Entity,
}
struct DeferredRemove {
    id: i32,
}
struct DeferredHp {
    id: i32,
    delta: i32,
}
struct DeferredCooldown {
    id: i32,
    field: u8,
    value: i32,
}

// ── Main sequential processor ────────────────────────────────────────────

/// Process replay sequentially, emitting one merged BC sample per bot per turn.
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
    let mut prev_own_ti: f32 = 0.0;
    let mut prev_own_ax: f32 = 0.0;
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
    let mut n_team_buildings: i32 = 0;
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
                    n_team_buildings += 1;
                }
            }
        }
    }

    for (turn_idx, turn_data) in turns.iter().enumerate() {
        for ent in entities.values_mut() {
            if ent.kind == EntityKind::Harvester && ent.cooldown > 0 {
                ent.cooldown -= 1;
            }
        }

        // Phase 1: Collect events into pending actions
        let mut pending: HashMap<i32, PendingBotAction> = HashMap::new();
        let mut deferred_moves: Vec<DeferredMove> = Vec::new();
        let mut deferred_places: Vec<DeferredPlace> = Vec::new();
        let mut deferred_removes: Vec<DeferredRemove> = Vec::new();
        let mut deferred_hp: Vec<DeferredHp> = Vec::new();
        let mut deferred_cooldowns: Vec<DeferredCooldown> = Vec::new();
        let mut deferred_res: Option<(Option<PlayerRes>, Option<PlayerRes>)> = None;
        let mut deferred_distributes: Vec<Vec<(Pos, Pos)>> = Vec::new();

        let turn_fields = decode_fields(turn_data, 0, turn_data.len());
        let updates = get_repeated_submsg(&turn_fields, 1, turn_data);

        for update_data in &updates {
            if let Some(event) = decode_update(update_data) {
                match event {
                    UpdateEvent::MoveBuilderBot { id, to } => {
                        if let Some(bot) = entities.get(&id) {
                            if bot.kind == EntityKind::BuilderBot && bot.team == team {
                                let dx = to.x - bot.x;
                                let dy = to.y - bot.y;
                                let pa = pending.entry(id).or_default();
                                pa.move_dir = dxdy_to_dir9(dx, dy);
                                pa.has_action = true;
                            }
                        }
                        deferred_moves.push(DeferredMove {
                            id,
                            to_x: to.x,
                            to_y: to.y,
                        });
                    }
                    UpdateEvent::PlaceEntity(ent) => {
                        if ent.kind == EntityKind::BuilderBot {
                            spawn_rounds.insert(ent.id, turn_idx as i32);
                            deferred_places.push(DeferredPlace { ent });
                        } else {
                            // Find adjacent builder on our team
                            let mut builder_id: Option<i32> = None;
                            const SEARCH_ORDER: [(i32, i32); 9] = [
                                (0, 0),
                                (-1, -1),
                                (0, -1),
                                (1, -1),
                                (-1, 0),
                                (1, 0),
                                (-1, 1),
                                (0, 1),
                                (1, 1),
                            ];
                            for &(dx, dy) in &SEARCH_ORDER {
                                let bx = ent.x + dx;
                                let by = ent.y + dy;
                                if let Some(&bid) = pos_bots.get(&(bx, by)) {
                                    if let Some(b) = entities.get(&bid) {
                                        if b.team == team
                                            && b.kind == EntityKind::BuilderBot
                                        {
                                            builder_id = Some(bid);
                                            break;
                                        }
                                    }
                                }
                            }

                            if let Some(bid) = builder_id {
                                if let Some(builder) = entities.get(&bid) {
                                    let bdx = ent.x - builder.x;
                                    let bdy = ent.y - builder.y;
                                    let pa = pending.entry(bid).or_default();
                                    pa.build_type = build_action_idx(ent.kind);
                                    pa.build_dir = dxdy_to_dir9(bdx, bdy);
                                    pa.conv_dir = if matches!(
                                        ent.kind,
                                        EntityKind::Conveyor
                                            | EntityKind::Splitter
                                            | EntityKind::ArmouredConveyor
                                    ) {
                                        proto_dir_to_conv4(ent.direction)
                                    } else {
                                        0
                                    };
                                    pa.bridge_target = if ent.kind == EntityKind::Bridge {
                                        bridge_target_to_idx(
                                            ent.target_x - ent.x,
                                            ent.target_y - ent.y,
                                        )
                                    } else {
                                        0
                                    };
                                    pa.has_action = true;
                                }
                            }

                            deferred_places.push(DeferredPlace { ent });
                        }
                    }
                    UpdateEvent::RemoveEntity { id } => {
                        deferred_removes.push(DeferredRemove { id });
                    }
                    UpdateEvent::UpdateHp { id, delta } => {
                        deferred_hp.push(DeferredHp { id, delta });
                    }
                    UpdateEvent::SetActionCooldown { id, value } => {
                        deferred_cooldowns.push(DeferredCooldown {
                            id,
                            field: 0,
                            value,
                        });
                    }
                    UpdateEvent::SetMoveCooldown { id, value } => {
                        deferred_cooldowns.push(DeferredCooldown {
                            id,
                            field: 1,
                            value,
                        });
                    }
                    UpdateEvent::DistributeResources(moves) => {
                        deferred_distributes.push(moves);
                    }
                    UpdateEvent::UpdatePlayers { a, b } => {
                        deferred_res = Some((a, b));
                    }
                }
            }
        }

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
                    prev_own_ti,
                    prev_own_ax,
                    &pos_buildings,
                    &pos_bots,
                    n_team_buildings,
                );
                bc_actions.push(pa.build_type);
                bc_actions.push(if pa.build_type > 0 { pa.build_dir } else { 4 });
                bc_actions.push(pa.conv_dir);
                bc_actions.push(pa.bridge_target);
                bc_actions.push(pa.move_dir);
                bc_turns.push(turn_idx as u32);
                bc_bot_ids.push(bot_id);
                n_samples += 1;
            }
        }

        // Phase 3: Apply deferred mutations
        for dm in &deferred_moves {
            if let Some(ent) = entities.get_mut(&dm.id) {
                let old_pos = (ent.x, ent.y);
                ent.x = dm.to_x;
                ent.y = dm.to_y;
                if ent.kind == EntityKind::Harvester {
                    harv_positions.remove(&old_pos);
                    harv_positions.insert((dm.to_x, dm.to_y), dm.id);
                }
                if ent.kind == EntityKind::BuilderBot {
                    pos_bots.remove(&old_pos);
                    pos_bots.insert((dm.to_x, dm.to_y), dm.id);
                }
            }
        }
        for dp in deferred_places {
            let eid = dp.ent.id;
            if dp.ent.kind == EntityKind::BuilderBot {
                pos_bots.insert((dp.ent.x, dp.ent.y), eid);
            } else {
                if dp.ent.kind == EntityKind::Harvester {
                    harv_positions.insert((dp.ent.x, dp.ent.y), eid);
                }
                pos_buildings.insert((dp.ent.x, dp.ent.y), eid);
                if dp.ent.team == team && dp.ent.kind != EntityKind::Core {
                    n_team_buildings += 1;
                }
            }
            entities.insert(eid, dp.ent);
        }
        for dr in &deferred_removes {
            if let Some(ent) = entities.remove(&dr.id) {
                if ent.kind == EntityKind::Harvester {
                    harv_positions.remove(&(ent.x, ent.y));
                }
                if ent.kind == EntityKind::BuilderBot {
                    pos_bots.remove(&(ent.x, ent.y));
                } else if ent.kind != EntityKind::Marker && ent.kind != EntityKind::Core {
                    pos_buildings.remove(&(ent.x, ent.y));
                    if ent.team == team {
                        n_team_buildings -= 1;
                    }
                }
            }
        }
        for dh in &deferred_hp {
            if let Some(ent) = entities.get_mut(&dh.id) {
                ent.hp += dh.delta;
                if ent.kind == EntityKind::Core {
                    core_hp[ent.team as usize] = ent.hp;
                }
            }
        }
        for dc in &deferred_cooldowns {
            if let Some(ent) = entities.get_mut(&dc.id) {
                if dc.field == 0 {
                    ent.action_cooldown = dc.value;
                } else {
                    ent.move_cooldown = dc.value;
                }
            }
        }
        for moves in &deferred_distributes {
            for (from, to) in moves {
                if let Some(&eid) = harv_positions.get(&(from.x, from.y)) {
                    if let Some(ent) = entities.get_mut(&eid) {
                        ent.cooldown = 3;
                    }
                }
                let mut moved_res = 1u8;
                // pos_buildings covers all storing entities (harvesters, conveyors, etc.)
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
        if let Some((a, b)) = deferred_res {
            if let Some(res) = a {
                player_res[0] = res;
            }
            if let Some(res) = b {
                player_res[1] = res;
            }
        }

        prev_own_ti = player_res[team_idx].titanium as f32;
        prev_own_ax = player_res[team_idx].axionite as f32;
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
