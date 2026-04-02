//! Full replay processing into contiguous tensor buffers.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::protobuf::*;
use crate::types::*;

pub const N_CHANNELS: usize = 21;
pub const N_SCALARS: usize = 19;

pub struct ReplayData {
    pub states: Vec<f32>,
    pub scalars: Vec<f32>,
    pub working_harvesters: Vec<i32>,
    pub resource_transfers: Vec<f32>,
    pub titanium_delivered: Vec<i32>,
    pub axionite_delivered: Vec<i32>,
    pub map_w: usize,
    pub map_h: usize,
    pub n_turns: usize,
    pub winner: Option<u8>,
}

/// BFS from all core tiles for one team, returning flow-field distance grid.
/// Walls (env_grid == 1) are impassable. Unreachable tiles get i32::MAX.
pub fn compute_flow_distance(
    cores: &[(i32, u8, Pos)],
    team: u8,
    env_grid: &[u8],
    map_w: usize,
    map_h: usize,
) -> Vec<i32> {
    let n = map_w * map_h;
    let mut dist = vec![i32::MAX; n];
    let mut queue = VecDeque::new();
    for (_id, t, pos) in cores {
        if *t == team {
            let idx = pos.y as usize * map_w + pos.x as usize;
            if dist[idx] == i32::MAX {
                dist[idx] = 0;
                queue.push_back((pos.x as i32, pos.y as i32));
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        let d = dist[y as usize * map_w + x as usize] + 1;
        for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
            if nx < 0 || ny < 0 || nx as usize >= map_w || ny as usize >= map_h {
                continue;
            }
            let ni = ny as usize * map_w + nx as usize;
            if env_grid[ni] == 1 || dist[ni] <= d {
                continue;
            }
            dist[ni] = d;
            queue.push_back((nx, ny));
        }
    }
    dist
}

/// Shared helper: compute per-team core center from map cores.
pub fn compute_core_centers(cores: &[(i32, u8, Pos)]) -> [(i32, i32); 2] {
    let mut center = [(0i32, 0i32); 2];
    let mut count = [0i32; 2];
    for (_id, t, pos) in cores {
        let ti = *t as usize;
        center[ti].0 += pos.x;
        center[ti].1 += pos.y;
        count[ti] += 1;
    }
    for ti in 0..2 {
        if count[ti] > 0 {
            center[ti].0 /= count[ti];
            center[ti].1 /= count[ti];
        }
    }
    center
}

/// Shared helper: seed entity map with initial core entities.
pub fn seed_core_entities(cores: &[(i32, u8, Pos)], entities: &mut HashMap<i32, Entity>) {
    for (id, t, pos) in cores {
        entities.insert(
            *id,
            Entity {
                id: *id,
                kind: EntityKind::Core,
                team: *t,
                x: pos.x,
                y: pos.y,
                hp: 500,
                max_hp: 500,
                direction: 0,
                stored: 0,
                cooldown: 0,
                action_cooldown: 0,
                move_cooldown: 0,
                target_x: 0,
                target_y: 0,
            },
        );
    }
}

/// Process a full replay into contiguous buffers for tensor construction.
pub fn process_replay(data: &[u8], team: u8) -> ReplayData {
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
    let plane_size = map_w * map_h;
    let state_stride = N_CHANNELS * plane_size;

    let winner = get_varint(&top, 4).map(|v| v as u8);

    let mut entities: HashMap<i32, Entity> = HashMap::new();
    let mut harv_positions: HashMap<(i32, i32), i32> = HashMap::new();
    let mut tainted: HashSet<(i32, i32)> = HashSet::new();

    let enemy_team = if team == 0 { 1u8 } else { 0u8 };
    let mut core_hp: [i32; 2] = [500, 500];

    let flow_dist: [Vec<i32>; 2] = [
        compute_flow_distance(&map.cores, 0, &map.env_grid, map_w, map_h),
        compute_flow_distance(&map.cores, 1, &map.env_grid, map_w, map_h),
    ];
    seed_core_entities(&map.cores, &mut entities);

    let mut player_res: [PlayerRes; 2] = [PlayerRes::default(), PlayerRes::default()];
    let mut prev_ti_collected: [i32; 2] = [0, 0];
    let mut prev_ax_collected: [i32; 2] = [0, 0];
    let mut prev_own_ti_collected: i32 = 0;
    let mut prev_own_ax_collected: i32 = 0;
    let mut prev_own_core_hp: f32 = 500.0;
    let mut prev_own_working: f32 = 0.0;
    let team_idx = team as usize;
    let enemy_idx = enemy_team as usize;

    let turns = get_repeated_submsg(&top, 3, data);
    let n_turns = turns.len();

    let mut states = vec![0.0f32; n_turns * state_stride];
    let mut scalars = vec![0.0f32; n_turns * N_SCALARS];
    let mut working_harvesters = vec![0i32; n_turns * 2];
    let mut resource_transfers = vec![0.0f32; n_turns * 2];
    let mut titanium_delivered = vec![0i32; n_turns * 2];
    let mut axionite_delivered = vec![0i32; n_turns * 2];

    // Precompute static env channels once (channels 0-3 never change)
    let env_plane_len = 4 * plane_size; // 4 env channels
    let mut env_base = vec![0.0f32; env_plane_len];
    for y in 0..map_h {
        for x in 0..map_w {
            let base = y * map_w + x;
            let env_val = map.env_grid[y * map_w + x] as usize;
            if env_val <= 3 {
                env_base[env_val * plane_size + base] = 1.0;
            }
        }
    }

    // Maintain position-to-entity lookup incrementally
    let mut pos_to_eid: HashMap<(i32, i32), i32> = HashMap::new();
    for ent in entities.values() {
        pos_to_eid.insert((ent.x, ent.y), ent.id);
    }

    for (turn_idx, turn_data) in turns.iter().enumerate() {
        let state = &mut states[turn_idx * state_stride..(turn_idx + 1) * state_stride];

        // Copy precomputed env channels
        state[..env_plane_len].copy_from_slice(&env_base);

        // Entity channels
        let mut n_own_builders = 0i32;
        let mut n_enemy_builders = 0i32;
        let mut n_own_working = 0i32;
        let mut n_enemy_working = 0i32;

        for ent in entities.values() {
            let ex = ent.x as usize;
            let ey = ent.y as usize;
            if ent.x < 0 || ex >= map_w || ent.y < 0 || ey >= map_h {
                continue;
            }
            let base = ey * map_w + ex;
            let sign = if ent.team == team { 1.0f32 } else { -1.0 };

            match ent.kind {
                EntityKind::BuilderBot => {
                    state[17 * plane_size + base] = 1.0;
                    state[18 * plane_size + base] = sign;
                    if ent.team == team {
                        n_own_builders += 1;
                    } else {
                        n_enemy_builders += 1;
                    }
                }
                EntityKind::Marker => {}
                kind => {
                    if let Some(ch) = kind.building_channel() {
                        state[ch * plane_size + base] = 1.0;
                    }
                    state[14 * plane_size + base] = sign;
                    let d = ent.direction as usize;
                    if d < 9 {
                        let (dx, dy) = DIR_DELTA[d];
                        state[15 * plane_size + base] = dx;
                        state[16 * plane_size + base] = dy;
                    }
                    if kind.is_storing() && ent.stored > 0 {
                        state[19 * plane_size + base] = if ent.stored == 1 { 1.0 } else { -1.0 };
                    }
                    if kind == EntityKind::Harvester && ent.cooldown > 0 {
                        state[20 * plane_size + base] = ent.cooldown as f32 / 3.0;
                        if ent.team == team {
                            n_own_working += 1;
                        } else {
                            n_enemy_working += 1;
                        }
                    }
                    if kind == EntityKind::Core {
                        core_hp[ent.team as usize] = ent.hp;
                    }
                }
            }
        }

        // Scalars
        let own_res = &player_res[team_idx];
        let enemy_res = &player_res[enemy_idx];
        let sc = &mut scalars[turn_idx * N_SCALARS..(turn_idx + 1) * N_SCALARS];
        sc[0] = turn_idx as f32 / 2000.0;
        sc[1] = own_res.titanium as f32 / 500.0;
        sc[2] = own_res.axionite as f32 / 500.0;
        sc[3] = enemy_res.titanium as f32 / 500.0;
        sc[4] = enemy_res.axionite as f32 / 500.0;
        sc[5] = core_hp[team_idx] as f32 / 500.0;
        sc[6] = core_hp[enemy_idx] as f32 / 500.0;
        sc[7] = n_own_builders as f32 / 10.0;
        sc[8] = n_enemy_builders as f32 / 10.0;
        sc[9] = own_res.resources_collected as f32 / 500.0;
        sc[10] = enemy_res.resources_collected as f32 / 500.0;
        sc[11] = entities.len() as f32 / 100.0;

        let cur_own_core_hp = core_hp[team_idx] as f32;
        let cur_own_working = n_own_working as f32;
        sc[12] = cur_own_working / 20.0;
        sc[13] = (own_res.titanium_collected - prev_own_ti_collected) as f32 / 500.0;
        sc[14] = (own_res.axionite_collected - prev_own_ax_collected) as f32 / 500.0;
        sc[15] = (cur_own_core_hp - prev_own_core_hp) / 500.0;
        sc[16] = (cur_own_working - prev_own_working) / 20.0;
        prev_own_ti_collected = own_res.titanium_collected;
        prev_own_ax_collected = own_res.axionite_collected;
        prev_own_core_hp = cur_own_core_hp;
        prev_own_working = cur_own_working;

        let wh = &mut working_harvesters[turn_idx * 2..turn_idx * 2 + 2];
        wh[team_idx] = n_own_working;
        wh[enemy_idx] = n_enemy_working;

        // Decrement harvester cooldowns
        for ent in entities.values_mut() {
            if ent.kind == EntityKind::Harvester && ent.cooldown > 0 {
                ent.cooldown -= 1;
            }
        }

        // Process updates
        let turn_fields = decode_fields(turn_data, 0, turn_data.len());
        let updates = get_repeated_submsg(&turn_fields, 1, turn_data);

        for update_data in &updates {
            if let Some(event) = decode_update(update_data) {
                match event {
                    UpdateEvent::PlaceEntity(ent) => {
                        let eid = ent.id;
                        if ent.kind == EntityKind::Harvester {
                            harv_positions.insert((ent.x, ent.y), eid);
                        }
                        pos_to_eid.insert((ent.x, ent.y), eid);
                        entities.insert(eid, ent);
                    }
                    UpdateEvent::MoveBuilderBot { id, to } => {
                        if let Some(ent) = entities.get_mut(&id) {
                            if ent.kind == EntityKind::Harvester {
                                harv_positions.remove(&(ent.x, ent.y));
                                harv_positions.insert((to.x, to.y), id);
                            }
                            pos_to_eid.remove(&(ent.x, ent.y));
                            pos_to_eid.insert((to.x, to.y), id);
                            ent.x = to.x;
                            ent.y = to.y;
                        }
                    }
                    UpdateEvent::RemoveEntity { id } => {
                        if let Some(ent) = entities.remove(&id) {
                            pos_to_eid.remove(&(ent.x, ent.y));
                            if ent.kind == EntityKind::Harvester {
                                harv_positions.remove(&(ent.x, ent.y));
                            }
                            // Taint tile when friendly storing entity with
                            // a resource is destroyed (self-rebuild detection).
                            if ent.team == team
                                && ent.kind.is_storing()
                                && ent.stored > 0
                            {
                                tainted.insert((ent.x, ent.y));
                            }
                        }
                    }
                    UpdateEvent::UpdateHp { id, delta } => {
                        if let Some(ent) = entities.get_mut(&id) {
                            ent.hp += delta;
                        }
                    }
                    UpdateEvent::SetActionCooldown { id, value } => {
                        if let Some(ent) = entities.get_mut(&id) {
                            ent.action_cooldown = value;
                        }
                    }
                    UpdateEvent::SetMoveCooldown { id, value } => {
                        if let Some(ent) = entities.get_mut(&id) {
                            ent.move_cooldown = value;
                        }
                    }
                    UpdateEvent::DistributeResources(moves) => {
                        for (from, to) in &moves {
                            if let Some(&eid) = harv_positions.get(&(from.x, from.y)) {
                                if let Some(ent) = entities.get_mut(&eid) {
                                    ent.cooldown = 3;
                                }
                            }
                            let src_meta = pos_to_eid
                                .get(&(from.x, from.y))
                                .and_then(|src_id| entities.get(src_id))
                                .map(|src| (src.kind, src.team, src.stored));
                            let dst_meta = pos_to_eid
                                .get(&(to.x, to.y))
                                .and_then(|dst_id| entities.get(dst_id))
                                .map(|dst| (dst.kind, dst.team));
                            let mut moved_res = 1u8;
                            if let Some(&src_id) = pos_to_eid.get(&(from.x, from.y)) {
                                if let Some(src) = entities.get_mut(&src_id) {
                                    if src.kind.is_storing() {
                                        moved_res = if src.stored > 0 { src.stored } else { 1 };
                                        if src.kind != EntityKind::Harvester {
                                            src.stored = 0;
                                        }
                                    }
                                }
                            }
                            if let Some(&dst_id) = pos_to_eid.get(&(to.x, to.y)) {
                                if let Some(dst) = entities.get_mut(&dst_id) {
                                    if dst.kind.is_storing() {
                                        dst.stored = moved_res;
                                    }
                                }
                            }
                            if let (
                                Some((src_kind, src_team, src_stored)),
                                Some((dst_kind, dst_team)),
                            ) = (src_meta, dst_meta)
                            {
                                if src_kind.is_storing()
                                    && dst_kind.is_storing()
                                    && src_team == dst_team
                                    && src_stored > 0
                                {
                                    // Skip reward if destination was just
                                    // rebuilt (resource-destroying rebuild).
                                    if tainted.remove(&(to.x, to.y)) {
                                        continue;
                                    }
                                    let fd = &flow_dist[src_team as usize];
                                    let src_fd = fd[from.y as usize * map_w + from.x as usize];
                                    let dst_fd = fd[to.y as usize * map_w + to.x as usize];
                                    if src_fd != i32::MAX && dst_fd != i32::MAX && dst_fd < src_fd {
                                        let improvement = src_fd - dst_fd;
                                        // Axionite is worth 4x titanium
                                        let res_mult = if src_stored == 2 { 4.0f32 } else { 1.0 };
                                        let weight =
                                            res_mult * improvement as f32 / (src_fd.max(1) as f32).sqrt();
                                        resource_transfers
                                            [turn_idx * 2 + src_team as usize] += weight;
                                    }
                                }
                            }
                        }
                    }
                    UpdateEvent::UpdatePlayers { a, b } => {
                        if let Some(res) = a {
                            player_res[0] = res;
                        }
                        if let Some(res) = b {
                            player_res[1] = res;
                        }
                    }
                }
            }
        }

        // Clear taints not consumed by a transfer (e.g. enemy kills)
        tainted.clear();

        // Titanium/axionite delivered delta per team
        let td = &mut titanium_delivered[turn_idx * 2..turn_idx * 2 + 2];
        let ad = &mut axionite_delivered[turn_idx * 2..turn_idx * 2 + 2];
        for t in 0..2 {
            let cur_ti = player_res[t].titanium_collected;
            let cur_ax = player_res[t].axionite_collected;
            td[t] = cur_ti - prev_ti_collected[t];
            ad[t] = cur_ax - prev_ax_collected[t];
            prev_ti_collected[t] = cur_ti;
            prev_ax_collected[t] = cur_ax;
        }

        // Flow reward and delivery reward for critic observation
        let sc2 = &mut scalars[turn_idx * N_SCALARS..(turn_idx + 1) * N_SCALARS];
        sc2[17] = resource_transfers[turn_idx * 2 + team_idx];
        sc2[18] = (td[team_idx] + ad[team_idx] * 4) as f32;
    }

    ReplayData {
        states,
        scalars,
        working_harvesters,
        resource_transfers,
        titanium_delivered,
        axionite_delivered,
        map_w,
        map_h,
        n_turns,
        winner,
    }
}
