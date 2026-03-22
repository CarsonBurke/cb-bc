//! Fast replay parser for .replay26 protobuf files.
//!
//! Replaces the pure-Python protobuf decoder + state extraction with native Rust.
//! Exposes the same interface via PyO3 so it's a drop-in replacement.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ── Protobuf wire format decoder ──────────────────────────────────────────

const VARINT: u8 = 0;
const FIXED64: u8 = 1;
const LEN_DELIM: u8 = 2;
const FIXED32: u8 = 5;

#[inline]
fn decode_varint(data: &[u8], mut pos: usize) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let b = data[pos];
        pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (result, pos)
}

/// Raw protobuf field: (field_id, wire_type, value_start, value_end_or_varint)
struct RawField {
    field_id: u32,
    wire_type: u8,
    // For VARINT: value stored directly
    // For LEN_DELIM/FIXED*: slice bounds into data
    varint_val: u64,
    slice_start: usize,
    slice_end: usize,
}

fn decode_fields(data: &[u8], start: usize, end: usize) -> Vec<RawField> {
    let mut fields = Vec::new();
    let mut pos = start;
    while pos < end {
        let (tag, new_pos) = decode_varint(data, pos);
        pos = new_pos;
        let field_id = (tag >> 3) as u32;
        let wire_type = (tag & 0x07) as u8;
        match wire_type {
            VARINT => {
                let (val, new_pos) = decode_varint(data, pos);
                pos = new_pos;
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: val,
                    slice_start: 0,
                    slice_end: 0,
                });
            }
            FIXED64 => {
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: 0,
                    slice_start: pos,
                    slice_end: pos + 8,
                });
                pos += 8;
            }
            LEN_DELIM => {
                let (len, new_pos) = decode_varint(data, pos);
                pos = new_pos;
                let len = len as usize;
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: 0,
                    slice_start: pos,
                    slice_end: pos + len,
                });
                pos += len;
            }
            FIXED32 => {
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: 0,
                    slice_start: pos,
                    slice_end: pos + 4,
                });
                pos += 4;
            }
            _ => {
                // Skip unknown wire types
                break;
            }
        }
    }
    fields
}

// ── Domain types ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum EntityKind {
    BuilderBot = 0,
    Conveyor = 1,
    Splitter = 2,
    ArmouredConveyor = 3,
    Bridge = 4,
    Harvester = 5,
    Foundry = 6,
    Road = 7,
    Barrier = 8,
    Marker = 9,
    Core = 10,
    Gunner = 11,
    Sentinel = 12,
    Breach = 13,
    Launcher = 14,
}

impl EntityKind {
    fn from_field_id(id: u32) -> Option<Self> {
        match id {
            10 => Some(Self::BuilderBot),
            11 => Some(Self::Conveyor),
            12 => Some(Self::Splitter),
            13 => Some(Self::ArmouredConveyor),
            14 => Some(Self::Bridge),
            15 => Some(Self::Harvester),
            16 => Some(Self::Foundry),
            17 => Some(Self::Road),
            18 => Some(Self::Barrier),
            19 => Some(Self::Marker),
            20 => Some(Self::Core),
            21 => Some(Self::Gunner),
            22 => Some(Self::Sentinel),
            23 => Some(Self::Breach),
            24 => Some(Self::Launcher),
            _ => None,
        }
    }

    fn is_storing(self) -> bool {
        matches!(
            self,
            Self::Conveyor
                | Self::Splitter
                | Self::ArmouredConveyor
                | Self::Harvester
                | Self::Foundry
                | Self::Bridge
        )
    }

    /// Map channel offset for building types (channels 4-11)
    fn building_channel(self) -> Option<usize> {
        match self {
            Self::Harvester => Some(5),
            Self::Conveyor | Self::Foundry | Self::Bridge => Some(6),
            Self::Splitter => Some(7),
            Self::Road | Self::Barrier => Some(8),
            Self::Gunner | Self::Sentinel | Self::Breach | Self::Launcher => Some(9),
            Self::ArmouredConveyor => Some(10),
            Self::Core => Some(11),
            Self::BuilderBot | Self::Marker => None,
        }
    }
}

#[derive(Clone, Debug)]
struct Entity {
    id: i32,
    kind: EntityKind,
    team: u8,
    x: i32,
    y: i32,
    hp: i32,
    max_hp: i32,
    direction: u8,
    stored: u8,
    cooldown: i32,
    action_cooldown: i32,
    move_cooldown: i32,
}

#[derive(Clone, Debug, Default)]
struct PlayerRes {
    titanium: i32,
    axionite: i32,
    resources_collected: i32,
    titanium_collected: i32,
    axionite_collected: i32,
}

#[derive(Clone, Debug)]
struct Pos {
    x: i32,
    y: i32,
}

// ── Protobuf field extraction helpers ─────────────────────────────────────

fn get_varint(fields: &[RawField], id: u32) -> Option<u64> {
    fields
        .iter()
        .rev()
        .find(|f| f.field_id == id && f.wire_type == VARINT)
        .map(|f| f.varint_val)
}

fn get_i32(fields: &[RawField], id: u32) -> i32 {
    get_varint(fields, id).unwrap_or(0) as i32
}

fn get_submsg<'a>(fields: &[RawField], id: u32, data: &'a [u8]) -> Option<&'a [u8]> {
    fields
        .iter()
        .rev()
        .find(|f| f.field_id == id && f.wire_type == LEN_DELIM)
        .map(|f| &data[f.slice_start..f.slice_end])
}

fn get_repeated_submsg<'a>(fields: &[RawField], id: u32, data: &'a [u8]) -> Vec<&'a [u8]> {
    fields
        .iter()
        .filter(|f| f.field_id == id && f.wire_type == LEN_DELIM)
        .map(|f| &data[f.slice_start..f.slice_end])
        .collect()
}

fn decode_pos(data: &[u8]) -> Pos {
    let fields = decode_fields(data, 0, data.len());
    Pos {
        x: get_i32(&fields, 1),
        y: get_i32(&fields, 2),
    }
}

fn decode_player_res(data: &[u8]) -> PlayerRes {
    let fields = decode_fields(data, 0, data.len());
    PlayerRes {
        titanium: get_i32(&fields, 1),
        axionite: get_i32(&fields, 2),
        resources_collected: get_i32(&fields, 3),
        titanium_collected: get_i32(&fields, 4),
        axionite_collected: get_i32(&fields, 5),
    }
}

/// Parse an Entity protobuf submessage.
fn decode_entity(data: &[u8], _outer_data: &[u8]) -> Entity {
    // Entity fields are at the outer level; kind submessage fields 10-24
    let fields = decode_fields(data, 0, data.len());
    let id = get_i32(&fields, 1);
    let team = get_varint(&fields, 2).unwrap_or(0) as u8;
    let pos = get_submsg(&fields, 3, data)
        .map(decode_pos)
        .unwrap_or(Pos { x: 0, y: 0 });
    let hp = get_i32(&fields, 4);
    let max_hp = get_i32(&fields, 5);

    // Determine kind from oneof fields 10-24
    let mut kind = EntityKind::BuilderBot;
    let mut direction: u8 = 0;
    let mut stored: u8 = 0;
    let mut cooldown: i32 = 0;

    for fid in 10..=24 {
        if let Some(kind_val) = EntityKind::from_field_id(fid) {
            if let Some(sub) = get_submsg(&fields, fid, data) {
                kind = kind_val;
                let sub_fields = decode_fields(sub, 0, sub.len());
                // Most entity subtypes have direction at field 1
                direction = get_varint(&sub_fields, 1).unwrap_or(0) as u8;
                // stored resource at field 2 for conveyors etc
                stored = get_varint(&sub_fields, 2).unwrap_or(0) as u8;
                // Harvester: cooldown at field 1, resourceType at field 2
                if kind_val == EntityKind::Harvester {
                    cooldown = get_i32(&sub_fields, 1);
                    stored = get_varint(&sub_fields, 2).unwrap_or(0) as u8;
                }
                // Gunner/etc: ammoAmount at field 3
                break;
            }
            // Check for varint too (empty submessage = default entity)
            if get_varint(&fields, fid).is_some() {
                kind = kind_val;
                break;
            }
        }
    }

    Entity {
        id,
        kind,
        team,
        x: pos.x,
        y: pos.y,
        hp,
        max_hp,
        direction,
        stored,
        cooldown,
        action_cooldown: 0,
        move_cooldown: 0,
    }
}

// ── Replay-level parsing ──────────────────────────────────────────────────

struct MapData {
    width: usize,
    height: usize,
    env_grid: Vec<u8>, // row-major, env enum values
    cores: Vec<(i32, u8, Pos)>, // (id, team, position)
}

fn decode_map(data: &[u8]) -> MapData {
    let fields = decode_fields(data, 0, data.len());
    let width = get_i32(&fields, 1) as usize;
    let height = get_i32(&fields, 2) as usize;

    let mut env_grid = vec![0u8; width * height];

    // Rows (field 3, repeated)
    let rows = get_repeated_submsg(&fields, 3, data);
    for (y, row_data) in rows.iter().enumerate() {
        if y >= height {
            break;
        }
        let row_fields = decode_fields(row_data, 0, row_data.len());
        // tiles field 1 - repeated enum (can be packed or individual)
        // Check for packed encoding first
        let packed = row_fields
            .iter()
            .find(|f| f.field_id == 1 && f.wire_type == LEN_DELIM);
        if let Some(p) = packed {
            let tile_data = &row_data[p.slice_start..p.slice_end];
            let mut pos = 0;
            let mut x = 0;
            while pos < tile_data.len() && x < width {
                let (val, new_pos) = decode_varint(tile_data, pos);
                pos = new_pos;
                env_grid[y * width + x] = val as u8;
                x += 1;
            }
        } else {
            // Individual varint fields
            let mut x = 0;
            for f in &row_fields {
                if f.field_id == 1 && f.wire_type == VARINT {
                    if x < width {
                        env_grid[y * width + x] = f.varint_val as u8;
                        x += 1;
                    }
                }
            }
        }
    }

    // Cores (field 4, repeated)
    let mut cores = Vec::new();
    for core_data in get_repeated_submsg(&fields, 4, data) {
        let cf = decode_fields(core_data, 0, core_data.len());
        let id = get_i32(&cf, 1);
        let team = get_varint(&cf, 2).unwrap_or(0) as u8;
        let pos = get_submsg(&cf, 3, core_data)
            .map(decode_pos)
            .unwrap_or(Pos { x: 0, y: 0 });
        cores.push((id, team, pos));
    }

    MapData {
        width,
        height,
        env_grid,
        cores,
    }
}

/// Update types we care about from Turn.updates
enum UpdateEvent {
    PlaceEntity(Entity),
    MoveBuilderBot { id: i32, to: Pos },
    RemoveEntity { id: i32 },
    DistributeResources(Vec<(Pos, Pos)>),
    UpdateHp { id: i32, delta: i32 },
    UpdatePlayers { a: Option<PlayerRes>, b: Option<PlayerRes> },
    SetActionCooldown { id: i32, value: i32 },
    SetMoveCooldown { id: i32, value: i32 },
}

fn decode_update(data: &[u8]) -> Option<UpdateEvent> {
    let fields = decode_fields(data, 0, data.len());

    // PlaceEntity (field 1)
    if let Some(sub) = get_submsg(&fields, 1, data) {
        let pf = decode_fields(sub, 0, sub.len());
        if let Some(ent_data) = get_submsg(&pf, 1, sub) {
            return Some(UpdateEvent::PlaceEntity(decode_entity(ent_data, data)));
        }
    }

    // MoveBuilderBot (field 2)
    if let Some(sub) = get_submsg(&fields, 2, data) {
        let mf = decode_fields(sub, 0, sub.len());
        let id = get_i32(&mf, 1);
        let to = get_submsg(&mf, 2, sub)
            .map(decode_pos)
            .unwrap_or(Pos { x: 0, y: 0 });
        return Some(UpdateEvent::MoveBuilderBot { id, to });
    }

    // RemoveEntity (field 3)
    if let Some(sub) = get_submsg(&fields, 3, data) {
        let rf = decode_fields(sub, 0, sub.len());
        let id = get_i32(&rf, 1);
        return Some(UpdateEvent::RemoveEntity { id });
    }

    // DistributeResources (field 4)
    if let Some(sub) = get_submsg(&fields, 4, data) {
        let df = decode_fields(sub, 0, sub.len());
        let mut moves = Vec::new();
        for mv_data in get_repeated_submsg(&df, 1, sub) {
            let mvf = decode_fields(mv_data, 0, mv_data.len());
            let from = get_submsg(&mvf, 1, mv_data)
                .map(decode_pos)
                .unwrap_or(Pos { x: 0, y: 0 });
            let to = get_submsg(&mvf, 2, mv_data)
                .map(decode_pos)
                .unwrap_or(Pos { x: 0, y: 0 });
            moves.push((from, to));
        }
        return Some(UpdateEvent::DistributeResources(moves));
    }

    // UpdateHp (field 5)
    if let Some(sub) = get_submsg(&fields, 5, data) {
        let hf = decode_fields(sub, 0, sub.len());
        let id = get_i32(&hf, 1);
        let delta = get_i32(&hf, 2);
        return Some(UpdateEvent::UpdateHp { id, delta });
    }

    // UpdatePlayers (field 6)
    if let Some(sub) = get_submsg(&fields, 6, data) {
        let uf = decode_fields(sub, 0, sub.len());
        if let Some(players_data) = get_submsg(&uf, 1, sub) {
            let pf = decode_fields(players_data, 0, players_data.len());
            let a = get_submsg(&pf, 1, players_data).map(decode_player_res);
            let b = get_submsg(&pf, 2, players_data).map(decode_player_res);
            return Some(UpdateEvent::UpdatePlayers { a, b });
        }
    }

    // SetActionCooldown (field 7)
    if let Some(sub) = get_submsg(&fields, 7, data) {
        let cf = decode_fields(sub, 0, sub.len());
        return Some(UpdateEvent::SetActionCooldown {
            id: get_i32(&cf, 1),
            value: get_i32(&cf, 2),
        });
    }

    // SetMoveCooldown (field 8)
    if let Some(sub) = get_submsg(&fields, 8, data) {
        let cf = decode_fields(sub, 0, sub.len());
        return Some(UpdateEvent::SetMoveCooldown {
            id: get_i32(&cf, 1),
            value: get_i32(&cf, 2),
        });
    }

    // Fields 9-12 (BotOutput, IndicatorLine, IndicatorDot, FireTurret) — skip
    None
}

// ── State extraction ──────────────────────────────────────────────────────

const N_CHANNELS: usize = 19;
const N_SCALARS: usize = 17;

const DIR_DELTA: [(f32, f32); 9] = [
    (0.0, 0.0),   // CENTRE
    (0.0, -1.0),  // NORTH
    (1.0, -1.0),  // NORTHEAST
    (1.0, 0.0),   // EAST
    (1.0, 1.0),   // SOUTHEAST
    (0.0, 1.0),   // SOUTH
    (-1.0, 1.0),  // SOUTHWEST
    (-1.0, 0.0),  // WEST
    (-1.0, -1.0), // NORTHWEST
];

/// Processed replay data with contiguous buffers.
struct ReplayData {
    /// Contiguous f32 buffer: n_turns * (N_CHANNELS * map_h * map_w) floats.
    /// Turn t starts at offset t * state_stride.
    states: Vec<f32>,
    /// Contiguous f32 buffer: n_turns * N_SCALARS floats.
    scalars: Vec<f32>,
    /// Flat i32 buffer: n_turns * 2 values [a0, b0, a1, b1, ...].
    working_harvesters: Vec<i32>,
    /// Flat i32 buffer: n_turns * 2 — per-turn titanium delivered delta [a0, b0, a1, b1, ...].
    titanium_delivered: Vec<i32>,
    map_w: usize,
    map_h: usize,
    n_turns: usize,
    winner: Option<u8>,
}

/// Full replay processing into contiguous buffers.
fn process_replay(data: &[u8], team: u8) -> ReplayData {
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

    let enemy_team = if team == 0 { 1u8 } else { 0u8 };
    let mut core_hp: [i32; 2] = [500, 500];

    for (id, t, pos) in &map.cores {
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
            },
        );
    }

    let mut player_res: [PlayerRes; 2] = [PlayerRes::default(), PlayerRes::default()];
    let mut prev_ti_collected: [i32; 2] = [0, 0];
    // Track previous-turn values for delta scalars
    let mut prev_own_ti: f32 = 0.0;
    let mut prev_own_ax: f32 = 0.0;
    let mut prev_own_core_hp: f32 = 500.0;
    let mut prev_own_working: f32 = 0.0;
    let team_idx = team as usize;
    let enemy_idx = enemy_team as usize;

    let turns = get_repeated_submsg(&top, 3, data);
    let n_turns = turns.len();

    // Pre-allocate contiguous buffers
    let mut states = vec![0.0f32; n_turns * state_stride];
    let mut scalars = vec![0.0f32; n_turns * N_SCALARS];
    let mut working_harvesters = vec![0i32; n_turns * 2];
    let mut titanium_delivered = vec![0i32; n_turns * 2];

    for (turn_idx, turn_data) in turns.iter().enumerate() {
        // ── Snapshot pre-turn state into contiguous buffer ──
        let state = &mut states[turn_idx * state_stride..(turn_idx + 1) * state_stride];

        // Environment channels (static)
        for y in 0..map_h {
            for x in 0..map_w {
                let base = y * map_w + x;
                let env_val = map.env_grid[y * map_w + x] as usize;
                if env_val <= 3 {
                    state[env_val * plane_size + base] = 1.0;
                }
            }
        }

        // Entity channels
        let mut n_own_builders = 0i32;
        let mut n_enemy_builders = 0i32;

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
                    state[15 * plane_size + base] = 1.0;
                    state[16 * plane_size + base] = sign;
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
                    state[12 * plane_size + base] = sign;
                    let d = ent.direction as usize;
                    if d < 9 {
                        let (dx, dy) = DIR_DELTA[d];
                        state[13 * plane_size + base] = dx;
                        state[14 * plane_size + base] = dy;
                    }
                    if kind.is_storing() && ent.stored > 0 {
                        state[17 * plane_size + base] = 1.0;
                    }
                    if kind == EntityKind::Harvester && ent.cooldown > 0 {
                        state[18 * plane_size + base] = 1.0;
                    }
                    if kind == EntityKind::Core {
                        core_hp[ent.team as usize] = ent.hp;
                    }
                }
            }
        }

        // Scalars into contiguous buffer
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

        // Delta and working harvester scalars
        let cur_own_ti = own_res.titanium as f32;
        let cur_own_ax = own_res.axionite as f32;
        let cur_own_core_hp = core_hp[team_idx] as f32;
        sc[12] = prev_own_working / 20.0;
        sc[13] = (cur_own_ti - prev_own_ti) / 500.0;
        sc[14] = (cur_own_ax - prev_own_ax) / 500.0;
        sc[15] = (cur_own_core_hp - prev_own_core_hp) / 500.0;
        sc[16] = 0.0; // filled after working harvesters are counted
        prev_own_ti = cur_own_ti;
        prev_own_ax = cur_own_ax;
        prev_own_core_hp = cur_own_core_hp;

        // ── Decrement harvester cooldowns ──
        for ent in entities.values_mut() {
            if ent.kind == EntityKind::Harvester && ent.cooldown > 0 {
                ent.cooldown -= 1;
            }
        }

        // ── Process updates to advance state ──
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
                        entities.insert(eid, ent);
                    }
                    UpdateEvent::MoveBuilderBot { id, to } => {
                        if let Some(ent) = entities.get_mut(&id) {
                            if ent.kind == EntityKind::Harvester {
                                harv_positions.remove(&(ent.x, ent.y));
                                harv_positions.insert((to.x, to.y), id);
                            }
                            ent.x = to.x;
                            ent.y = to.y;
                        }
                    }
                    UpdateEvent::RemoveEntity { id } => {
                        if let Some(ent) = entities.remove(&id) {
                            if ent.kind == EntityKind::Harvester {
                                harv_positions.remove(&(ent.x, ent.y));
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
                        let pos_to_eid: HashMap<(i32, i32), i32> = entities
                            .iter()
                            .map(|(eid, e)| ((e.x, e.y), *eid))
                            .collect();

                        for (from, to) in &moves {
                            if let Some(&eid) = harv_positions.get(&(from.x, from.y)) {
                                if let Some(ent) = entities.get_mut(&eid) {
                                    ent.cooldown = 3;
                                }
                            }
                            let mut moved_res = 1u8;
                            if let Some(&src_id) = pos_to_eid.get(&(from.x, from.y)) {
                                if let Some(src) = entities.get_mut(&src_id) {
                                    if src.kind.is_storing() {
                                        moved_res = if src.stored > 0 { src.stored } else { 1 };
                                        src.stored = 0;
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

        // ── Count working harvesters per team ──
        let wh = &mut working_harvesters[turn_idx * 2..turn_idx * 2 + 2];
        for &eid in harv_positions.values() {
            if let Some(ent) = entities.get(&eid) {
                if ent.cooldown > 0 {
                    wh[ent.team as usize] += 1;
                }
            }
        }
        // Write working harvester delta scalar now that wh is computed
        let cur_own_working = wh[team_idx] as f32;
        let sc = &mut scalars[turn_idx * N_SCALARS..(turn_idx + 1) * N_SCALARS];
        sc[12] = cur_own_working / 20.0;
        sc[16] = (cur_own_working - prev_own_working) / 20.0;
        prev_own_working = cur_own_working;

        // ── Titanium delivered delta per team ──
        let td = &mut titanium_delivered[turn_idx * 2..turn_idx * 2 + 2];
        for t in 0..2 {
            let cur = player_res[t].titanium_collected;
            td[t] = cur - prev_ti_collected[t];
            prev_ti_collected[t] = cur;
        }
    }

    ReplayData {
        states,
        scalars,
        working_harvesters,
        titanium_delivered,
        map_w,
        map_h,
        n_turns,
        winner,
    }
}

// ── Python bindings ───────────────────────────────────────────────────────

/// Parse a replay into contiguous buffers for zero-copy tensor construction.
///
/// Returns dict with:
///   states: bytes — contiguous f32[n_turns, N_CHANNELS, map_h, map_w]
///   scalars: bytes — contiguous f32[n_turns, N_SCALARS]
///   working_harvesters: bytes — contiguous i32[n_turns, 2] (team_a, team_b)
///   titanium_delivered: bytes — contiguous i32[n_turns, 2] (per-turn delta)
///   map_w, map_h, n_turns, n_channels, n_scalars: int
///   state_stride: int — floats per turn in states buffer
///   winner: Optional[int]
#[pyfunction]
#[pyo3(signature = (path, team=0))]
fn parse_replay_full<'py>(
    py: Python<'py>,
    path: &str,
    team: u8,
) -> PyResult<PyObject> {
    let data = fs::read(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e)))?;

    let rd = process_replay(&data, team);
    let result = PyDict::new(py);

    // Single contiguous bytes for each buffer — torch.frombuffer directly
    let states_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(rd.states.as_ptr() as *const u8, rd.states.len() * 4)
    };
    result.set_item("states", PyBytes::new(py, states_bytes))?;

    let scalars_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(rd.scalars.as_ptr() as *const u8, rd.scalars.len() * 4)
    };
    result.set_item("scalars", PyBytes::new(py, scalars_bytes))?;

    let wh_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            rd.working_harvesters.as_ptr() as *const u8,
            rd.working_harvesters.len() * 4,
        )
    };
    result.set_item("working_harvesters", PyBytes::new(py, wh_bytes))?;

    let td_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            rd.titanium_delivered.as_ptr() as *const u8,
            rd.titanium_delivered.len() * 4,
        )
    };
    result.set_item("titanium_delivered", PyBytes::new(py, td_bytes))?;

    result.set_item("map_w", rd.map_w)?;
    result.set_item("map_h", rd.map_h)?;
    result.set_item("n_turns", rd.n_turns)?;
    result.set_item("n_channels", N_CHANNELS)?;
    result.set_item("n_scalars", N_SCALARS)?;
    result.set_item("state_stride", N_CHANNELS * rd.map_h * rd.map_w)?;
    result.set_item("winner", rd.winner)?;

    Ok(result.into())
}

/// Minimal parse: just extract working harvesters and winner.
#[pyfunction]
fn parse_replay_working(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let data = fs::read(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e)))?;

    let rd = process_replay(&data, 0);
    let result = PyDict::new(py);

    let wh_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            rd.working_harvesters.as_ptr() as *const u8,
            rd.working_harvesters.len() * 4,
        )
    };
    result.set_item("working_harvesters", PyBytes::new(py, wh_bytes))?;
    result.set_item("n_turns", rd.n_turns)?;
    result.set_item("winner", rd.winner)?;

    Ok(result.into())
}

/// Parse replay and return full Python dict (compatible with Python replay_parser.parse_replay).
#[pyfunction]
fn parse_replay(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let data = fs::read(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e)))?;

    let top = decode_fields(&data, 0, data.len());

    let result = PyDict::new(py);

    // Winner
    if let Some(w) = get_varint(&top, 4) {
        result.set_item("winner", w)?;
    }

    // Map
    if let Some(map_data) = get_submsg(&top, 1, &data) {
        let map = decode_map(map_data);
        let map_dict = PyDict::new(py);
        map_dict.set_item("width", map.width)?;
        map_dict.set_item("height", map.height)?;

        // Rows as list of dicts with tiles
        let rows_list = PyList::empty(py);
        for y in 0..map.height {
            let row_dict = PyDict::new(py);
            let tiles: Vec<u8> = (0..map.width)
                .map(|x| map.env_grid[y * map.width + x])
                .collect();
            row_dict.set_item("tiles", tiles)?;
            rows_list.append(row_dict)?;
        }
        map_dict.set_item("rows", rows_list)?;

        // Cores
        let cores_list = PyList::empty(py);
        for (id, team, pos) in &map.cores {
            let core_dict = PyDict::new(py);
            core_dict.set_item("id", *id)?;
            core_dict.set_item("team", *team)?;
            let pos_dict = PyDict::new(py);
            pos_dict.set_item("x", pos.x)?;
            pos_dict.set_item("y", pos.y)?;
            core_dict.set_item("position", pos_dict)?;
            cores_list.append(core_dict)?;
        }
        map_dict.set_item("cores", cores_list)?;

        result.set_item("map", map_dict)?;
    }

    // Turns - decode into same dict structure as Python version
    let turns_data = get_repeated_submsg(&top, 3, &data);
    let turns_list = PyList::empty(py);
    for turn_data in &turns_data {
        let turn_fields = decode_fields(turn_data, 0, turn_data.len());
        let updates_data = get_repeated_submsg(&turn_fields, 1, turn_data);
        let updates_list = PyList::empty(py);

        for update_data in &updates_data {
            let update_dict = PyDict::new(py);
            let fields = decode_fields(update_data, 0, update_data.len());

            // We decode each update type into the same dict format as Python
            // PlaceEntity (field 1)
            if let Some(sub) = get_submsg(&fields, 1, update_data) {
                let pf = decode_fields(sub, 0, sub.len());
                if let Some(ent_data) = get_submsg(&pf, 1, sub) {
                    let ent = decode_entity(ent_data, update_data);
                    let ent_dict = entity_to_pydict(py, &ent)?;
                    let pe_dict = PyDict::new(py);
                    pe_dict.set_item("entity", ent_dict)?;
                    update_dict.set_item("placeEntity", pe_dict)?;
                }
            }
            // MoveBuilderBot (field 2)
            else if let Some(sub) = get_submsg(&fields, 2, update_data) {
                let mf = decode_fields(sub, 0, sub.len());
                let mv_dict = PyDict::new(py);
                mv_dict.set_item("id", get_i32(&mf, 1))?;
                if let Some(to_data) = get_submsg(&mf, 2, sub) {
                    let to = decode_pos(to_data);
                    let to_dict = PyDict::new(py);
                    to_dict.set_item("x", to.x)?;
                    to_dict.set_item("y", to.y)?;
                    mv_dict.set_item("to", to_dict)?;
                }
                update_dict.set_item("moveBuilderBot", mv_dict)?;
            }
            // RemoveEntity (field 3)
            else if let Some(sub) = get_submsg(&fields, 3, update_data) {
                let rf = decode_fields(sub, 0, sub.len());
                let re_dict = PyDict::new(py);
                re_dict.set_item("id", get_i32(&rf, 1))?;
                update_dict.set_item("removeEntity", re_dict)?;
            }
            // DistributeResources (field 4)
            else if let Some(sub) = get_submsg(&fields, 4, update_data) {
                let df = decode_fields(sub, 0, sub.len());
                let dr_dict = PyDict::new(py);
                let moves_list = PyList::empty(py);
                for mv_data in get_repeated_submsg(&df, 1, sub) {
                    let mvf = decode_fields(mv_data, 0, mv_data.len());
                    let mv_dict = PyDict::new(py);
                    if let Some(from_data) = get_submsg(&mvf, 1, mv_data) {
                        let from = decode_pos(from_data);
                        let d = PyDict::new(py);
                        d.set_item("x", from.x)?;
                        d.set_item("y", from.y)?;
                        mv_dict.set_item("from", d)?;
                    }
                    if let Some(to_data) = get_submsg(&mvf, 2, mv_data) {
                        let to = decode_pos(to_data);
                        let d = PyDict::new(py);
                        d.set_item("x", to.x)?;
                        d.set_item("y", to.y)?;
                        mv_dict.set_item("to", d)?;
                    }
                    moves_list.append(mv_dict)?;
                }
                dr_dict.set_item("moves", moves_list)?;
                update_dict.set_item("distributeResources", dr_dict)?;
            }
            // UpdateHp (field 5)
            else if let Some(sub) = get_submsg(&fields, 5, update_data) {
                let hf = decode_fields(sub, 0, sub.len());
                let hp_dict = PyDict::new(py);
                hp_dict.set_item("id", get_i32(&hf, 1))?;
                hp_dict.set_item("delta", get_i32(&hf, 2))?;
                update_dict.set_item("updateHp", hp_dict)?;
            }
            // UpdatePlayers (field 6)
            else if let Some(sub) = get_submsg(&fields, 6, update_data) {
                let uf = decode_fields(sub, 0, sub.len());
                let up_dict = PyDict::new(py);
                if let Some(players_data) = get_submsg(&uf, 1, sub) {
                    let pf = decode_fields(players_data, 0, players_data.len());
                    let players_dict = PyDict::new(py);
                    if let Some(a_data) = get_submsg(&pf, 1, players_data) {
                        let a = decode_player_res(a_data);
                        let d = PyDict::new(py);
                        d.set_item("titanium", a.titanium)?;
                        d.set_item("axionite", a.axionite)?;
                        d.set_item("resourcesCollected", a.resources_collected)?;
                        d.set_item("titaniumCollected", a.titanium_collected)?;
                        d.set_item("axioniteCollected", a.axionite_collected)?;
                        players_dict.set_item("a", d)?;
                    }
                    if let Some(b_data) = get_submsg(&pf, 2, players_data) {
                        let b = decode_player_res(b_data);
                        let d = PyDict::new(py);
                        d.set_item("titanium", b.titanium)?;
                        d.set_item("axionite", b.axionite)?;
                        d.set_item("resourcesCollected", b.resources_collected)?;
                        d.set_item("titaniumCollected", b.titanium_collected)?;
                        d.set_item("axioniteCollected", b.axionite_collected)?;
                        players_dict.set_item("b", d)?;
                    }
                    up_dict.set_item("players", players_dict)?;
                }
                update_dict.set_item("updatePlayers", up_dict)?;
            }
            // SetActionCooldown (field 7)
            else if let Some(sub) = get_submsg(&fields, 7, update_data) {
                let cf = decode_fields(sub, 0, sub.len());
                let cd_dict = PyDict::new(py);
                cd_dict.set_item("id", get_i32(&cf, 1))?;
                cd_dict.set_item("value", get_i32(&cf, 2))?;
                update_dict.set_item("setActionCooldown", cd_dict)?;
            }
            // SetMoveCooldown (field 8)
            else if let Some(sub) = get_submsg(&fields, 8, update_data) {
                let cf = decode_fields(sub, 0, sub.len());
                let cd_dict = PyDict::new(py);
                cd_dict.set_item("id", get_i32(&cf, 1))?;
                cd_dict.set_item("value", get_i32(&cf, 2))?;
                update_dict.set_item("setMoveCooldown", cd_dict)?;
            }

            updates_list.append(update_dict)?;
        }

        let turn_dict = PyDict::new(py);
        turn_dict.set_item("updates", updates_list)?;
        turns_list.append(turn_dict)?;
    }
    result.set_item("turns", turns_list)?;

    Ok(result.into())
}

fn entity_to_pydict(py: Python<'_>, ent: &Entity) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("id", ent.id)?;
    d.set_item("team", ent.team)?;
    let pos = PyDict::new(py);
    pos.set_item("x", ent.x)?;
    pos.set_item("y", ent.y)?;
    d.set_item("position", pos)?;
    d.set_item("hp", ent.hp)?;
    d.set_item("maxHp", ent.max_hp)?;

    // Kind-specific submessage
    let kind_name = match ent.kind {
        EntityKind::BuilderBot => "builderBot",
        EntityKind::Conveyor => "conveyor",
        EntityKind::Splitter => "splitter",
        EntityKind::ArmouredConveyor => "armouredConveyor",
        EntityKind::Bridge => "bridge",
        EntityKind::Harvester => "harvester",
        EntityKind::Foundry => "foundry",
        EntityKind::Road => "road",
        EntityKind::Barrier => "barrier",
        EntityKind::Marker => "marker",
        EntityKind::Core => "core",
        EntityKind::Gunner => "gunner",
        EntityKind::Sentinel => "sentinel",
        EntityKind::Breach => "breach",
        EntityKind::Launcher => "launcher",
    };

    let kind_dict = PyDict::new(py);
    match ent.kind {
        EntityKind::BuilderBot => {
            kind_dict.set_item("actionCooldown", ent.action_cooldown)?;
            kind_dict.set_item("moveCooldown", ent.move_cooldown)?;
        }
        EntityKind::Conveyor | EntityKind::Splitter | EntityKind::ArmouredConveyor => {
            kind_dict.set_item("direction", ent.direction)?;
            kind_dict.set_item("stored", ent.stored)?;
        }
        EntityKind::Harvester => {
            kind_dict.set_item("cooldown", ent.cooldown)?;
            kind_dict.set_item("resourceType", ent.stored)?;
        }
        EntityKind::Core => {
            kind_dict.set_item("actionCooldown", ent.action_cooldown)?;
        }
        EntityKind::Gunner | EntityKind::Sentinel | EntityKind::Breach | EntityKind::Launcher => {
            kind_dict.set_item("direction", ent.direction)?;
            kind_dict.set_item("ammoType", ent.stored)?;
        }
        _ => {}
    }
    d.set_item(kind_name, kind_dict)?;

    Ok(d.into())
}

/// Module definition
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_replay_full, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replay_working, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replay, m)?)?;
    Ok(())
}
