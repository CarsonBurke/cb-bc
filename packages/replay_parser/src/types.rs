//! Domain types for the Battlecode replay format.

use crate::protobuf::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum EntityKind {
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
    pub fn from_field_id(id: u32) -> Option<Self> {
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

    pub fn is_storing(self) -> bool {
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

    /// Map channel offset for building types (channels 4-11).
    pub fn building_channel(self) -> Option<usize> {
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

    pub fn proto_name(self) -> &'static str {
        match self {
            Self::BuilderBot => "builderBot",
            Self::Conveyor => "conveyor",
            Self::Splitter => "splitter",
            Self::ArmouredConveyor => "armouredConveyor",
            Self::Bridge => "bridge",
            Self::Harvester => "harvester",
            Self::Foundry => "foundry",
            Self::Road => "road",
            Self::Barrier => "barrier",
            Self::Marker => "marker",
            Self::Core => "core",
            Self::Gunner => "gunner",
            Self::Sentinel => "sentinel",
            Self::Breach => "breach",
            Self::Launcher => "launcher",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: i32,
    pub kind: EntityKind,
    pub team: u8,
    pub x: i32,
    pub y: i32,
    pub hp: i32,
    pub max_hp: i32,
    pub direction: u8,
    pub stored: u8,
    pub cooldown: i32,
    pub action_cooldown: i32,
    pub move_cooldown: i32,
    pub target_x: i32,
    pub target_y: i32,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerRes {
    pub titanium: i32,
    pub axionite: i32,
    pub resources_collected: i32,
    pub titanium_collected: i32,
    pub axionite_collected: i32,
}

#[derive(Clone, Debug)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

pub struct MapData {
    pub width: usize,
    pub height: usize,
    pub env_grid: Vec<u8>,
    pub cores: Vec<(i32, u8, Pos)>,
}

pub enum UpdateEvent {
    PlaceEntity(Entity),
    MoveBuilderBot { id: i32, to: Pos },
    RemoveEntity { id: i32 },
    DistributeResources(Vec<(Pos, Pos)>),
    UpdateHp { id: i32, delta: i32 },
    UpdatePlayers { a: Option<PlayerRes>, b: Option<PlayerRes> },
    SetActionCooldown { id: i32, value: i32 },
    SetMoveCooldown { id: i32, value: i32 },
}

pub const DIR_DELTA: [(f32, f32); 9] = [
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

// ── Protobuf decoding for domain types ───────────────────────────────────

/// Decode a Pos directly from wire bytes without allocating a field Vec.
#[inline]
pub fn decode_pos(data: &[u8]) -> Pos {
    let mut x = 0i32;
    let mut y = 0i32;
    let mut pos = 0;
    let end = data.len();
    while pos < end {
        let (tag, new_pos) = decode_varint(data, pos);
        pos = new_pos;
        let field_id = (tag >> 3) as u32;
        let wire_type = (tag & 0x07) as u8;
        if wire_type == VARINT {
            let (val, new_pos) = decode_varint(data, pos);
            pos = new_pos;
            match field_id {
                1 => x = val as i32,
                2 => y = val as i32,
                _ => {}
            }
        } else {
            break;
        }
    }
    Pos { x, y }
}

pub fn decode_player_res(data: &[u8]) -> PlayerRes {
    let fields = decode_fields(data, 0, data.len());
    PlayerRes {
        titanium: get_i32(&fields, 1),
        axionite: get_i32(&fields, 2),
        resources_collected: get_i32(&fields, 3),
        titanium_collected: get_i32(&fields, 4),
        axionite_collected: get_i32(&fields, 5),
    }
}

pub fn decode_entity(data: &[u8]) -> Entity {
    let fields = decode_fields(data, 0, data.len());
    let id = get_i32(&fields, 1);
    let team = get_varint(&fields, 2).unwrap_or(0) as u8;
    let pos = get_submsg(&fields, 3, data)
        .map(decode_pos)
        .unwrap_or(Pos { x: 0, y: 0 });
    let hp = get_i32(&fields, 4);
    let max_hp = get_i32(&fields, 5);

    let mut kind = EntityKind::BuilderBot;
    let mut direction: u8 = 0;
    let mut stored: u8 = 0;
    let mut cooldown: i32 = 0;
    let mut target_x: i32 = 0;
    let mut target_y: i32 = 0;

    for fid in 10..=24 {
        if let Some(kind_val) = EntityKind::from_field_id(fid) {
            if let Some(sub) = get_submsg(&fields, fid, data) {
                kind = kind_val;
                let sub_fields = decode_fields(sub, 0, sub.len());
                if kind_val == EntityKind::Bridge {
                    if let Some(target_sub) = get_submsg(&sub_fields, 1, sub) {
                        let target_pos = decode_pos(target_sub);
                        target_x = target_pos.x;
                        target_y = target_pos.y;
                    }
                    stored = get_varint(&sub_fields, 2).unwrap_or(0) as u8;
                } else if kind_val == EntityKind::Harvester {
                    cooldown = get_i32(&sub_fields, 1);
                    stored = get_varint(&sub_fields, 2).unwrap_or(0) as u8;
                } else {
                    direction = get_varint(&sub_fields, 1).unwrap_or(0) as u8;
                    stored = get_varint(&sub_fields, 2).unwrap_or(0) as u8;
                }
                break;
            }
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
        target_x,
        target_y,
    }
}

pub fn decode_map(data: &[u8]) -> MapData {
    let fields = decode_fields(data, 0, data.len());
    let width = get_i32(&fields, 1) as usize;
    let height = get_i32(&fields, 2) as usize;

    let mut env_grid = vec![0u8; width * height];

    let rows = get_repeated_submsg(&fields, 3, data);
    for (y, row_data) in rows.iter().enumerate() {
        if y >= height {
            break;
        }
        let row_fields = decode_fields(row_data, 0, row_data.len());
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

pub fn decode_update(data: &[u8]) -> Option<UpdateEvent> {
    let fields = decode_fields(data, 0, data.len());

    if let Some(sub) = get_submsg(&fields, 1, data) {
        let pf = decode_fields(sub, 0, sub.len());
        if let Some(ent_data) = get_submsg(&pf, 1, sub) {
            return Some(UpdateEvent::PlaceEntity(decode_entity(ent_data)));
        }
    }

    if let Some(sub) = get_submsg(&fields, 2, data) {
        let mf = decode_fields(sub, 0, sub.len());
        let id = get_i32(&mf, 1);
        let to = get_submsg(&mf, 2, sub)
            .map(decode_pos)
            .unwrap_or(Pos { x: 0, y: 0 });
        return Some(UpdateEvent::MoveBuilderBot { id, to });
    }

    if let Some(sub) = get_submsg(&fields, 3, data) {
        let rf = decode_fields(sub, 0, sub.len());
        let id = get_i32(&rf, 1);
        return Some(UpdateEvent::RemoveEntity { id });
    }

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

    if let Some(sub) = get_submsg(&fields, 5, data) {
        let hf = decode_fields(sub, 0, sub.len());
        let id = get_i32(&hf, 1);
        let delta = get_i32(&hf, 2);
        return Some(UpdateEvent::UpdateHp { id, delta });
    }

    if let Some(sub) = get_submsg(&fields, 6, data) {
        let uf = decode_fields(sub, 0, sub.len());
        if let Some(players_data) = get_submsg(&uf, 1, sub) {
            let pf = decode_fields(players_data, 0, players_data.len());
            let a = get_submsg(&pf, 1, players_data).map(decode_player_res);
            let b = get_submsg(&pf, 2, players_data).map(decode_player_res);
            return Some(UpdateEvent::UpdatePlayers { a, b });
        }
    }

    if let Some(sub) = get_submsg(&fields, 7, data) {
        let cf = decode_fields(sub, 0, sub.len());
        return Some(UpdateEvent::SetActionCooldown {
            id: get_i32(&cf, 1),
            value: get_i32(&cf, 2),
        });
    }

    if let Some(sub) = get_submsg(&fields, 8, data) {
        let cf = decode_fields(sub, 0, sub.len());
        return Some(UpdateEvent::SetMoveCooldown {
            id: get_i32(&cf, 1),
            value: get_i32(&cf, 2),
        });
    }

    None
}
