use crate::common::game_constants::{
    ACTION_RADIUS_SQ, ARMOURED_CONVEYOR_MAX_HP, BARRIER_MAX_HP, BREACH_MAX_HP,
    BREACH_VISION_RADIUS_SQ, BRIDGE_MAX_HP, BUILDER_BOT_VISION_RADIUS_SQ, CONVEYOR_MAX_HP,
    CORE_ACTION_RADIUS_SQ, CORE_VISION_RADIUS_SQ, FOUNDRY_MAX_HP, GUNNER_MAX_HP,
    GUNNER_VISION_RADIUS_SQ, HARVESTER_MAX_HP, LAUNCHER_MAX_HP, LAUNCHER_VISION_RADIUS_SQ,
    MARKER_MAX_HP, ROAD_MAX_HP, SENTINEL_MAX_HP, SENTINEL_VISION_RADIUS_SQ, SPLITTER_MAX_HP,
    STACK_SIZE,
};
use crate::common::{Direction, Environment, Pos, ResourceType, Team};
use paste::paste;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};


macro_rules! impl_derefs {
    ($ty:ty, $target:ident) => {
        paste! {
            impl Deref for $ty {
                type Target = [< $target:camel Base>];

                fn deref(&self) -> &Self::Target {
                    &self.$target
                }
            }

            impl DerefMut for $ty {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.$target
                }
            }
        }
    };
}

#[derive(Clone, Debug)]
pub struct EntityBase {
    pub id: i32,
    pub team: Team,
    pub position: Pos,
    pub hp: i32,
    pub max_hp: i32,
}

#[derive(Clone, Debug)]
pub enum Entity {
    BuilderBot(BuilderBot),
    Conveyor(Conveyor),
    Splitter(Splitter),
    ArmouredConveyor(ArmouredConveyor),
    Bridge(Bridge),
    Harvester(Harvester),
    Foundry(Foundry),
    Road(Road),
    Barrier(Barrier),
    Marker(Marker),
    Core(Core),
    Gunner(Gunner),
    Sentinel(Sentinel),
    Breach(Breach),
    Launcher(Launcher),
}

impl Deref for Entity {
    type Target = EntityBase;

    fn deref(&self) -> &Self::Target {
        match self {
            Entity::BuilderBot(bot) => bot,
            Entity::Conveyor(c) => c,
            Entity::Splitter(c) => c,
            Entity::ArmouredConveyor(c) => c,
            Entity::Bridge(b) => b,
            Entity::Harvester(h) => h,
            Entity::Foundry(f) => f,
            Entity::Road(r) => r,
            Entity::Barrier(b) => b,
            Entity::Marker(m) => m,
            Entity::Core(c) => c,
            Entity::Gunner(t) => t,
            Entity::Sentinel(t) => t,
            Entity::Breach(t) => t,
            Entity::Launcher(t) => t,
        }
    }
}

impl DerefMut for Entity {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Entity::BuilderBot(bot) => bot,
            Entity::Conveyor(c) => c,
            Entity::Splitter(c) => c,
            Entity::ArmouredConveyor(c) => c,
            Entity::Bridge(b) => b,
            Entity::Harvester(h) => h,
            Entity::Foundry(f) => f,
            Entity::Road(r) => r,
            Entity::Barrier(b) => b,
            Entity::Marker(m) => m,
            Entity::Core(c) => c,
            Entity::Gunner(t) => t,
            Entity::Sentinel(t) => t,
            Entity::Breach(t) => t,
            Entity::Launcher(t) => t,
        }
    }
}

impl Entity {
    pub fn as_unit(&self) -> Option<Unit<'_>> {
        match self {
            Entity::BuilderBot(bot) => Some(Unit::BuilderBot(bot)),
            Entity::Core(core) => Some(Unit::Core(core)),
            Entity::Gunner(t) => Some(Unit::Gunner(t)),
            Entity::Sentinel(t) => Some(Unit::Sentinel(t)),
            Entity::Breach(t) => Some(Unit::Breach(t)),
            Entity::Launcher(t) => Some(Unit::Launcher(t)),
            _ => None,
        }
    }

    pub fn as_unit_mut(&mut self) -> Option<UnitMut<'_>> {
        match self {
            Entity::BuilderBot(bot) => Some(UnitMut::BuilderBot(bot)),
            Entity::Core(core) => Some(UnitMut::Core(core)),
            Entity::Gunner(t) => Some(UnitMut::Gunner(t)),
            Entity::Sentinel(t) => Some(UnitMut::Sentinel(t)),
            Entity::Breach(t) => Some(UnitMut::Breach(t)),
            Entity::Launcher(t) => Some(UnitMut::Launcher(t)),
            _ => None,
        }
    }

    pub fn as_turret(&self) -> Option<Turret<'_>> {
        match self {
            Entity::Gunner(t) => Some(Turret::Gunner(t)),
            Entity::Sentinel(t) => Some(Turret::Sentinel(t)),
            Entity::Breach(t) => Some(Turret::Breach(t)),
            Entity::Launcher(t) => Some(Turret::Launcher(t)),
            _ => None,
        }
    }

    pub fn as_turret_mut(&mut self) -> Option<TurretMut<'_>> {
        match self {
            Entity::Gunner(t) => Some(TurretMut::Gunner(t)),
            Entity::Sentinel(t) => Some(TurretMut::Sentinel(t)),
            Entity::Breach(t) => Some(TurretMut::Breach(t)),
            Entity::Launcher(t) => Some(TurretMut::Launcher(t)),
            _ => None,
        }
    }

    pub fn as_building(&self) -> Option<Building<'_>> {
        match self {
            Entity::BuilderBot(_) => None,
            Entity::Conveyor(c) => Some(Building::Conveyor(c)),
            Entity::Splitter(c) => Some(Building::Splitter(c)),
            Entity::ArmouredConveyor(c) => Some(Building::ArmouredConveyor(c)),
            Entity::Bridge(b) => Some(Building::Bridge(b)),
            Entity::Harvester(h) => Some(Building::Harvester(h)),
            Entity::Foundry(f) => Some(Building::Foundry(f)),
            Entity::Road(r) => Some(Building::Road(r)),
            Entity::Barrier(b) => Some(Building::Barrier(b)),
            Entity::Marker(m) => Some(Building::Marker(m)),
            Entity::Core(c) => Some(Building::Core(c)),
            Entity::Gunner(t) => Some(Building::Gunner(t)),
            Entity::Sentinel(t) => Some(Building::Sentinel(t)),
            Entity::Breach(t) => Some(Building::Breach(t)),
            Entity::Launcher(t) => Some(Building::Launcher(t)),
        }
    }

    pub fn as_building_mut(&mut self) -> Option<BuildingMut<'_>> {
        match self {
            Entity::BuilderBot(_) => None,
            Entity::Conveyor(c) => Some(BuildingMut::Conveyor(c)),
            Entity::Splitter(c) => Some(BuildingMut::Splitter(c)),
            Entity::ArmouredConveyor(c) => Some(BuildingMut::ArmouredConveyor(c)),
            Entity::Bridge(b) => Some(BuildingMut::Bridge(b)),
            Entity::Harvester(h) => Some(BuildingMut::Harvester(h)),
            Entity::Foundry(f) => Some(BuildingMut::Foundry(f)),
            Entity::Road(r) => Some(BuildingMut::Road(r)),
            Entity::Barrier(b) => Some(BuildingMut::Barrier(b)),
            Entity::Marker(m) => Some(BuildingMut::Marker(m)),
            Entity::Core(c) => Some(BuildingMut::Core(c)),
            Entity::Gunner(t) => Some(BuildingMut::Gunner(t)),
            Entity::Sentinel(t) => Some(BuildingMut::Sentinel(t)),
            Entity::Breach(t) => Some(BuildingMut::Breach(t)),
            Entity::Launcher(t) => Some(BuildingMut::Launcher(t)),
        }
    }

    pub fn scale_contribution(&self) -> i32 {
        match self {
            Entity::Conveyor(_)
            | Entity::Splitter(_)
            | Entity::ArmouredConveyor(_)
            | Entity::Bridge(_) => 10,
            Entity::Road(_) => 5,
            Entity::Barrier(_) => 10,
            Entity::Harvester(_)
            | Entity::Gunner(_)
            | Entity::Sentinel(_)
            | Entity::Breach(_)
            | Entity::Launcher(_) => 100,
            Entity::Foundry(_) => 1000,
            _ => 0,
        }
    }

    pub fn resource_to_feed(&self) -> Option<ResourceType> {
        match self {
            Entity::Conveyor(c) => c.stored,
            Entity::Splitter(s) => s.stored,
            Entity::ArmouredConveyor(c) => c.stored,
            Entity::Bridge(b) => b.stored,
            Entity::Harvester(h) => {
                if h.cooldown == 0 {
                    Some(h.resource_type)
                } else {
                    None
                }
            }
            Entity::Foundry(f) => {
                if f.stored == Some(ResourceType::RefinedAxionite) {
                    Some(ResourceType::RefinedAxionite)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn output_targets(&self) -> Vec<Pos> {
        match self {
            Entity::Conveyor(c) => vec![c.position + c.direction],
            Entity::ArmouredConveyor(c) => vec![c.position + c.direction],
            Entity::Bridge(b) => vec![b.target],
            Entity::Splitter(s) => {
                let excluded = s.direction.opposite();
                let dirs = [
                    Direction::North,
                    Direction::East,
                    Direction::South,
                    Direction::West,
                ];
                dirs.iter()
                    .filter(|d| **d != excluded)
                    .map(|d| s.position + *d)
                    .collect()
            }
            Entity::Harvester(h) => {
                let dirs = [
                    Direction::North,
                    Direction::East,
                    Direction::South,
                    Direction::West,
                ];
                dirs.iter().map(|d| h.position + *d).collect()
            }
            Entity::Foundry(f) => {
                let dirs = [
                    Direction::North,
                    Direction::East,
                    Direction::South,
                    Direction::West,
                ];
                dirs.iter().map(|d| f.position + *d).collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn consume_feed(&mut self) {
        match self {
            Entity::Conveyor(c) => c.stored = None,
            Entity::Splitter(s) => s.stored = None,
            Entity::ArmouredConveyor(c) => c.stored = None,
            Entity::Bridge(b) => b.stored = None,
            Entity::Harvester(h) => h.cooldown = 4,
            Entity::Foundry(f) => f.stored = None,
            _ => panic!("consume_feed called on non-feeder entity"),
        }
    }

    pub fn can_accept_from(&self, resource: ResourceType, source_pos: Pos, source_is_bridge: bool) -> bool {
        match self {
            Entity::Conveyor(c) => {
                // Rejects input from its output direction.
                (source_is_bridge || source_pos != c.position + c.direction) && c.stored.is_none()
            }
            Entity::Splitter(s) => {
                // Splitter only accepts input from its entry side (direction.opposite()).
                let input_pos = s.position + s.direction.opposite();
                if !source_is_bridge && source_pos != input_pos {
                    return false;
                }
                s.stored.is_none()
            }
            Entity::ArmouredConveyor(c) => {
                // Rejects input from its output direction.
                (source_is_bridge || source_pos != c.position + c.direction) && c.stored.is_none()
            }
            Entity::Bridge(b) => b.stored.is_none(),
            Entity::Foundry(f) => {
                matches!(
                    (resource, f.stored),
                    (
                        ResourceType::Titanium,
                        Some(ResourceType::RawAxionite) | None
                    ) | (
                        ResourceType::RawAxionite,
                        Some(ResourceType::Titanium) | None
                    )
                )
            }
            Entity::Core(_) => true,
            Entity::Gunner(t) => {
                t.ammo_amount == 0 && (source_is_bridge || source_pos != t.position + t.direction)
            }
            Entity::Sentinel(t) => {
                t.ammo_amount == 0 && (source_is_bridge || source_pos != t.position + t.direction)
            }
            Entity::Breach(t) => {
                t.ammo_amount == 0
                    && resource == ResourceType::RefinedAxionite
                    && (source_is_bridge || source_pos != t.position + t.direction)
            }

            Entity::Launcher(_) => false,
            _ => false,
        }
    }

    pub fn receive_resource(&mut self, resource: ResourceType) {
        match self {
            Entity::Conveyor(c) => c.stored = Some(resource),
            Entity::Splitter(s) => s.stored = Some(resource),
            Entity::ArmouredConveyor(c) => c.stored = Some(resource),
            Entity::Bridge(b) => b.stored = Some(resource),
            Entity::Core(core) => core.received.push(resource),
            Entity::Gunner(t) => t.receive_ammo(resource),
            Entity::Sentinel(t) => t.receive_ammo(resource),
            Entity::Breach(t) => t.receive_ammo(resource),
            Entity::Foundry(f) => match (resource, f.stored) {
                (r @ (ResourceType::Titanium | ResourceType::RawAxionite), None) => {
                    f.stored = Some(r)
                }
                (ResourceType::Titanium, Some(ResourceType::RawAxionite))
                | (ResourceType::RawAxionite, Some(ResourceType::Titanium)) => {
                    f.stored = Some(ResourceType::RefinedAxionite)
                }
                _ => panic!(
                    "foundry received unexpected resource {:?} with stored {:?}",
                    resource, f.stored
                ),
            },
            _ => panic!("receive_resource called on non-receiver entity"),
        }
    }
}

macro_rules! define_category {
    ($name:ident : $parent:ident [$base_type:ty] ($($variant:ident),* $(,)?) {$($field:ident : $type:ty),* $(,)?}) => {
        paste! {
            #[derive(Clone, Debug)]
            pub struct [< $name Base >] {
                pub [< $parent:lower >]: [< $parent Base >],
                $(
                    pub $field: $type,
                 )*
            }
            impl_derefs!([< $name Base >], [< $parent:lower >]);

            #[derive(Clone, Copy, Debug)]
            pub enum $name<'a> {
                $($variant(&'a $variant)),*
            }
            impl<'a> Deref for $name<'a> {
                type Target = [< $base_type Base >];
                fn deref(&self) -> &Self::Target {
                    match self {
                        $(
                            Self::$variant(v) => &v.[< $base_type:lower >],
                        )*
                    }
                }
            }

            #[derive(Debug)]
            pub enum [< $name Mut >]<'a> {
                $($variant(&'a mut $variant)),*
            }
            impl<'a> Deref for [< $name Mut >]<'a> {
                type Target = [< $base_type Base >];
                fn deref(&self) -> &Self::Target {
                    match self {
                        $(
                            Self::$variant(v) => &v.[< $base_type:lower >],
                        )*
                    }
                }
            }
            impl<'a> DerefMut for [< $name Mut >]<'a> {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    match self {
                        $(
                            Self::$variant(v) => &mut v.[< $base_type:lower >],
                        )*
                    }
                }
            }
        }
    };
}

define_category! {
    Unit : Entity [Unit] (
        BuilderBot,
        Core,
        Gunner,
        Sentinel,
        Breach,
        Launcher,
    ) {
        action_cooldown: i32,
        move_cooldown: i32,
    }
}

impl UnitBase {
    pub fn can_act(&self) -> bool {
        self.action_cooldown <= 0
    }

    pub fn can_move(&self) -> bool {
        self.move_cooldown <= 0
    }

    pub fn end_turn(&mut self) {
        if self.action_cooldown > 0 {
            self.action_cooldown -= 1;
        }
        if self.move_cooldown > 0 {
            self.move_cooldown -= 1;
        }
    }
}

impl Unit<'_> {
    pub fn vision_radius_sq(&self) -> i32 {
        match self {
            Unit::BuilderBot(_) => BUILDER_BOT_VISION_RADIUS_SQ,
            Unit::Core(_) => CORE_VISION_RADIUS_SQ,
            Unit::Gunner(_) => GUNNER_VISION_RADIUS_SQ,
            Unit::Sentinel(_) => SENTINEL_VISION_RADIUS_SQ,
            Unit::Breach(_) => BREACH_VISION_RADIUS_SQ,
            Unit::Launcher(_) => LAUNCHER_VISION_RADIUS_SQ,
        }
    }

    pub fn action_radius_sq(&self) -> i32 {
        match self {
            Unit::Core(_) => CORE_ACTION_RADIUS_SQ,
            Unit::Launcher(_) => LAUNCHER_VISION_RADIUS_SQ,
            _ => ACTION_RADIUS_SQ,
        }
    }
}

define_category! {
    Turret : Unit [Turret] (
        Gunner,
        Sentinel,
        Breach,
        Launcher,
    ) {
        ammo_type: Option<ResourceType>,
        ammo_amount: i32,
    }
}

impl TurretBase {
    pub fn receive_ammo(&mut self, resource: ResourceType) {
        assert!(self.ammo_amount == 0);
        self.ammo_type = match resource {
            ResourceType::Titanium => Some(ResourceType::Titanium),
            ResourceType::RawAxionite => Some(ResourceType::Titanium),
            ResourceType::RefinedAxionite => Some(ResourceType::RefinedAxionite),
        };
        self.ammo_amount = STACK_SIZE;
    }
}

impl Turret<'_> {
    pub fn vision_radius_sq(&self) -> i32 {
        match self {
            Turret::Gunner(_) => GUNNER_VISION_RADIUS_SQ,
            Turret::Sentinel(_) => SENTINEL_VISION_RADIUS_SQ,
            Turret::Breach(_) => BREACH_VISION_RADIUS_SQ,
            Turret::Launcher(_) => LAUNCHER_VISION_RADIUS_SQ,
        }
    }
}

define_category! {
    Building : Entity [Entity] (
        Conveyor,
        Splitter,
        ArmouredConveyor,
        Bridge,
        Harvester,
        Foundry,
        Road,
        Barrier,
        Marker,
        Core,
        Gunner,
        Sentinel,
        Breach,
        Launcher,
    ) {}
}

#[derive(Clone, Debug)]
pub struct Conveyor {
    pub building: BuildingBase,
    pub direction: Direction,
    pub stored: Option<ResourceType>,
}
impl_derefs!(Conveyor, building);

#[derive(Clone, Debug)]
pub struct Splitter {
    pub building: BuildingBase,
    pub direction: Direction,
    pub stored: Option<ResourceType>,
}
impl_derefs!(Splitter, building);

#[derive(Clone, Debug)]
pub struct Bridge {
    pub building: BuildingBase,
    pub target: Pos,
    pub stored: Option<ResourceType>,
}
impl_derefs!(Bridge, building);

#[derive(Clone, Debug)]
pub struct ArmouredConveyor {
    pub building: BuildingBase,
    pub direction: Direction,
    pub stored: Option<ResourceType>,
}
impl_derefs!(ArmouredConveyor, building);

#[derive(Clone, Debug)]
pub struct Harvester {
    pub building: BuildingBase,
    pub resource_type: ResourceType,
    pub cooldown: i32,
}
impl_derefs!(Harvester, building);

#[derive(Clone, Debug)]
pub struct Foundry {
    pub building: BuildingBase,
    pub stored: Option<ResourceType>,
}
impl_derefs!(Foundry, building);

#[derive(Clone, Debug)]
pub struct Road {
    pub building: BuildingBase,
}
impl_derefs!(Road, building);

#[derive(Clone, Debug)]
pub struct Barrier {
    pub building: BuildingBase,
}
impl_derefs!(Barrier, building);

#[derive(Clone, Debug)]
pub struct Marker {
    pub building: BuildingBase,
    pub value: u32,
}
impl_derefs!(Marker, building);

#[derive(Clone, Debug)]
pub struct Core {
    pub unit: UnitBase,
    pub received: Vec<ResourceType>,
}
impl_derefs!(Core, unit);

#[derive(Clone, Debug)]
pub struct Gunner {
    pub turret: TurretBase,
    pub direction: Direction,
}
impl_derefs!(Gunner, turret);

#[derive(Clone, Debug)]
pub struct Sentinel {
    pub turret: TurretBase,
    pub direction: Direction,
}
impl_derefs!(Sentinel, turret);

#[derive(Clone, Debug)]
pub struct Breach {
    pub turret: TurretBase,
    pub direction: Direction,
}
impl_derefs!(Breach, turret);

#[derive(Clone, Debug)]
pub struct Launcher {
    pub turret: TurretBase,
}
impl_derefs!(Launcher, turret);

#[derive(Clone, Debug)]
pub struct BuilderBot {
    pub unit: UnitBase,
}
impl_derefs!(BuilderBot, unit);

#[derive(Clone, Debug)]
pub struct Tile {
    pub position: Pos,
    pub building: Option<i32>,
    pub builder_bot: Option<i32>,
    pub environment: Environment,
}

impl Tile {
    pub fn is_empty(&self) -> bool {
        self.building.is_none() && self.environment != Environment::Wall
    }

    pub fn is_bot_passable(&self, entities: &HashMap<i32, Entity>, team: Team) -> bool {
        if self.builder_bot.is_some() {
            return false;
        }
        if let Some(id) = self.building {
            let entity = entities
                .get(&id)
                .unwrap_or_else(|| panic!("tile building id missing entity {}", id));
            matches!(
                entity,
                Entity::Conveyor(_)
                    | Entity::Splitter(_)
                    | Entity::ArmouredConveyor(_)
                    | Entity::Bridge(_)
                    | Entity::Road(_)
            ) || matches!(entity, Entity::Core(_) if entity.team == team)
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameMap {
    pub width: i32,
    pub height: i32,
    pub tiles: Vec<Vec<Tile>>,
}

impl GameMap {
    pub fn in_bounds(&self, pos: Pos) -> bool {
        pos.x >= 0 && pos.x < self.width && pos.y >= 0 && pos.y < self.height
    }

    pub fn tile(&self, pos: Pos) -> &Tile {
        assert!(self.in_bounds(pos), "position out of bounds: {:?}", pos);
        &self.tiles[pos.y as usize][pos.x as usize]
    }

    pub fn tile_mut(&mut self, pos: Pos) -> &mut Tile {
        assert!(self.in_bounds(pos), "position out of bounds: {:?}", pos);
        &mut self.tiles[pos.y as usize][pos.x as usize]
    }

    pub fn place_building_tile(&mut self, id: i32, pos: Pos) {
        self.tile_mut(pos).building = Some(id);
    }

    pub fn build_conveyor(
        &mut self,
        id: i32,
        team: Team,
        position: Pos,
        direction: Direction,
    ) -> Conveyor {
        self.place_building_tile(id, position);
        Conveyor {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: CONVEYOR_MAX_HP,
                    max_hp: CONVEYOR_MAX_HP,
                },
            },
            direction,
            stored: None,
        }
    }

    pub fn build_splitter(
        &mut self,
        id: i32,
        team: Team,
        position: Pos,
        direction: Direction,
    ) -> Splitter {
        self.place_building_tile(id, position);
        Splitter {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: SPLITTER_MAX_HP,
                    max_hp: SPLITTER_MAX_HP,
                },
            },
            direction,
            stored: None,
        }
    }

    pub fn build_bridge(&mut self, id: i32, team: Team, position: Pos, target: Pos) -> Bridge {
        self.place_building_tile(id, position);
        Bridge {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: BRIDGE_MAX_HP,
                    max_hp: BRIDGE_MAX_HP,
                },
            },
            target,
            stored: None,
        }
    }

    pub fn build_armoured_conveyor(
        &mut self,
        id: i32,
        team: Team,
        position: Pos,
        direction: Direction,
    ) -> ArmouredConveyor {
        self.place_building_tile(id, position);
        ArmouredConveyor {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: ARMOURED_CONVEYOR_MAX_HP,
                    max_hp: ARMOURED_CONVEYOR_MAX_HP,
                },
            },
            direction,
            stored: None,
        }
    }

    pub fn build_harvester(&mut self, id: i32, team: Team, position: Pos) -> Harvester {
        self.place_building_tile(id, position);
        let resource_type = match self.tile(position).environment {
            Environment::OreTitanium => ResourceType::Titanium,
            Environment::OreAxionite => ResourceType::RawAxionite,
            env => panic!(
                "build_harvester called on non-ore tile {:?}: {:?}",
                position, env
            ),
        };
        Harvester {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: HARVESTER_MAX_HP,
                    max_hp: HARVESTER_MAX_HP,
                },
            },
            resource_type,
            cooldown: 0,
        }
    }

    pub fn build_road(&mut self, id: i32, team: Team, position: Pos) -> Road {
        self.place_building_tile(id, position);
        Road {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: ROAD_MAX_HP,
                    max_hp: ROAD_MAX_HP,
                },
            },
        }
    }

    pub fn build_barrier(&mut self, id: i32, team: Team, position: Pos) -> Barrier {
        self.place_building_tile(id, position);
        Barrier {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: BARRIER_MAX_HP,
                    max_hp: BARRIER_MAX_HP,
                },
            },
        }
    }

    pub fn build_gunner(
        &mut self,
        id: i32,
        team: Team,
        position: Pos,
        direction: Direction,
    ) -> Gunner {
        self.place_building_tile(id, position);
        Gunner {
            turret: TurretBase {
                unit: UnitBase {
                    entity: EntityBase {
                        id,
                        team,
                        position,
                        hp: GUNNER_MAX_HP,
                        max_hp: GUNNER_MAX_HP,
                    },
                    action_cooldown: 0,
                    move_cooldown: 0,
                },
                ammo_type: None,
                ammo_amount: 0,
            },
            direction,
        }
    }

    pub fn build_sentinel(
        &mut self,
        id: i32,
        team: Team,
        position: Pos,
        direction: Direction,
    ) -> Sentinel {
        self.place_building_tile(id, position);
        Sentinel {
            turret: TurretBase {
                unit: UnitBase {
                    entity: EntityBase {
                        id,
                        team,
                        position,
                        hp: SENTINEL_MAX_HP,
                        max_hp: SENTINEL_MAX_HP,
                    },
                    action_cooldown: 0,
                    move_cooldown: 0,
                },
                ammo_type: None,
                ammo_amount: 0,
            },
            direction,
        }
    }

    pub fn build_breach(
        &mut self,
        id: i32,
        team: Team,
        position: Pos,
        direction: Direction,
    ) -> Breach {
        self.place_building_tile(id, position);
        Breach {
            turret: TurretBase {
                unit: UnitBase {
                    entity: EntityBase {
                        id,
                        team,
                        position,
                        hp: BREACH_MAX_HP,
                        max_hp: BREACH_MAX_HP,
                    },
                    action_cooldown: 0,
                    move_cooldown: 0,
                },
                ammo_type: None,
                ammo_amount: 0,
            },
            direction,
        }
    }

    pub fn build_launcher(&mut self, id: i32, team: Team, position: Pos) -> Launcher {
        self.place_building_tile(id, position);
        Launcher {
            turret: TurretBase {
                unit: UnitBase {
                    entity: EntityBase {
                        id,
                        team,
                        position,
                        hp: LAUNCHER_MAX_HP,
                        max_hp: LAUNCHER_MAX_HP,
                    },
                    action_cooldown: 0,
                    move_cooldown: 0,
                },
                ammo_type: None,
                ammo_amount: 0,
            },
        }
    }

    pub fn build_foundry(&mut self, id: i32, team: Team, position: Pos) -> Foundry {
        self.place_building_tile(id, position);
        Foundry {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: FOUNDRY_MAX_HP,
                    max_hp: FOUNDRY_MAX_HP,
                },
            },
            stored: None,
        }
    }

    pub fn build_marker(&mut self, id: i32, team: Team, position: Pos, value: u32) -> Marker {
        self.place_building_tile(id, position);
        Marker {
            building: BuildingBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: MARKER_MAX_HP,
                    max_hp: MARKER_MAX_HP,
                },
            },
            value,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlayerState {
    pub titanium: i32,
    pub axionite: i32,
    pub titanium_collected: i32,
    pub axionite_collected: i32,
    pub scale_milli: i32,
}

impl PlayerState {
    pub fn can_afford(&self, cost: (i32, i32)) -> bool {
        self.titanium >= cost.0 && self.axionite >= cost.1
    }

    pub fn spend(&mut self, cost: (i32, i32)) {
        self.titanium -= cost.0;
        self.axionite -= cost.1;
    }

    pub fn add_resource(&mut self, resource: ResourceType) {
        match resource {
            ResourceType::Titanium | ResourceType::RawAxionite => {
                self.titanium += STACK_SIZE;
                self.titanium_collected += STACK_SIZE;
            }
            ResourceType::RefinedAxionite => {
                self.axionite += STACK_SIZE;
                self.axionite_collected += STACK_SIZE;
            }
        }
    }
}
