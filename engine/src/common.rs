use std::fmt;
use std::ops::Add;
use std::str::FromStr;

pub mod game_constants {
    pub const MAX_TURNS: i32 = 2000;
    pub const STACK_SIZE: i32 = 10;
    pub const STARTING_TITANIUM: i32 = 1000;
    pub const STARTING_AXIONITE: i32 = 0;

    pub const ACTION_RADIUS_SQ: i32 = 2;
    pub const CORE_SPAWNING_RADIUS_SQ: i32 = 2;
    pub const CORE_ACTION_RADIUS_SQ: i32 = 8;

    pub const CORE_VISION_RADIUS_SQ: i32 = 36;
    pub const BUILDER_BOT_VISION_RADIUS_SQ: i32 = 20;
    pub const GUNNER_VISION_RADIUS_SQ: i32 = 13;
    pub const SENTINEL_VISION_RADIUS_SQ: i32 = 32;
    pub const BREACH_VISION_RADIUS_SQ: i32 = 10;
    pub const LAUNCHER_VISION_RADIUS_SQ: i32 = 26;

    pub const CONVEYOR_BASE_COST: (i32, i32) = (3, 0);
    pub const SPLITTER_BASE_COST: (i32, i32) = (6, 0);
    pub const BRIDGE_BASE_COST: (i32, i32) = (10, 0);
    pub const ARMOURED_CONVEYOR_BASE_COST: (i32, i32) = (10, 5);
    pub const HARVESTER_BASE_COST: (i32, i32) = (80, 0);
    pub const ROAD_BASE_COST: (i32, i32) = (1, 0);
    pub const BARRIER_BASE_COST: (i32, i32) = (3, 0);
    pub const GUNNER_BASE_COST: (i32, i32) = (10, 0);
    pub const SENTINEL_BASE_COST: (i32, i32) = (15, 0);
    pub const BREACH_BASE_COST: (i32, i32) = (30, 10);
    pub const LAUNCHER_BASE_COST: (i32, i32) = (20, 0);
    pub const FOUNDRY_BASE_COST: (i32, i32) = (120, 0);
    pub const BUILDER_BOT_BASE_COST: (i32, i32) = (10, 0);

    pub const CONVEYOR_MAX_HP: i32 = 20;
    pub const SPLITTER_MAX_HP: i32 = 20;
    pub const BRIDGE_MAX_HP: i32 = 20;
    pub const ARMOURED_CONVEYOR_MAX_HP: i32 = 50;
    pub const HARVESTER_MAX_HP: i32 = 30;
    pub const ROAD_MAX_HP: i32 = 10;
    pub const BARRIER_MAX_HP: i32 = 30;
    pub const FOUNDRY_MAX_HP: i32 = 50;
    pub const MARKER_MAX_HP: i32 = 1;

    pub const BUILDER_BOT_MAX_HP: i32 = 30;
    pub const CORE_MAX_HP: i32 = 500;
    pub const GUNNER_MAX_HP: i32 = 40;
    pub const SENTINEL_MAX_HP: i32 = 30;
    pub const BREACH_MAX_HP: i32 = 60;
    pub const LAUNCHER_MAX_HP: i32 = 30;

    pub const BUILDER_BOT_SELF_DESTRUCT_DAMAGE: i32 = 20;
    pub const HEAL_AMOUNT: i32 = 10;

    pub const BRIDGE_TARGET_RADIUS_SQ: i32 = 9;

    // Turret firing constants
    pub const GUNNER_DAMAGE: i32 = 10;
    pub const GUNNER_FIRE_COOLDOWN: i32 = 1;
    pub const GUNNER_AMMO_COST: i32 = 2;

    pub const SENTINEL_DAMAGE: i32 = 20;
    pub const SENTINEL_FIRE_COOLDOWN: i32 = 4;
    pub const SENTINEL_AMMO_COST: i32 = 10;

    pub const BREACH_DAMAGE: i32 = 40;
    pub const BREACH_SPLASH_DAMAGE: i32 = 20;
    pub const BREACH_FIRE_COOLDOWN: i32 = 1;
    pub const BREACH_AMMO_COST: i32 = 5;
    pub const BREACH_ATTACK_RADIUS_SQ: i32 = 5;

    pub const LAUNCHER_FIRE_COOLDOWN: i32 = 1;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

impl Add<Direction> for Pos {
    type Output = Pos;

    fn add(self, d: Direction) -> Pos {
        let (dx, dy) = d.delta();
        Pos {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

impl Pos {
    pub fn distance_squared(self, other: Pos) -> i32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Team {
    A,
    B,
}

impl Team {
    pub fn index(self) -> usize {
        match self {
            Team::A => 0,
            Team::B => 1,
        }
    }
}

impl fmt::Display for Team {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Team::A => write!(f, "a"),
            Team::B => write!(f, "b"),
        }
    }
}

impl FromStr for Team {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "a" => Ok(Team::A),
            "b" => Ok(Team::B),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Titanium,
    RawAxionite,
    RefinedAxionite,
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceType::Titanium => write!(f, "titanium"),
            ResourceType::RawAxionite => write!(f, "raw_axionite"),
            ResourceType::RefinedAxionite => write!(f, "refined_axionite"),
        }
    }
}

impl FromStr for ResourceType {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "titanium" => Ok(ResourceType::Titanium),
            "raw_axionite" => Ok(ResourceType::RawAxionite),
            "refined_axionite" => Ok(ResourceType::RefinedAxionite),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Environment {
    Empty,
    Wall,
    OreTitanium,
    OreAxionite,
}

impl fmt::Display for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Environment::Empty => write!(f, "empty"),
            Environment::Wall => write!(f, "wall"),
            Environment::OreTitanium => write!(f, "ore_titanium"),
            Environment::OreAxionite => write!(f, "ore_axionite"),
        }
    }
}

impl FromStr for Environment {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "empty" => Ok(Environment::Empty),
            "wall" => Ok(Environment::Wall),
            "ore_titanium" => Ok(Environment::OreTitanium),
            "ore_axionite" => Ok(Environment::OreAxionite),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    North,
    Northeast,
    East,
    Southeast,
    South,
    Southwest,
    West,
    Northwest,
    Centre,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::North => write!(f, "north"),
            Direction::Northeast => write!(f, "northeast"),
            Direction::East => write!(f, "east"),
            Direction::Southeast => write!(f, "southeast"),
            Direction::South => write!(f, "south"),
            Direction::Southwest => write!(f, "southwest"),
            Direction::West => write!(f, "west"),
            Direction::Northwest => write!(f, "northwest"),
            Direction::Centre => write!(f, "centre"),
        }
    }
}

impl FromStr for Direction {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "north" => Ok(Direction::North),
            "northeast" => Ok(Direction::Northeast),
            "east" => Ok(Direction::East),
            "southeast" => Ok(Direction::Southeast),
            "south" => Ok(Direction::South),
            "southwest" => Ok(Direction::Southwest),
            "west" => Ok(Direction::West),
            "northwest" => Ok(Direction::Northwest),
            "centre" => Ok(Direction::Centre),
            _ => Err(()),
        }
    }
}

impl Direction {
    pub fn is_cardinal(self) -> bool {
        matches!(
            self,
            Direction::North | Direction::East | Direction::South | Direction::West
        )
    }

    pub fn is_directional(self) -> bool {
        self != Direction::Centre
    }

    pub fn delta(self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::Northeast => (1, -1),
            Direction::East => (1, 0),
            Direction::Southeast => (1, 1),
            Direction::South => (0, 1),
            Direction::Southwest => (-1, 1),
            Direction::West => (-1, 0),
            Direction::Northwest => (-1, -1),
            Direction::Centre => (0, 0),
        }
    }

    pub fn opposite(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::Northeast => Direction::Southwest,
            Direction::East => Direction::West,
            Direction::Southeast => Direction::Northwest,
            Direction::South => Direction::North,
            Direction::Southwest => Direction::Northeast,
            Direction::West => Direction::East,
            Direction::Northwest => Direction::Southeast,
            Direction::Centre => Direction::Centre,
        }
    }
}
