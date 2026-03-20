use crate::common::{Direction, Environment, Pos, ResourceType, Team};
use crate::game_map::{Entity, PlayerState};
use crate::proto;
use crate::replay::recorder::GameDiff;

pub trait ToProto {
    type Output;
    fn to_proto(&self) -> Self::Output;
}

pub fn build_proto_map(environment: &[Vec<Environment>], cores: &[(Pos, Team)]) -> proto::Map {
    let height = environment.len() as i32;
    let width = environment.first().map(|row| row.len()).unwrap_or(0) as i32;
    let rows = environment
        .iter()
        .map(|row| proto::TileRow {
            tiles: row.iter().map(|env| env.to_proto()).collect(),
        })
        .collect();
    let cores = cores
        .iter()
        .enumerate()
        .map(|(id, (pos, team))| proto::CorePosition {
            id: id as i32 + 1,
            team: team.to_proto(),
            position: Some(pos.to_proto()),
        })
        .collect();
    proto::Map {
        width,
        height,
        rows,
        cores,
    }
}

impl ToProto for [PlayerState; 2] {
    type Output = proto::Players;

    fn to_proto(&self) -> Self::Output {
        proto::Players {
            a: Some(self[0].to_proto()),
            b: Some(self[1].to_proto()),
        }
    }
}

impl ToProto for PlayerState {
    type Output = proto::Player;

    fn to_proto(&self) -> Self::Output {
        proto::Player {
            titanium: self.titanium,
            axionite: self.axionite,
            resources_collected: self.axionite_collected,
            titanium_collected: self.titanium_collected,
            axionite_collected: self.axionite_collected,
        }
    }
}

impl ToProto for [GameDiff] {
    type Output = proto::Turn;

    fn to_proto(&self) -> Self::Output {
        proto::Turn {
            updates: self.iter().map(|diff| diff.to_proto()).collect(),
        }
    }
}

impl ToProto for GameDiff {
    type Output = proto::Update;

    fn to_proto(&self) -> Self::Output {
        match self {
            GameDiff::PlaceEntity { entity, .. } => proto::Update {
                kind: Some(proto::update::Kind::PlaceEntity(proto::PlaceEntity {
                    entity: Some(entity.to_proto()),
                })),
            },
            GameDiff::MoveBuilderBot { id, to } => proto::Update {
                kind: Some(proto::update::Kind::MoveBuilderBot(proto::MoveBuilderBot {
                    id: *id,
                    to: Some(to.to_proto()),
                })),
            },
            GameDiff::RemoveEntity { id } => proto::Update {
                kind: Some(proto::update::Kind::RemoveEntity(proto::RemoveEntity {
                    id: *id,
                })),
            },
            GameDiff::DistributeResources { moves } => proto::Update {
                kind: Some(proto::update::Kind::DistributeResources(
                    proto::DistributeResources {
                        moves: moves
                            .iter()
                            .map(|(from, to)| proto::ResourceMove {
                                from: Some(from.to_proto()),
                                to: Some(to.to_proto()),
                            })
                            .collect(),
                    },
                )),
            },
            GameDiff::UpdateHp { id, delta } => proto::Update {
                kind: Some(proto::update::Kind::UpdateHp(proto::UpdateHp {
                    id: *id,
                    delta: *delta,
                })),
            },
            GameDiff::UpdatePlayers { players } => proto::Update {
                kind: Some(proto::update::Kind::UpdatePlayers(proto::UpdatePlayers {
                    players: Some(players.to_proto()),
                })),
            },
            GameDiff::SetActionCooldown { id, value } => proto::Update {
                kind: Some(proto::update::Kind::SetActionCooldown(
                    proto::SetActionCooldown {
                        id: *id,
                        value: *value,
                    },
                )),
            },
            GameDiff::SetMoveCooldown { id, value } => proto::Update {
                kind: Some(proto::update::Kind::SetMoveCooldown(
                    proto::SetMoveCooldown {
                        id: *id,
                        value: *value,
                    },
                )),
            },
            GameDiff::BotOutput {
                id,
                stdout,
                exec_time_us,
                tled,
            } => proto::Update {
                kind: Some(proto::update::Kind::BotOutput(proto::BotOutput {
                    id: *id,
                    stdout: stdout.clone(),
                    exec_time_us: *exec_time_us,
                    tled: *tled,
                })),
            },
            GameDiff::IndicatorLine {
                id,
                pos_a,
                pos_b,
                r,
                g,
                b,
            } => proto::Update {
                kind: Some(proto::update::Kind::IndicatorLine(proto::IndicatorLine {
                    id: *id,
                    pos_a: Some(pos_a.to_proto()),
                    pos_b: Some(pos_b.to_proto()),
                    r: *r,
                    g: *g,
                    b: *b,
                })),
            },
            GameDiff::IndicatorDot { id, pos, r, g, b } => proto::Update {
                kind: Some(proto::update::Kind::IndicatorDot(proto::IndicatorDot {
                    id: *id,
                    pos: Some(pos.to_proto()),
                    r: *r,
                    g: *g,
                    b: *b,
                })),
            },
            GameDiff::FireTurret { from, to } => proto::Update {
                kind: Some(proto::update::Kind::FireTurret(proto::FireTurret {
                    from: Some(from.to_proto()),
                    to: Some(to.to_proto()),
                })),
            },
        }
    }
}

impl ToProto for Entity {
    type Output = proto::Entity;

    fn to_proto(&self) -> Self::Output {
        let mut proto_entity = proto::Entity {
            id: self.id,
            team: self.team.to_proto(),
            position: Some(self.position.to_proto()),
            hp: self.hp,
            max_hp: self.max_hp,
            kind: None,
        };
        proto_entity.kind = Some(match self {
            Entity::BuilderBot(bot) => proto::entity::Kind::BuilderBot(proto::BuilderBot {
                action_cooldown: bot.action_cooldown,
                move_cooldown: bot.move_cooldown,
            }),
            Entity::Conveyor(conveyor) => proto::entity::Kind::Conveyor(proto::Conveyor {
                direction: conveyor.direction.to_proto(),
                stored: conveyor.stored.to_proto(),
            }),
            Entity::Splitter(splitter) => proto::entity::Kind::Splitter(proto::Splitter {
                direction: splitter.direction.to_proto(),
                stored: splitter.stored.to_proto(),
            }),
            Entity::ArmouredConveyor(conveyor) => {
                proto::entity::Kind::ArmouredConveyor(proto::ArmouredConveyor {
                    direction: conveyor.direction.to_proto(),
                    stored: conveyor.stored.to_proto(),
                })
            }
            Entity::Bridge(bridge) => proto::entity::Kind::Bridge(proto::Bridge {
                target: Some(bridge.target.to_proto()),
                stored: bridge.stored.to_proto(),
            }),
            Entity::Harvester(harvester) => proto::entity::Kind::Harvester(proto::Harvester {
                cooldown: harvester.cooldown,
                resource_type: Some(harvester.resource_type).to_proto(),
            }),
            Entity::Foundry(foundry) => proto::entity::Kind::Foundry(proto::Foundry {
                stored: foundry.stored.to_proto(),
            }),
            Entity::Road(_) => proto::entity::Kind::Road(proto::Road {}),
            Entity::Barrier(_) => proto::entity::Kind::Barrier(proto::Barrier {}),
            Entity::Marker(marker) => proto::entity::Kind::Marker(proto::Marker {
                value: marker.value,
            }),
            Entity::Core(core) => proto::entity::Kind::Core(proto::Core {
                action_cooldown: core.action_cooldown,
            }),
            Entity::Gunner(gunner) => proto::entity::Kind::Gunner(proto::Gunner {
                direction: gunner.direction.to_proto(),
                ammo_type: gunner.ammo_type.to_proto(),
                ammo_amount: gunner.ammo_amount,
            }),
            Entity::Sentinel(sentinel) => proto::entity::Kind::Sentinel(proto::Sentinel {
                direction: sentinel.direction.to_proto(),
                ammo_type: sentinel.ammo_type.to_proto(),
                ammo_amount: sentinel.ammo_amount,
            }),
            Entity::Breach(breach) => proto::entity::Kind::Breach(proto::Breach {
                direction: breach.direction.to_proto(),
                ammo_type: breach.ammo_type.to_proto(),
                ammo_amount: breach.ammo_amount,
            }),
            Entity::Launcher(launcher) => proto::entity::Kind::Launcher(proto::Launcher {
                ammo_type: launcher.ammo_type.to_proto(),
                ammo_amount: launcher.ammo_amount,
            }),
        });
        proto_entity
    }
}

impl ToProto for Pos {
    type Output = proto::Pos;

    fn to_proto(&self) -> Self::Output {
        proto::Pos {
            x: self.x,
            y: self.y,
        }
    }
}

impl ToProto for Team {
    type Output = i32;

    fn to_proto(&self) -> Self::Output {
        match self {
            Team::A => proto::Team::A as i32,
            Team::B => proto::Team::B as i32,
        }
    }
}

impl ToProto for Direction {
    type Output = i32;

    fn to_proto(&self) -> Self::Output {
        match self {
            Direction::Centre => proto::Direction::DirCentre as i32,
            Direction::North => proto::Direction::DirNorth as i32,
            Direction::Northeast => proto::Direction::DirNortheast as i32,
            Direction::East => proto::Direction::DirEast as i32,
            Direction::Southeast => proto::Direction::DirSoutheast as i32,
            Direction::South => proto::Direction::DirSouth as i32,
            Direction::Southwest => proto::Direction::DirSouthwest as i32,
            Direction::West => proto::Direction::DirWest as i32,
            Direction::Northwest => proto::Direction::DirNorthwest as i32,
        }
    }
}

impl ToProto for Option<ResourceType> {
    type Output = i32;

    fn to_proto(&self) -> Self::Output {
        match self {
            None => proto::ResourceType::ResourceNone as i32,
            Some(ResourceType::Titanium) => proto::ResourceType::ResourceTitanium as i32,
            Some(ResourceType::RawAxionite) => proto::ResourceType::ResourceRawAxionite as i32,
            Some(ResourceType::RefinedAxionite) => {
                proto::ResourceType::ResourceRefinedAxionite as i32
            }
        }
    }
}

impl ToProto for Environment {
    type Output = i32;

    fn to_proto(&self) -> Self::Output {
        match self {
            Environment::Empty => proto::Environment::EnvEmpty as i32,
            Environment::Wall => proto::Environment::EnvWall as i32,
            Environment::OreTitanium => proto::Environment::EnvOreTitanium as i32,
            Environment::OreAxionite => proto::Environment::EnvOreAxionite as i32,
        }
    }
}
