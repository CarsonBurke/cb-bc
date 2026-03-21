use std::fs;
use std::io;
use std::path::Path;

use prost::Message;

use crate::common::{Environment, Pos, Team};
use crate::game_map::{Entity, PlayerState};
use crate::proto;
use crate::replay::proto_conversions::{build_proto_map, ToProto};

#[derive(Clone, Debug)]
pub enum GameDiff {
    PlaceEntity {
        id: i32,
        entity: Entity,
    },
    MoveBuilderBot {
        id: i32,
        to: Pos,
    },
    RemoveEntity {
        id: i32,
    },
    DistributeResources {
        moves: Vec<(Pos, Pos)>,
    },
    UpdateHp {
        id: i32,
        delta: i32,
    },
    UpdatePlayers {
        players: [PlayerState; 2],
    },
    SetActionCooldown {
        id: i32,
        value: i32,
    },
    SetMoveCooldown {
        id: i32,
        value: i32,
    },
    BotOutput {
        id: i32,
        stdout: String,
        exec_time_us: u32,
        tled: bool,
    },
    IndicatorLine {
        id: i32,
        pos_a: Pos,
        pos_b: Pos,
        r: i32,
        g: i32,
        b: i32,
    },
    IndicatorDot {
        id: i32,
        pos: Pos,
        r: i32,
        g: i32,
        b: i32,
    },
    FireTurret {
        from: Pos,
        to: Pos,
    },
}

#[derive(Clone, Debug)]
pub struct ReplayRecorder {
    environment: Vec<Vec<Environment>>,
    cores: Vec<(Pos, Team)>,
    diffs: Vec<Vec<GameDiff>>,
    suppress_indicators: bool,
}

impl ReplayRecorder {
    pub fn new(
        environment: Vec<Vec<Environment>>,
        cores: Vec<(Pos, Team)>,
        suppress_indicators: bool,
    ) -> Self {
        Self {
            environment,
            cores,
            diffs: vec![],
            suppress_indicators,
        }
    }

    pub fn new_turn(&mut self) {
        self.diffs.push(vec![]);
    }

    pub fn append(&mut self, diff: GameDiff) {
        if self.suppress_indicators
            && matches!(
                diff,
                GameDiff::IndicatorLine { .. } | GameDiff::IndicatorDot { .. }
            )
        {
            return;
        }
        self.diffs
            .last_mut()
            .expect("append called before new_turn")
            .push(if self.suppress_indicators {
                match diff {
                    GameDiff::BotOutput {
                        id,
                        exec_time_us,
                        tled,
                        ..
                    } => GameDiff::BotOutput {
                        id,
                        stdout: String::new(),
                        exec_time_us,
                        tled,
                    },
                    other => other,
                }
            } else {
                diff
            });
    }

    pub fn build(&self, winner: Option<Team>) -> proto::Replay {
        let map = build_proto_map(&self.environment, &self.cores);
        let turns = self
            .diffs
            .iter()
            .map(|turn| turn.as_slice().to_proto())
            .collect();
        proto::Replay {
            map: Some(map),
            turns,
            winner: winner.map(|team| team.to_proto()),
        }
    }

    pub fn write_to_path(&self, path: &str, winner: Option<Team>) -> io::Result<()> {
        let replay = self.build(winner);
        let mut buf = Vec::new();
        replay
            .encode(&mut buf)
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, buf)
    }
}
