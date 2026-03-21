use std::cmp::min;
use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::common::game_constants::{
    ARMOURED_CONVEYOR_BASE_COST, BARRIER_BASE_COST, BREACH_AMMO_COST, BREACH_ATTACK_RADIUS_SQ,
    BREACH_BASE_COST, BREACH_DAMAGE, BREACH_FIRE_COOLDOWN, BREACH_SPLASH_DAMAGE, BRIDGE_BASE_COST,
    BUILDER_BOT_BASE_COST, BUILDER_BOT_MAX_HP, CONVEYOR_BASE_COST, CORE_MAX_HP, FOUNDRY_BASE_COST,
    GUNNER_AMMO_COST, GUNNER_BASE_COST, GUNNER_DAMAGE, GUNNER_FIRE_COOLDOWN,
    GUNNER_VISION_RADIUS_SQ, HARVESTER_BASE_COST,
    HEAL_AMOUNT, LAUNCHER_BASE_COST, LAUNCHER_FIRE_COOLDOWN, LAUNCHER_VISION_RADIUS_SQ, MAX_TURNS,
    ROAD_BASE_COST, SENTINEL_AMMO_COST, SENTINEL_BASE_COST, SENTINEL_DAMAGE,
    SENTINEL_FIRE_COOLDOWN, SENTINEL_VISION_RADIUS_SQ, SPLITTER_BASE_COST, STARTING_AXIONITE,
    STARTING_TITANIUM,
};
use crate::common::{Direction, Environment, Pos, Team};
use crate::game_map::{
    BuilderBot, Core, Entity, EntityBase, GameMap, PlayerState, Tile, Turret, UnitBase,
};
use crate::replay::recorder::{GameDiff, ReplayRecorder};
use paste::paste;

mod build;
mod distribute;
mod turret;

#[derive(Clone, Debug)]
pub struct Game {
    pub game_map: GameMap,
    pub players: [PlayerState; 2],
    pub turn: i32,
    pub next_id: i32,
    pub entities: HashMap<i32, Entity>,
    pub unit_order: Vec<i32>,
    pub harvesters: Vec<i32>,
    pub rng: StdRng,
    pub replay_recorder: ReplayRecorder,
    /// Maps (source_pos, sink_pos) -> last turn the edge was used, for global LRU priority.
    pub edge_last_used: HashMap<(Pos, Pos), i32>,
}

impl Game {
    pub fn new(
        environment: Vec<Vec<Environment>>,
        cores: Vec<(Pos, Team)>,
        seed: u64,
        suppress_indicators: bool,
    ) -> Self {
        let height = environment.len() as i32;
        let width = environment.first().map(|row| row.len()).unwrap_or(0) as i32;
        let mut tiles = Vec::new();
        for y in 0..height {
            let mut row = Vec::new();
            for x in 0..width {
                row.push(Tile {
                    position: Pos { x, y },
                    building: None,
                    builder_bot: None,
                    environment: environment[y as usize][x as usize],
                });
            }
            tiles.push(row);
        }
        let mut game = Self {
            game_map: GameMap {
                width,
                height,
                tiles,
            },
            players: [
                PlayerState {
                    titanium: STARTING_TITANIUM,
                    axionite: STARTING_AXIONITE,
                    titanium_collected: 0,
                    axionite_collected: 0,
                    scale_milli: 1000,
                },
                PlayerState {
                    titanium: STARTING_TITANIUM,
                    axionite: STARTING_AXIONITE,
                    titanium_collected: 0,
                    axionite_collected: 0,
                    scale_milli: 1000,
                },
            ],
            turn: 0,
            next_id: 2, // there are always exactly 2 cores, which use ids 1 and 2
            entities: HashMap::new(),
            unit_order: Vec::new(),
            harvesters: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            edge_last_used: HashMap::new(),
            replay_recorder: ReplayRecorder::new(
                environment.clone(),
                cores.clone(),
                suppress_indicators,
            ),
        };

        for (idx, (pos, team)) in cores.into_iter().enumerate() {
            let id = idx as i32 + 1;
            let core = Entity::Core(Core {
                unit: UnitBase {
                    entity: EntityBase {
                        id,
                        team,
                        position: pos,
                        hp: CORE_MAX_HP,
                        max_hp: CORE_MAX_HP,
                    },
                    action_cooldown: 0,
                    move_cooldown: 0,
                },
                received: Vec::new(),
            });
            game.entities.insert(id, core);
            game.unit_order.push(id);
            for d in [
                Direction::North,
                Direction::Northeast,
                Direction::East,
                Direction::Southeast,
                Direction::South,
                Direction::Southwest,
                Direction::West,
                Direction::Northwest,
                Direction::Centre,
            ] {
                let p = pos + d;
                assert!(game.game_map.in_bounds(p));
                let tile = game.game_map.tile_mut(p);
                tile.building = Some(id);
            }
        }

        game
    }

    pub fn new_id(&mut self) -> i32 {
        self.next_id += 1;
        self.next_id
    }

    pub fn spend(&mut self, team: Team, cost: (i32, i32)) {
        self.players[team.index()].spend(cost);
    }

    pub fn scaled_cost(&self, team: Team, base: (i32, i32)) -> (i32, i32) {
        let scale_milli = self.players[team.index()].scale_milli;
        (base.0 * scale_milli / 1000, base.1 * scale_milli / 1000)
    }

    pub fn is_tile_bot_passable(&self, pos: Pos, team: Team) -> bool {
        if !self.game_map.in_bounds(pos) {
            return false;
        }
        let tile = self.game_map.tile(pos);
        tile.is_bot_passable(&self.entities, team)
    }

    pub fn entity(&self, id: i32) -> Option<&Entity> {
        self.entities.get(&id)
    }

    pub fn entity_mut(&mut self, id: i32) -> Option<&mut Entity> {
        self.entities.get_mut(&id)
    }

    pub fn has_core(&self, team: Team) -> bool {
        self.entities.values().any(|entity| match entity {
            Entity::Core(core) => core.team == team,
            _ => false,
        })
    }

    pub fn winner_team(&mut self) -> Option<Team> {
        let alive = [Team::A, Team::B]
            .into_iter()
            .filter(|t| self.has_core(*t))
            .collect::<Vec<_>>();
        if alive.len() == 1 {
            return Some(alive[0]);
        }
        if self.turn >= MAX_TURNS || alive.is_empty() {
            let a = &self.players[0];
            let b = &self.players[1];
            // Tiebreak 1: most axionite collected
            if a.axionite_collected != b.axionite_collected {
                return Some(if a.axionite_collected > b.axionite_collected {
                    Team::A
                } else {
                    Team::B
                });
            }
            // Tiebreak 2: most titanium collected
            if a.titanium_collected != b.titanium_collected {
                return Some(if a.titanium_collected > b.titanium_collected {
                    Team::A
                } else {
                    Team::B
                });
            }
            // Tiebreak 3: most harvesters owned
            let a_harvesters = self
                .harvesters
                .iter()
                .filter(|id| self.entities.get(id).is_some_and(|e| e.team == Team::A))
                .count();
            let b_harvesters = self
                .harvesters
                .iter()
                .filter(|id| self.entities.get(id).is_some_and(|e| e.team == Team::B))
                .count();
            if a_harvesters != b_harvesters {
                return Some(if a_harvesters > b_harvesters {
                    Team::A
                } else {
                    Team::B
                });
            }
            // Tiebreak 4: most stored axionite
            if a.axionite != b.axionite {
                return Some(if a.axionite > b.axionite {
                    Team::A
                } else {
                    Team::B
                });
            }
            // Tiebreak 5: most stored titanium
            if a.titanium != b.titanium {
                return Some(if a.titanium > b.titanium {
                    Team::A
                } else {
                    Team::B
                });
            }
            // Tiebreak 6: coinflip
            return Some(if self.rng.gen::<bool>() {
                Team::A
            } else {
                Team::B
            });
        }
        None
    }

    pub fn new_turn(&mut self) {
        self.replay_recorder.new_turn();
    }

    pub fn update_cooldowns(&mut self) {
        for unit_id in &self.unit_order {
            match self.entities.get_mut(unit_id) {
                Some(entity) => {
                    let mut unit = entity
                        .as_unit_mut()
                        .unwrap_or_else(|| panic!("unit_order contains non-unit id {}", unit_id));
                    unit.end_turn();
                }
                None => panic!("unit_order contains unknown id {}", unit_id),
            }
        }

        for id in &self.harvesters {
            match self.entities.get_mut(id) {
                Some(Entity::Harvester(h)) => {
                    if h.cooldown > 0 {
                        h.cooldown -= 1;
                    }
                }
                Some(_) => panic!("harvesters contains non-harvester id {}", id),
                None => panic!("harvesters contains unknown id {}", id),
            }
        }
    }

    pub fn move_builder_bot(&mut self, id: i32, to_pos: Pos) {
        let (from_pos, bot_team) = match self.entity(id) {
            Some(Entity::BuilderBot(bot)) => (bot.position, bot.team),
            Some(_) => panic!("id {} is not a builder bot", id),
            None => panic!("unknown builder bot id {}", id),
        };
        assert_ne!(
            from_pos, to_pos,
            "builder bot {} moved to its current position",
            id
        );
        assert!(
            self.game_map.in_bounds(from_pos),
            "builder bot position out of bounds: {:?}",
            from_pos
        );
        assert!(
            self.game_map.in_bounds(to_pos),
            "target position out of bounds: {:?}",
            to_pos
        );
        let tile = self.game_map.tile_mut(to_pos);
        assert!(
            tile.is_bot_passable(&self.entities, bot_team),
            "target tile is not passable"
        );
        tile.builder_bot = Some(id);
        self.game_map.tile_mut(from_pos).builder_bot = None;
        if let Some(Entity::BuilderBot(bot)) = self.entity_mut(id) {
            bot.position = to_pos;
            assert!(bot.move_cooldown == 0);
            bot.move_cooldown = 1;
        } else {
            panic!("builder bot id is not a builder bot");
        }
        self.replay_recorder
            .append(GameDiff::MoveBuilderBot { id, to: to_pos });
        self.replay_recorder
            .append(GameDiff::SetMoveCooldown { id, value: 1 });
    }

    pub fn apply_damage(&mut self, id: i32, amount: i32) {
        let entity = self
            .entities
            .get_mut(&id)
            .unwrap_or_else(|| panic!("unknown entity id {}", id));
        entity.hp -= amount;
        self.replay_recorder
            .append(GameDiff::UpdateHp { id, delta: -amount });
        if entity.hp <= 0 {
            self.destroy_entity(id);
        }
    }

    /// Apply damage to everything on a tile (building + builder bot).
    pub fn damage_tile(&mut self, pos: Pos, amount: i32) {
        if !self.game_map.in_bounds(pos) {
            return;
        }
        let tile = self.game_map.tile(pos);
        let building_id = tile.building;
        let bot_id = tile.builder_bot;
        if let Some(id) = building_id {
            self.apply_damage(id, amount);
        }
        if let Some(id) = bot_id {
            self.apply_damage(id, amount);
        }
    }

    /// Heal all friendly entities on a tile (building + builder bot) by HEAL_AMOUNT, capped at max HP.
    pub fn heal_tile(&mut self, pos: Pos, team: Team) {
        assert!(self.game_map.in_bounds(pos));
        let tile = self.game_map.tile(pos);
        let ids: Vec<i32> = [tile.building, tile.builder_bot]
            .iter()
            .filter_map(|id| *id)
            .filter(|id| self.entities.get(id).is_some_and(|e| e.team == team))
            .collect();
        for id in ids {
            let entity = self.entities.get_mut(&id).expect("unknown entity id");
            let heal = min(HEAL_AMOUNT, entity.max_hp - entity.hp);
            if heal > 0 {
                entity.hp += heal;
                self.replay_recorder
                    .append(GameDiff::UpdateHp { id, delta: heal });
            }
        }
    }

    pub fn remove_builder_bot(&mut self, id: i32) {
        let bot = match self.entities.remove(&id) {
            Some(Entity::BuilderBot(bot)) => bot,
            Some(_) => panic!("id {} is not a builder bot", id),
            None => panic!("unknown builder bot id {}", id),
        };
        self.players[bot.team.index()].scale_milli -= 100;
        let pos = bot.position;
        assert!(
            self.game_map.in_bounds(pos),
            "builder bot position out of bounds: {:?}",
            pos
        );
        let tile = self.game_map.tile_mut(pos);
        assert_eq!(
            tile.builder_bot,
            Some(id),
            "builder bot tile reference mismatch at {:?}",
            pos
        );
        tile.builder_bot = None;
        self.unit_order.retain(|unit_id| *unit_id != id);
        self.replay_recorder.append(GameDiff::RemoveEntity { id });
    }

    pub fn remove_building(&mut self, id: i32) {
        let building = match self.entities.remove(&id) {
            Some(Entity::BuilderBot(_)) => panic!("id {} is not a building", id),
            Some(entity) => entity,
            None => panic!("unknown building id {}", id),
        };
        self.players[building.team.index()].scale_milli -= building.scale_contribution();
        if matches!(
            building,
            Entity::Core(_)
                | Entity::Gunner(_)
                | Entity::Sentinel(_)
                | Entity::Breach(_)
                | Entity::Launcher(_)
        ) {
            self.unit_order.retain(|unit_id| *unit_id != id);
        }
        if matches!(building, Entity::Harvester(_)) {
            self.harvesters.retain(|hid| *hid != id);
        }
        let pos = building.position;
        match building {
            Entity::Core(_) => {
                for d in [
                    Direction::North,
                    Direction::Northeast,
                    Direction::East,
                    Direction::Southeast,
                    Direction::South,
                    Direction::Southwest,
                    Direction::West,
                    Direction::Northwest,
                    Direction::Centre,
                ] {
                    let p = pos + d;
                    assert!(self.game_map.in_bounds(p));
                    let tile = self.game_map.tile_mut(p);
                    assert!(tile.building == Some(id));
                    tile.building = None;
                }
            }
            _ => {
                assert!(
                    self.game_map.in_bounds(pos),
                    "building position out of bounds: {:?}",
                    pos
                );
                self.game_map.tile_mut(pos).building = None;
            }
        }
        self.replay_recorder.append(GameDiff::RemoveEntity { id });
    }

    pub fn destroy_entity(&mut self, id: i32) {
        match self.entity(id) {
            Some(Entity::BuilderBot(_)) => {
                self.remove_builder_bot(id);
            }
            Some(_) => {
                self.remove_building(id);
            }
            None => panic!("unknown unit id {}", id),
        }
    }
}
