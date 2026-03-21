use super::*;

macro_rules! build_methods {
    ($($name:ident ($pos:ident : Pos $(, $arg:ident : $ty:ty)* $(,)?));* $(;)?) => {
        paste! {
            $(
                pub fn [< build_ $name:snake:lower >](&mut self, bot_id: i32, $pos: Pos $(, $arg: $ty)*) {
                    let team = self.entity(bot_id).expect("unknown builder bot").team;
                    let cost = self.scaled_cost(team, [< $name:snake:upper _BASE_COST >]);
                    self.spend(team, cost);
                    self.destroy_marker_if_present($pos);
                    let id = self.new_id();
                    let building = self.game_map.[< build_ $name:snake:lower >](id, team, $pos $(, $arg)*);
                    self.finish_building(bot_id, Entity::$name(building));
                }
            )*
        }
    };
}

impl Game {
    fn destroy_marker_if_present(&mut self, pos: Pos) {
        let Some(existing_id) = self.game_map.tile(pos).building else {
            return;
        };
        if matches!(self.entity(existing_id), Some(Entity::Marker(_))) {
            self.destroy_entity(existing_id);
        }
    }

    fn finish_building(&mut self, bot_id: i32, building: Entity) {
        let id = building.id;
        let team = building.team;
        assert!(
            !self.entities.contains_key(&id),
            "entity id already exists {}",
            id
        );
        self.players[team.index()].scale_milli += building.scale_contribution();
        self.entities.insert(id, building.clone());
        if matches!(
            building,
            Entity::Core(_)
                | Entity::Gunner(_)
                | Entity::Sentinel(_)
                | Entity::Breach(_)
                | Entity::Launcher(_)
        ) {
            self.unit_order.push(id);
        }
        if matches!(building, Entity::Harvester(_)) {
            self.harvesters.push(id);
        }
        if let Some(Entity::BuilderBot(bot)) = self.entity_mut(bot_id) {
            bot.action_cooldown += 1;
            let diff = GameDiff::SetActionCooldown {
                id: bot_id,
                value: bot.action_cooldown,
            };
            self.replay_recorder.append(diff);
        } else {
            panic!("builder bot id is not a builder bot");
        }
        self.replay_recorder.append(GameDiff::PlaceEntity {
            id,
            entity: building,
        });
    }

    build_methods! {
        Conveyor(position: Pos, direction: Direction);
        Splitter(position: Pos, direction: Direction);
        Bridge(position: Pos, target: Pos);
        ArmouredConveyor(position: Pos, direction: Direction);
        Harvester(position: Pos);
        Foundry(position: Pos);
        Road(position: Pos);
        Barrier(position: Pos);
        Gunner(position: Pos, direction: Direction);
        Sentinel(position: Pos, direction: Direction);
        Breach(position: Pos, direction: Direction);
        Launcher(position: Pos);
    }

    pub fn place_marker(&mut self, team: Team, position: Pos, value: u32) {
        assert!(
            self.game_map.in_bounds(position),
            "marker position out of bounds: {:?}",
            position
        );
        let tile = self.game_map.tile(position);
        if let Some(id) = tile.building {
            match self.entities.get_mut(&id) {
                Some(Entity::Marker(marker)) if marker.team == team => {
                    marker.value = value;
                    self.replay_recorder.append(GameDiff::PlaceEntity {
                        id,
                        entity: Entity::Marker(marker.clone()),
                    });
                }
                Some(_) => panic!(
                    "marker placed on enemy marker or non-marker building id {}",
                    id
                ),
                None => panic!("tile building id missing entity {}", id),
            }
        } else {
            let id = self.new_id();
            let building = self.game_map.build_marker(id, team, position, value);
            self.entities.insert(id, Entity::Marker(building.clone()));
            self.replay_recorder.append(GameDiff::PlaceEntity {
                id,
                entity: Entity::Marker(building),
            });
        }
    }

    pub fn spawn_builder(&mut self, core_id: i32, position: Pos) {
        let team = self.entity(core_id).expect("unknown core").team;
        assert!(
            self.game_map.in_bounds(position),
            "builder spawn position out of bounds: {:?}",
            position
        );
        let cost = self.scaled_cost(team, BUILDER_BOT_BASE_COST);
        self.spend(team, cost);
        let id = self.new_id();
        let bot = BuilderBot {
            unit: UnitBase {
                entity: EntityBase {
                    id,
                    team,
                    position,
                    hp: BUILDER_BOT_MAX_HP,
                    max_hp: BUILDER_BOT_MAX_HP,
                },
                action_cooldown: 0,
                move_cooldown: 0,
            },
        };
        self.entities.insert(id, Entity::BuilderBot(bot.clone()));
        self.players[team.index()].scale_milli += 100;
        let tile = self.game_map.tile_mut(position);
        assert!(tile.builder_bot.is_none(), "tile already has builder bot");
        tile.builder_bot = Some(id);
        self.unit_order.push(id);
        self.replay_recorder.append(GameDiff::PlaceEntity {
            id,
            entity: Entity::BuilderBot(bot),
        });
        match self.entity_mut(core_id) {
            Some(Entity::Core(core)) => {
                core.action_cooldown = 1;
            }
            _ => panic!("core id is not a core"),
        }
        self.replay_recorder.append(GameDiff::SetActionCooldown {
            id: core_id,
            value: 1,
        });
    }
}
