use super::*;

impl Game {
    /// Returns the position the gunner would hit (closest non-empty tile in its
    /// facing direction within vision range), or `None` if nothing is in range.
    pub fn gunner_target(&self, turret_id: i32) -> Option<Pos> {
        let entity = self.entity(turret_id).expect("unknown turret");
        let Turret::Gunner(turret) = entity.as_turret().expect("not a turret") else {
            panic!("gunner_target called on non-gunner");
        };
        let origin = turret.position;
        let dir = turret.direction;
        let vision_sq = GUNNER_VISION_RADIUS_SQ;
        let mut pos = origin + dir;
        loop {
            if !self.game_map.in_bounds(pos) {
                return None;
            }
            if origin.distance_squared(pos) > vision_sq {
                return None;
            }
            let tile = self.game_map.tile(pos);
            let occupied = tile.building.is_some()
                || tile.builder_bot.is_some()
                || tile.environment == Environment::Wall;
            if occupied {
                return Some(pos);
            }
            pos = pos + dir;
        }
    }

    /// Gunner: hits the closest non-empty tile in its facing direction.
    pub fn fire_gunner(&mut self, turret_id: i32, axionite: bool) {
        let damage = if axionite {
            GUNNER_DAMAGE * 2
        } else {
            GUNNER_DAMAGE
        };
        let from = self.entity(turret_id).expect("unknown turret").position;
        if let Some(target) = self.gunner_target(turret_id) {
            self.damage_tile(target, damage);
            self.replay_recorder.append(GameDiff::FireTurret { from, to: target });
        }
        self.finish_firing_turret(turret_id, GUNNER_AMMO_COST, GUNNER_FIRE_COOLDOWN);
    }

    /// Check whether `target` is in the Sentinel's attack range: the intersection
    /// of the Chebyshev-1 line shape and the vision radius.
    pub fn sentinel_target_valid(&self, turret_id: i32, target: Pos) -> bool {
        let entity = self.entity(turret_id).expect("unknown turret");
        let Turret::Sentinel(turret) = entity.as_turret().expect("not a turret") else {
            panic!("sentinel_target_valid called on non-sentinel");
        };
        let origin = turret.position;
        let dist_sq = origin.distance_squared(target);
        if dist_sq == 0 || dist_sq > SENTINEL_VISION_RADIUS_SQ {
            return false;
        }
        let (dx, dy) = turret.direction.delta();
        let rx = target.x - origin.x;
        let ry = target.y - origin.y;
        let mut k = 1;
        while k * k * (dx * dx + dy * dy) <= SENTINEL_VISION_RADIUS_SQ {
            let lx = rx - k * dx;
            let ly = ry - k * dy;
            if lx.abs() <= 1 && ly.abs() <= 1 {
                return true;
            }
            k += 1;
        }
        false
    }

    /// Sentinel: hits the target tile. With axionite, stuns the direct target 3 turns.
    pub fn fire_sentinel(&mut self, turret_id: i32, target: Pos, axionite: bool) {
        let from = self.entity(turret_id).expect("unknown turret").position;
        self.replay_recorder.append(GameDiff::FireTurret { from, to: target });
        self.damage_tile(target, SENTINEL_DAMAGE);
        if axionite {
            // Stun any unit on the direct target tile for 3 turns.
            if self.game_map.in_bounds(target) {
                let tile = self.game_map.tile(target);
                // Stun building if it's a unit (turret/core).
                let ids: Vec<i32> = [tile.building, tile.builder_bot]
                    .iter()
                    .filter_map(|id| *id)
                    .collect();
                for id in ids {
                    let entity = self
                        .entities
                        .get_mut(&id)
                        .unwrap_or_else(|| panic!("tile entity id missing from entities: {}", id));
                    if let Some(mut unit) = entity.as_unit_mut() {
                        unit.action_cooldown += 3;
                        unit.move_cooldown += 3;
                        self.replay_recorder.append(GameDiff::SetActionCooldown {
                            id,
                            value: unit.action_cooldown,
                        });
                        self.replay_recorder.append(GameDiff::SetMoveCooldown {
                            id,
                            value: unit.move_cooldown,
                        });
                    }
                }
            }
        }
        self.finish_firing_turret(turret_id, SENTINEL_AMMO_COST, SENTINEL_FIRE_COOLDOWN);
    }

    /// Check whether `target` is in the Breach's 180° cone (radius_sq ≤ 5, forward half-plane).
    pub fn breach_target_valid(&self, turret_id: i32, target: Pos) -> bool {
        let entity = self.entity(turret_id).expect("unknown turret");
        let Turret::Breach(turret) = entity.as_turret().expect("not a turret") else {
            panic!("breach_target_valid called on non-breach");
        };
        let origin = turret.position;
        let dist_sq = origin.distance_squared(target);
        if dist_sq == 0 || dist_sq > BREACH_ATTACK_RADIUS_SQ {
            return false;
        }
        // Must be in the forward half-plane (dot product with direction >= 0).
        // dot == 0 means exactly perpendicular (90°), which is within the 180° cone.
        let (dx, dy) = turret.direction.delta();
        let rx = target.x - origin.x;
        let ry = target.y - origin.y;
        let dot = rx * dx + ry * dy;
        dot >= 0
    }

    /// Breach: hits the target tile with high damage and deals splash damage
    /// to the 8 surrounding tiles (friendly fire enabled, except the breach itself).
    pub fn fire_breach(&mut self, turret_id: i32, target: Pos) {
        let origin = self.entity(turret_id).expect("unknown turret").position;
        self.replay_recorder.append(GameDiff::FireTurret { from: origin, to: target });
        self.damage_tile(target, BREACH_DAMAGE);
        for dir in [
            Direction::North,
            Direction::Northeast,
            Direction::East,
            Direction::Southeast,
            Direction::South,
            Direction::Southwest,
            Direction::West,
            Direction::Northwest,
        ] {
            let splash_pos = target + dir;
            if splash_pos == origin {
                continue;
            }
            self.damage_tile(splash_pos, BREACH_SPLASH_DAMAGE);
        }
        self.finish_firing_turret(turret_id, BREACH_AMMO_COST, BREACH_FIRE_COOLDOWN);
    }

    /// Check whether `target` is within the Launcher's action range.
    pub fn launcher_target_valid(&self, turret_id: i32, target: Pos) -> bool {
        let entity = self.entity(turret_id).expect("unknown turret");
        let origin = entity.position;
        let dist_sq = origin.distance_squared(target);
        dist_sq > 0 && dist_sq <= LAUNCHER_VISION_RADIUS_SQ
    }

    /// Launcher: picks up an adjacent builder bot and throws it to the target tile.
    pub fn fire_launcher(&mut self, turret_id: i32, bot_id: i32, target: Pos) {
        // Move the bot from its current position to the target.
        let from_pos = match self.entity(bot_id) {
            Some(Entity::BuilderBot(bot)) => bot.position,
            _ => panic!("fire_launcher: bot_id {} is not a builder bot", bot_id),
        };
        self.game_map.tile_mut(from_pos).builder_bot = None;
        self.game_map.tile_mut(target).builder_bot = Some(bot_id);
        let Some(Entity::BuilderBot(bot)) = self.entity_mut(bot_id) else {
            unreachable!()
        };
        bot.position = target;
        self.replay_recorder.append(GameDiff::MoveBuilderBot {
            id: bot_id,
            to: target,
        });
        self.finish_firing_turret(turret_id, 0, LAUNCHER_FIRE_COOLDOWN);
    }

    fn finish_firing_turret(&mut self, turret_id: i32, ammo_cost: i32, cooldown: i32) {
        let entity = self.entity_mut(turret_id).expect("unknown turret id");
        let mut turret = entity.as_turret_mut().expect("not a turret");
        turret.ammo_amount -= ammo_cost;
        assert!(
            turret.ammo_amount >= 0,
            "turret {} does not have enough ammo",
            turret_id
        );
        if turret.ammo_amount == 0 {
            turret.ammo_type = None;
        }
        turret.action_cooldown += cooldown;
        let new_cd = turret.action_cooldown;
        self.replay_recorder.append(GameDiff::SetActionCooldown {
            id: turret_id,
            value: new_cd,
        });
    }
}
