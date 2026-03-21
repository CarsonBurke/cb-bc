// NOTE: When changing the Controller API, also update the Python stub in
// engine/py/cambc.py (class Controller) so that type-checkers stay in sync.

use pyo3::prelude::*;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::rc::Rc;

use crate::common::game_constants::{
    ACTION_RADIUS_SQ, ARMOURED_CONVEYOR_BASE_COST, BARRIER_BASE_COST, BREACH_BASE_COST,
    BRIDGE_BASE_COST, BRIDGE_TARGET_RADIUS_SQ, BUILDER_BOT_BASE_COST,
    BUILDER_BOT_SELF_DESTRUCT_DAMAGE, CONVEYOR_BASE_COST, CORE_SPAWNING_RADIUS_SQ,
    FOUNDRY_BASE_COST, GUNNER_BASE_COST, HARVESTER_BASE_COST, LAUNCHER_BASE_COST, ROAD_BASE_COST,
    SENTINEL_BASE_COST, SPLITTER_BASE_COST,
};
use crate::common::{Environment, Pos};
use crate::game::Game;
use crate::game_map::{BuilderBot, Entity, Tile, Turret};
use crate::replay::recorder::GameDiff;

use super::py_convert::game_error;

#[pyclass(unsendable)]
pub struct Controller {
    game: Rc<RefCell<Game>>,
    unit: i32,
    cpu_deadline_ns: u64,
    cpu_start_ns: u64,
    has_placed_marker: Cell<bool>,
}

impl Controller {
    pub fn new(game: Rc<RefCell<Game>>, unit: i32) -> Self {
        Self {
            game,
            unit,
            cpu_deadline_ns: 0,
            cpu_start_ns: 0,
            has_placed_marker: Cell::new(false),
        }
    }

    pub fn set_cpu_deadline(&mut self, cpu_start_ns: u64, cpu_deadline_ns: u64) {
        self.cpu_start_ns = cpu_start_ns;
        self.cpu_deadline_ns = cpu_deadline_ns;
    }
}

fn game_assert(cond: bool, message: &str) -> PyResult<()> {
    if !cond {
        Err(game_error(message))
    } else {
        Ok(())
    }
}

impl Controller {
    fn can_build_checks(&self, position: Pos, base_cost: (i32, i32)) -> bool {
        let game = self.game();
        let bot = match game.entity(self.unit) {
            Some(Entity::BuilderBot(bot)) => bot,
            _ => return false,
        };
        if !bot.can_act() {
            return false;
        }
        if bot.position.distance_squared(position) > ACTION_RADIUS_SQ {
            return false;
        }
        if !game.game_map.in_bounds(position) {
            return false;
        }
        let tile = game.game_map.tile(position);
        if tile.environment == Environment::Wall {
            return false;
        }
        if let Some(existing_id) = tile.building {
            if !matches!(game.entity(existing_id), Some(Entity::Marker(_))) {
                return false;
            }
        }
        let cost = game.scaled_cost(bot.team, base_cost);
        game.players[bot.team.index()].can_afford(cost)
    }
}

#[pymethods]
impl Controller {
    #[pyo3(signature = (id=None))]
    fn get_team(&self, py: Python, id: Option<i32>) -> PyResult<PyObject> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        let game = self.game();
        let team = game
            .entity(id)
            .ok_or_else(|| game_error("Unknown id"))?
            .team;
        Ok(team.into_pyobject(py)?.unbind())
    }

    #[pyo3(signature = (id=None))]
    fn get_position(&self, py: Python, id: Option<i32>) -> PyResult<PyObject> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        Ok(self.entity(id)?.position.into_pyobject(py)?.unbind())
    }

    fn get_id(&self) -> PyResult<i32> {
        self.check_deadline()?;
        Ok(self.unit)
    }

    fn get_action_cooldown(&self) -> PyResult<i32> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entities
            .get(&self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let unit = entity
            .as_unit()
            .ok_or_else(|| game_error("Unit is not a unit"))?;
        Ok(unit.action_cooldown)
    }

    fn get_move_cooldown(&self) -> PyResult<i32> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entities
            .get(&self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let unit = entity
            .as_unit()
            .ok_or_else(|| game_error("Unit is not a unit"))?;
        Ok(unit.move_cooldown)
    }

    fn get_ammo_amount(&self) -> PyResult<i32> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entities
            .get(&self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let turret = entity
            .as_turret()
            .ok_or_else(|| game_error("Unit is not a turret"))?;
        Ok(turret.ammo_amount)
    }

    fn get_ammo_type(&self, py: Python) -> PyResult<PyObject> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entities
            .get(&self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let turret = entity
            .as_turret()
            .ok_or_else(|| game_error("Unit is not a turret"))?;
        match turret.ammo_type {
            Some(r) => Ok(r.into_pyobject(py)?.unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (id=None))]
    fn get_vision_radius_sq(&self, id: Option<i32>) -> PyResult<i32> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        let game = self.game();
        let entity = game.entity(id).ok_or_else(|| game_error("Unknown id"))?;
        let unit = entity
            .as_unit()
            .ok_or_else(|| game_error("Unit is not a unit"))?;
        Ok(unit.vision_radius_sq())
    }

    #[pyo3(signature = (id=None))]
    fn get_hp(&self, id: Option<i32>) -> PyResult<i32> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        Ok(self.entity(id)?.hp)
    }

    #[pyo3(signature = (id=None))]
    fn get_max_hp(&self, id: Option<i32>) -> PyResult<i32> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        Ok(self.entity(id)?.max_hp)
    }

    #[pyo3(signature = (id=None))]
    fn get_entity_type(&self, py: Python, id: Option<i32>) -> PyResult<PyObject> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        let entity = self.entity(id)?;
        let entity_py = entity.clone().into_pyobject(py)?.unbind();
        Ok(entity_py)
    }

    #[pyo3(signature = (id=None))]
    fn get_direction(&self, py: Python, id: Option<i32>) -> PyResult<PyObject> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        let game = self.game();
        let entity = game
            .entities
            .get(&id)
            .ok_or_else(|| game_error("Unknown id"))?;
        let dir = match entity {
            Entity::Conveyor(c) => c.direction,
            Entity::Splitter(s) => s.direction,
            Entity::ArmouredConveyor(c) => c.direction,
            Entity::Foundry(_) => return Err(game_error("Entity has no direction")),
            Entity::Gunner(t) => t.direction,
            Entity::Sentinel(t) => t.direction,
            Entity::Breach(t) => t.direction,
            _ => return Err(game_error("Entity has no direction")),
        };
        Ok(dir.into_pyobject(py)?.unbind())
    }

    fn get_bridge_target(&self, py: Python, id: i32) -> PyResult<PyObject> {
        self.check_deadline()?;
        self.assert_entity_in_vision(id)?;
        let game = self.game();
        let entity = game
            .entities
            .get(&id)
            .ok_or_else(|| game_error("Unknown id"))?;
        match entity {
            Entity::Bridge(b) => Ok(b.target.into_pyobject(py)?.unbind()),
            _ => Err(game_error("Entity is not a bridge")),
        }
    }

    #[pyo3(signature = (id=None))]
    fn get_stored_resource(&self, py: Python, id: Option<i32>) -> PyResult<PyObject> {
        self.check_deadline()?;
        let id = id.unwrap_or(self.unit);
        self.assert_entity_in_vision(id)?;
        let game = self.game();
        let entity = game
            .entities
            .get(&id)
            .ok_or_else(|| game_error("Unknown id"))?;
        let stored = match entity {
            Entity::Conveyor(c) => c.stored,
            Entity::Splitter(s) => s.stored,
            Entity::ArmouredConveyor(c) => c.stored,
            Entity::Bridge(b) => b.stored,
            Entity::Foundry(f) => f.stored,
            _ => return Err(game_error("Entity has no stored resource")),
        };
        match stored {
            Some(r) => Ok(r.into_pyobject(py)?.unbind()),
            None => Ok(py.None()),
        }
    }

    fn get_tile_env(&self, py: Python, pos: Pos) -> PyResult<PyObject> {
        self.check_deadline()?;
        self.assert_in_vision(pos)?;
        let game = self.game();
        let tile = game.tile_at(pos)?;
        Ok(tile.environment.into_pyobject(py)?.unbind())
    }

    fn get_tile_building_id(&self, pos: Pos) -> PyResult<Option<i32>> {
        self.check_deadline()?;
        self.assert_in_vision(pos)?;
        let game = self.game();
        let tile = game.tile_at(pos)?;
        Ok(tile.building)
    }

    fn get_tile_builder_bot_id(&self, pos: Pos) -> PyResult<Option<i32>> {
        self.check_deadline()?;
        self.assert_in_vision(pos)?;
        let game = self.game();
        let tile = game.tile_at(pos)?;
        Ok(tile.builder_bot)
    }

    fn is_tile_empty(&self, pos: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        self.assert_in_vision(pos)?;
        let game = self.game();
        let tile = game.tile_at(pos)?;
        Ok(tile.is_empty())
    }

    fn is_tile_passable(&self, pos: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        self.assert_in_vision(pos)?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.is_tile_bot_passable(pos, team))
    }

    fn is_in_vision(&self, pos: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entities
            .get(&self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let unit = entity
            .as_unit()
            .ok_or_else(|| game_error("Unit is not a unit"))?;
        Ok(unit.position.distance_squared(pos) <= unit.vision_radius_sq())
    }

    fn get_map_width(&self) -> PyResult<i32> {
        self.check_deadline()?;
        Ok(self.game().game_map.width)
    }

    fn get_map_height(&self) -> PyResult<i32> {
        self.check_deadline()?;
        Ok(self.game().game_map.height)
    }

    fn get_current_round(&self) -> PyResult<i32> {
        self.check_deadline()?;
        Ok(self.game().turn)
    }

    fn get_cpu_time_elapsed(&self) -> PyResult<u64> {
        self.check_deadline()?;
        let elapsed_ns = crate::runner::thread_cpu_time_ns().saturating_sub(self.cpu_start_ns);
        Ok(elapsed_ns / 1_000)
    }

    fn get_global_resources(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        let idx = team.index();
        let player = &game.players[idx];
        Ok((player.titanium, player.axionite))
    }

    fn get_scale_percent(&self) -> PyResult<f64> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.players[team.index()].scale_milli as f64 / 10.0)
    }

    fn get_conveyor_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, CONVEYOR_BASE_COST))
    }

    fn get_splitter_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, SPLITTER_BASE_COST))
    }

    fn get_bridge_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, BRIDGE_BASE_COST))
    }

    fn get_armoured_conveyor_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, ARMOURED_CONVEYOR_BASE_COST))
    }

    fn get_harvester_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, HARVESTER_BASE_COST))
    }

    fn get_road_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, ROAD_BASE_COST))
    }

    fn get_barrier_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, BARRIER_BASE_COST))
    }

    fn get_gunner_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, GUNNER_BASE_COST))
    }

    fn get_sentinel_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, SENTINEL_BASE_COST))
    }

    fn get_breach_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, BREACH_BASE_COST))
    }

    fn get_launcher_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, LAUNCHER_BASE_COST))
    }

    fn get_foundry_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, FOUNDRY_BASE_COST))
    }

    fn get_builder_bot_cost(&self) -> PyResult<(i32, i32)> {
        self.check_deadline()?;
        let game = self.game();
        let team = game.entity(self.unit).expect("unknown unit").team;
        Ok(game.scaled_cost(team, BUILDER_BOT_BASE_COST))
    }

    fn r#move(&self, direction: crate::common::Direction) -> PyResult<()> {
        self.check_deadline()?;
        let dir = direction;
        game_assert(self.can_move(dir)?, "Cannot move")?;
        let mut game = self.game_mut();
        let bot = game.builder_bot_ref(self.unit)?;
        let to_pos = bot.position + dir;
        let bot_id = bot.id;
        game.move_builder_bot(bot_id, to_pos);
        Ok(())
    }

    fn can_move(&self, direction: crate::common::Direction) -> PyResult<bool> {
        self.check_deadline()?;
        let dir = direction;
        let game = self.game();
        let bot = match game.entity(self.unit) {
            Some(Entity::BuilderBot(bot)) => bot,
            _ => return Ok(false),
        };
        if !bot.can_move() {
            return Ok(false);
        }
        let to_pos = bot.position + dir;
        Ok(game.is_tile_bot_passable(to_pos, bot.team))
    }

    // --- can_* helpers (quality-of-life) ---

    fn can_build_conveyor(
        &self,
        position: Pos,
        direction: crate::common::Direction,
    ) -> PyResult<bool> {
        self.check_deadline()?;
        if !direction.is_cardinal() {
            return Ok(false);
        }
        Ok(self.can_build_checks(position, CONVEYOR_BASE_COST))
    }

    fn can_build_splitter(
        &self,
        position: Pos,
        direction: crate::common::Direction,
    ) -> PyResult<bool> {
        self.check_deadline()?;
        if !direction.is_cardinal() {
            return Ok(false);
        }
        Ok(self.can_build_checks(position, SPLITTER_BASE_COST))
    }

    fn can_build_bridge(&self, position: Pos, target: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        if !self.can_build_checks(position, BRIDGE_BASE_COST) {
            return Ok(false);
        }
        if !self.game().game_map.in_bounds(target) {
            return Ok(false);
        }
        let dist_sq = position.distance_squared(target);
        Ok(dist_sq > 0 && dist_sq <= BRIDGE_TARGET_RADIUS_SQ)
    }

    fn can_build_armoured_conveyor(
        &self,
        position: Pos,
        direction: crate::common::Direction,
    ) -> PyResult<bool> {
        self.check_deadline()?;
        if !direction.is_cardinal() {
            return Ok(false);
        }
        Ok(self.can_build_checks(position, ARMOURED_CONVEYOR_BASE_COST))
    }

    fn can_build_harvester(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        if !self.can_build_checks(position, HARVESTER_BASE_COST) {
            return Ok(false);
        }
        let game = self.game();
        if game.game_map.tile(position).builder_bot.is_some() {
            return Ok(false);
        }
        Ok(matches!(
            game.game_map.tile(position).environment,
            Environment::OreTitanium | Environment::OreAxionite
        ))
    }

    fn can_build_road(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        Ok(self.can_build_checks(position, ROAD_BASE_COST))
    }

    fn can_build_barrier(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        if !self.can_build_checks(position, BARRIER_BASE_COST) {
            return Ok(false);
        }
        Ok(self.game().game_map.tile(position).builder_bot.is_none())
    }

    fn can_build_gunner(
        &self,
        position: Pos,
        direction: crate::common::Direction,
    ) -> PyResult<bool> {
        self.check_deadline()?;
        if !direction.is_directional() {
            return Ok(false);
        }
        if !self.can_build_checks(position, GUNNER_BASE_COST) {
            return Ok(false);
        }
        Ok(self.game().game_map.tile(position).builder_bot.is_none())
    }

    fn can_build_sentinel(
        &self,
        position: Pos,
        direction: crate::common::Direction,
    ) -> PyResult<bool> {
        self.check_deadline()?;
        if !direction.is_directional() {
            return Ok(false);
        }
        if !self.can_build_checks(position, SENTINEL_BASE_COST) {
            return Ok(false);
        }
        Ok(self.game().game_map.tile(position).builder_bot.is_none())
    }

    fn can_build_breach(
        &self,
        position: Pos,
        direction: crate::common::Direction,
    ) -> PyResult<bool> {
        self.check_deadline()?;
        if !direction.is_directional() {
            return Ok(false);
        }
        if !self.can_build_checks(position, BREACH_BASE_COST) {
            return Ok(false);
        }
        Ok(self.game().game_map.tile(position).builder_bot.is_none())
    }

    fn can_build_launcher(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        if !self.can_build_checks(position, LAUNCHER_BASE_COST) {
            return Ok(false);
        }
        Ok(self.game().game_map.tile(position).builder_bot.is_none())
    }

    fn can_build_foundry(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        if !self.can_build_checks(position, FOUNDRY_BASE_COST) {
            return Ok(false);
        }
        Ok(self.game().game_map.tile(position).builder_bot.is_none())
    }

    fn build_conveyor(&self, position: Pos, direction: crate::common::Direction) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_conveyor(position, direction)?,
            "Cannot build conveyor",
        )?;
        let mut game = self.game_mut();
        game.build_conveyor(self.unit, position, direction);
        Ok(())
    }

    fn build_splitter(&self, position: Pos, direction: crate::common::Direction) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_splitter(position, direction)?,
            "Cannot build splitter",
        )?;
        let mut game = self.game_mut();
        game.build_splitter(self.unit, position, direction);
        Ok(())
    }

    fn build_bridge(&self, position: Pos, target: Pos) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_bridge(position, target)?,
            "Cannot build bridge",
        )?;
        let mut game = self.game_mut();
        game.build_bridge(self.unit, position, target);
        Ok(())
    }

    fn build_armoured_conveyor(
        &self,
        position: Pos,
        direction: crate::common::Direction,
    ) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_armoured_conveyor(position, direction)?,
            "Cannot build armoured conveyor",
        )?;
        let mut game = self.game_mut();
        game.build_armoured_conveyor(self.unit, position, direction);
        Ok(())
    }

    fn build_harvester(&self, position: Pos) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_harvester(position)?,
            "Cannot build harvester",
        )?;
        let mut game = self.game_mut();
        game.build_harvester(self.unit, position);
        Ok(())
    }

    fn build_road(&self, position: Pos) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(self.can_build_road(position)?, "Cannot build road")?;
        let mut game = self.game_mut();
        game.build_road(self.unit, position);
        Ok(())
    }

    fn build_barrier(&self, position: Pos) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(self.can_build_barrier(position)?, "Cannot build barrier")?;
        let mut game = self.game_mut();
        game.build_barrier(self.unit, position);
        Ok(())
    }

    fn build_gunner(&self, position: Pos, direction: crate::common::Direction) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_gunner(position, direction)?,
            "Cannot build gunner",
        )?;
        let mut game = self.game_mut();
        game.build_gunner(self.unit, position, direction);
        Ok(())
    }

    fn build_sentinel(&self, position: Pos, direction: crate::common::Direction) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_sentinel(position, direction)?,
            "Cannot build sentinel",
        )?;
        let mut game = self.game_mut();
        game.build_sentinel(self.unit, position, direction);
        Ok(())
    }

    fn build_breach(&self, position: Pos, direction: crate::common::Direction) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(
            self.can_build_breach(position, direction)?,
            "Cannot build breach",
        )?;
        let mut game = self.game_mut();
        game.build_breach(self.unit, position, direction);
        Ok(())
    }

    fn build_launcher(&self, position: Pos) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(self.can_build_launcher(position)?, "Cannot build launcher")?;
        let mut game = self.game_mut();
        game.build_launcher(self.unit, position);
        Ok(())
    }

    fn build_foundry(&self, position: Pos) -> PyResult<()> {
        self.check_deadline()?;
        game_assert(self.can_build_foundry(position)?, "Cannot build foundry")?;
        let mut game = self.game_mut();
        game.build_foundry(self.unit, position);
        Ok(())
    }

    fn can_destroy(&self, building_pos: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        let game = self.game();
        let bot = match game.entity(self.unit) {
            Some(Entity::BuilderBot(bot)) => bot,
            _ => return Ok(false),
        };
        if bot.position.distance_squared(building_pos) > ACTION_RADIUS_SQ {
            return Ok(false);
        }
        if !game.game_map.in_bounds(building_pos) {
            return Ok(false);
        }
        let tile = game.game_map.tile(building_pos);
        let Some(building_id) = tile.building else {
            return Ok(false);
        };
        if matches!(game.entity(building_id), Some(Entity::Core(_))) {
            return Ok(false);
        }
        let Some(building) = game
            .entity(building_id)
            .and_then(|entity| entity.as_building())
        else {
            return Ok(false);
        };
        Ok(building.team == bot.team)
    }

    fn destroy(&self, building_pos: Pos) -> PyResult<()> {
        self.check_deadline()?;
        let pos = building_pos;
        game_assert(self.can_destroy(pos)?, "Cannot destroy")?;
        let building_id = self
            .game()
            .game_map
            .tile(pos)
            .building
            .expect("can_destroy was true but tile had no building");
        let mut game = self.game_mut();
        game.destroy_entity(building_id);
        Ok(())
    }

    fn heal(&self, position: Pos) -> PyResult<()> {
        self.check_deadline()?;
        let pos = position;
        game_assert(self.can_heal(pos)?, "Cannot heal")?;
        let bot_id = self.unit;
        let team = self
            .game()
            .entity(bot_id)
            .and_then(|e| e.as_unit())
            .expect("healer is not a unit")
            .team;
        let mut game = self.game_mut();
        game.heal_tile(pos, team);
        let Some(Entity::BuilderBot(bot)) = game.entity_mut(bot_id) else {
            unreachable!()
        };
        bot.action_cooldown += 1;
        let cd = bot.action_cooldown;
        game.replay_recorder.append(GameDiff::SetActionCooldown {
            id: bot_id,
            value: cd,
        });
        Ok(())
    }

    fn can_heal(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        let pos = position;
        let game = self.game();
        let bot = match game.entity(self.unit) {
            Some(Entity::BuilderBot(bot)) => bot,
            _ => return Ok(false),
        };
        if !bot.can_act() {
            return Ok(false);
        }
        if bot.position.distance_squared(pos) > ACTION_RADIUS_SQ {
            return Ok(false);
        }
        if !game.game_map.in_bounds(pos) {
            return Ok(false);
        }
        // Require at least one friendly entity on the tile.
        let team = bot.team;
        let tile = game.game_map.tile(pos);
        let has_friendly = [tile.building, tile.builder_bot]
            .iter()
            .filter_map(|id| *id)
            .any(|id| game.entity(id).is_some_and(|e| e.team == team));
        Ok(has_friendly)
    }

    fn self_destruct(&self) -> PyResult<()> {
        self.check_deadline()?;
        let mut game = self.game_mut();
        let (pos, is_builder_bot) = match game.entity(self.unit) {
            Some(Entity::BuilderBot(bot)) => (bot.position, true),
            Some(entity) => match entity.as_unit() {
                Some(unit) => (unit.position, false),
                None => return Err(game_error("Unit is not a unit")),
            },
            None => return Err(game_error("Unknown unit")),
        };
        game.destroy_entity(self.unit);
        if is_builder_bot {
            game.damage_tile(pos, BUILDER_BOT_SELF_DESTRUCT_DAMAGE);
        }
        Ok(())
    }

    fn can_place_marker(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        if self.has_placed_marker.get() {
            return Ok(false);
        }
        let game = self.game();
        let entity = match game.entity(self.unit) {
            Some(e) => e,
            None => return Ok(false),
        };
        let unit = match entity.as_unit() {
            Some(u) => u,
            None => return Ok(false),
        };
        if !game.game_map.in_bounds(position) {
            return Ok(false);
        }
        if unit.position.distance_squared(position) > unit.action_radius_sq() {
            return Ok(false);
        }
        let tile = game.game_map.tile(position);
        if tile.environment == Environment::Wall {
            return Ok(false);
        }
        match tile.building {
            None => Ok(true),
            Some(id) => match game.entity(id) {
                Some(Entity::Marker(marker)) => Ok(marker.team == unit.team),
                _ => Ok(false),
            },
        }
    }

    fn place_marker(&self, position: Pos, value: u32) -> PyResult<()> {
        self.check_deadline()?;
        let pos = position;
        game_assert(self.can_place_marker(pos)?, "Cannot place marker")?;
        let team = self
            .game()
            .entity(self.unit)
            .and_then(|e| e.as_unit())
            .expect("can_place_marker was true but entity is not a unit")
            .team;
        let mut game = self.game_mut();
        game.place_marker(team, pos, value);
        self.has_placed_marker.set(true);
        Ok(())
    }

    fn get_marker_value(&self, id: i32) -> PyResult<u32> {
        self.check_deadline()?;
        self.assert_entity_in_vision(id)?;
        let game = self.game();
        let entity = game.entity(id).ok_or_else(|| game_error("Unknown id"))?;
        match entity {
            Entity::Marker(marker) => {
                let unit = game
                    .entity(self.unit)
                    .and_then(|e| e.as_unit())
                    .ok_or_else(|| game_error("Unit is not a unit"))?;
                game_assert(marker.team == unit.team, "Marker belongs to enemy team")?;
                Ok(marker.value)
            }
            _ => Err(game_error("Entity is not a marker")),
        }
    }

    fn get_gunner_target(&self, py: Python) -> PyResult<PyObject> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entity(self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        game_assert(matches!(entity, Entity::Gunner(_)), "Unit is not a gunner")?;
        match game.gunner_target(self.unit) {
            Some(pos) => Ok(pos.into_pyobject(py)?.unbind()),
            None => Ok(py.None()),
        }
    }

    fn can_fire(&self, target: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        let game = self.game();
        let entity = match game.entity(self.unit) {
            Some(e) => e,
            None => return Ok(false),
        };
        let turret = match entity.as_turret() {
            Some(t) => t,
            None => return Ok(false),
        };
        if !turret.can_act() || turret.ammo_amount <= 0 {
            return Ok(false);
        }
        match turret {
            Turret::Gunner(_) => Ok(game.gunner_target(self.unit) == Some(target)),
            Turret::Sentinel(_) => Ok(
                game.game_map.in_bounds(target) && game.sentinel_target_valid(self.unit, target)
            ),
            Turret::Breach(_) => {
                Ok(game.game_map.in_bounds(target) && game.breach_target_valid(self.unit, target))
            }
            Turret::Launcher(_) => {
                Err(game_error("Use can_launch() for launchers, not can_fire()"))
            }
        }
    }

    fn fire(&self, target: Pos) -> PyResult<()> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entity(self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let turret = entity
            .as_turret()
            .ok_or_else(|| game_error("Unit is not a turret"))?;
        if matches!(turret, Turret::Launcher(_)) {
            return Err(game_error("Use launch() for launchers, not fire()"));
        }
        game_assert(self.can_fire(target)?, "Cannot fire")?;
        let axionite = turret.ammo_type == Some(crate::common::ResourceType::RefinedAxionite);
        match turret {
            Turret::Gunner(_) => {
                drop(game);
                self.game_mut().fire_gunner(self.unit, axionite);
            }
            Turret::Sentinel(_) => {
                drop(game);
                self.game_mut().fire_sentinel(self.unit, target, axionite);
            }
            Turret::Breach(_) => {
                drop(game);
                self.game_mut().fire_breach(self.unit, target);
            }
            Turret::Launcher(_) => unreachable!(),
        }
        Ok(())
    }

    fn can_launch(&self, bot_pos: Pos, target: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        let game = self.game();
        let entity = match game.entity(self.unit) {
            Some(e) => e,
            None => return Ok(false),
        };
        let turret = match entity.as_turret() {
            Some(t) => t,
            None => return Ok(false),
        };
        if !matches!(turret, Turret::Launcher(_)) {
            return Ok(false);
        }
        if !turret.can_act() || turret.ammo_amount <= 0 {
            return Ok(false);
        }
        if !game.game_map.in_bounds(bot_pos) {
            return Ok(false);
        }
        let launcher_pos = turret.position;
        if launcher_pos.distance_squared(bot_pos) > ACTION_RADIUS_SQ {
            return Ok(false);
        }
        if game.game_map.tile(bot_pos).builder_bot.is_none() {
            return Ok(false);
        }
        if !game.launcher_target_valid(self.unit, target) {
            return Ok(false);
        }
        let bot_id = game.game_map.tile(bot_pos).builder_bot.unwrap();
        let bot_team = game.entity(bot_id).expect("unknown bot").team;
        Ok(game.is_tile_bot_passable(target, bot_team))
    }

    fn launch(&self, bot_pos: Pos, target: Pos) -> PyResult<()> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entity(self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let turret = entity
            .as_turret()
            .ok_or_else(|| game_error("Unit is not a turret"))?;
        game_assert(
            matches!(turret, Turret::Launcher(_)),
            "Unit is not a launcher",
        )?;
        game_assert(self.can_launch(bot_pos, target)?, "Cannot launch")?;
        let bot_id = game
            .game_map
            .tile(bot_pos)
            .builder_bot
            .expect("can_launch was true but no builder bot at bot_pos");
        drop(game);
        self.game_mut().fire_launcher(self.unit, bot_id, target);
        Ok(())
    }

    fn spawn_builder(&self, position: Pos) -> PyResult<()> {
        self.check_deadline()?;
        let pos = position;
        game_assert(self.can_spawn(pos)?, "Cannot spawn")?;
        let mut game = self.game_mut();
        game.spawn_builder(self.unit, pos);
        Ok(())
    }

    #[pyo3(signature = (dist_sq=None))]
    fn get_nearby_tiles(&self, dist_sq: Option<i32>) -> PyResult<Vec<Pos>> {
        self.check_deadline()?;
        let game = self.game();
        let entity = game
            .entities
            .get(&self.unit)
            .ok_or_else(|| game_error("Unknown id"))?;
        let unit = entity
            .as_unit()
            .ok_or_else(|| game_error("Unit is not a unit"))?;
        let pos = unit.position;
        let vision = unit.vision_radius_sq();
        if let Some(d) = dist_sq {
            game_assert(d <= vision, "dist_sq exceeds vision radius")?;
        }
        let radius_sq = dist_sq.unwrap_or(vision);
        let r = (radius_sq as f64).sqrt().ceil() as i32;
        let mut result = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > radius_sq {
                    continue;
                }
                let p = Pos {
                    x: pos.x + dx,
                    y: pos.y + dy,
                };
                if game.game_map.in_bounds(p) {
                    result.push(p);
                }
            }
        }
        Ok(result)
    }

    #[pyo3(signature = (dist_sq=None))]
    fn get_nearby_entities(&self, dist_sq: Option<i32>) -> PyResult<Vec<i32>> {
        let tiles = self.get_nearby_tiles(dist_sq)?;
        let game = self.game();
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for p in tiles {
            let tile = game.game_map.tile(p);
            for id in [tile.building, tile.builder_bot].into_iter().flatten() {
                if seen.insert(id) {
                    result.push(id);
                }
            }
        }
        Ok(result)
    }

    #[pyo3(signature = (dist_sq=None))]
    fn get_nearby_buildings(&self, dist_sq: Option<i32>) -> PyResult<Vec<i32>> {
        let all = self.get_nearby_entities(dist_sq)?;
        let game = self.game();
        Ok(all
            .into_iter()
            .filter(|id| game.entity(*id).and_then(|e| e.as_building()).is_some())
            .collect())
    }

    #[pyo3(signature = (dist_sq=None))]
    fn get_nearby_units(&self, dist_sq: Option<i32>) -> PyResult<Vec<i32>> {
        let all = self.get_nearby_entities(dist_sq)?;
        let game = self.game();
        Ok(all
            .into_iter()
            .filter(|id| game.entity(*id).and_then(|e| e.as_unit()).is_some())
            .collect())
    }

    fn can_spawn(&self, position: Pos) -> PyResult<bool> {
        self.check_deadline()?;
        let pos = position;
        let game = self.game();
        let core = match game.core_ref(self.unit) {
            Some(core) => core,
            None => return Ok(false),
        };
        if !core.can_act() {
            return Ok(false);
        }
        if core.position.distance_squared(pos) > CORE_SPAWNING_RADIUS_SQ {
            return Ok(false);
        }
        if !game.is_tile_bot_passable(pos, core.team) {
            return Ok(false);
        }
        let cost = game.scaled_cost(core.team, BUILDER_BOT_BASE_COST);
        Ok(game.players[core.team.index()].can_afford(cost))
    }

    fn draw_indicator_line(&self, pos_a: Pos, pos_b: Pos, r: i32, g: i32, b: i32) -> PyResult<()> {
        self.check_deadline()?;
        let mut game = self.game_mut();
        game.replay_recorder.append(GameDiff::IndicatorLine {
            id: self.unit,
            pos_a,
            pos_b,
            r,
            g,
            b,
        });
        Ok(())
    }

    fn draw_indicator_dot(&self, pos: Pos, r: i32, g: i32, b: i32) -> PyResult<()> {
        self.check_deadline()?;
        let mut game = self.game_mut();
        game.replay_recorder.append(GameDiff::IndicatorDot {
            id: self.unit,
            pos,
            r,
            g,
            b,
        });
        Ok(())
    }
}

impl Game {
    fn builder_bot_ref(&self, id: i32) -> PyResult<&BuilderBot> {
        let bot = match self.entity(id) {
            Some(Entity::BuilderBot(bot)) => bot,
            _ => {
                game_assert(false, "Unit is not a builder bot")?;
                unreachable!();
            }
        };
        Ok(bot)
    }

    fn core_ref(&self, id: i32) -> Option<&crate::game_map::Core> {
        match self.entity(id) {
            Some(Entity::Core(core)) => Some(core),
            _ => None,
        }
    }

    fn tile_at(&self, pos: Pos) -> PyResult<&Tile> {
        game_assert(self.game_map.in_bounds(pos), "Position out of bounds")?;
        Ok(self.game_map.tile(pos))
    }
}

impl Controller {
    fn check_deadline(&self) -> PyResult<()> {
        if crate::runner::thread_cpu_time_ns() >= self.cpu_deadline_ns {
            Err(pyo3::exceptions::PySystemExit::new_err(()))
        } else {
            Ok(())
        }
    }

    fn game(&self) -> Ref<'_, Game> {
        self.game.borrow()
    }

    fn game_mut(&self) -> RefMut<'_, Game> {
        self.game.borrow_mut()
    }

    fn entity(&self, id: i32) -> PyResult<Ref<'_, Entity>> {
        let game = self.game();
        if !game.entities.contains_key(&id) {
            return Err(game_error("Unknown id"));
        }
        Ok(Ref::map(game, |game| {
            game.entities
                .get(&id)
                .expect("id validated for entity lookup")
        }))
    }

    fn assert_in_vision(&self, pos: Pos) -> PyResult<()> {
        game_assert(self.is_in_vision(pos)?, "Position out of vision range")
    }

    fn assert_entity_in_vision(&self, target_id: i32) -> PyResult<()> {
        let game = self.game();
        let entity = game
            .entity(target_id)
            .ok_or_else(|| game_error("Unknown id"))?;
        let centre = entity.position;
        // The core occupies a 3x3 area; it counts as in-vision if any of its
        // 9 tiles is within range.
        if matches!(entity, Entity::Core(_)) {
            use crate::common::Direction::*;
            let in_vision = [
                North, Northeast, East, Southeast, South, Southwest, West, Northwest, Centre,
            ]
            .iter()
            .any(|d| self.is_in_vision(centre + *d).unwrap_or(false));
            return game_assert(in_vision, "Entity out of vision range");
        }
        self.assert_in_vision(centre)
    }

    // (build_checks removed; action methods now validate via can_* helpers)
}

#[pymodule]
pub fn controller_mod(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Controller>()?;
    Ok(())
}
