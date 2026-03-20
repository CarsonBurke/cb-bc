use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;
use std::str::FromStr;

use crate::common::{Direction, Environment, Pos, ResourceType, Team};
use crate::game_map::{Entity, Unit};

/// Cached references to Python cambc types. Initialized once at startup
/// so that IntoPyObject impls never call py.import("cambc") during gameplay.
/// This prevents module poisoning and eliminates import audit-event overhead.
struct PyTypeCache {
    // Team variants
    team_a: Py<PyAny>,
    team_b: Py<PyAny>,
    // Direction variants
    dir_north: Py<PyAny>,
    dir_northeast: Py<PyAny>,
    dir_east: Py<PyAny>,
    dir_southeast: Py<PyAny>,
    dir_south: Py<PyAny>,
    dir_southwest: Py<PyAny>,
    dir_west: Py<PyAny>,
    dir_northwest: Py<PyAny>,
    dir_centre: Py<PyAny>,
    // Environment variants
    env_empty: Py<PyAny>,
    env_wall: Py<PyAny>,
    env_ore_titanium: Py<PyAny>,
    env_ore_axionite: Py<PyAny>,
    // ResourceType variants
    res_titanium: Py<PyAny>,
    res_raw_axionite: Py<PyAny>,
    res_refined_axionite: Py<PyAny>,
    // EntityType variants
    et_builder_bot: Py<PyAny>,
    et_core: Py<PyAny>,
    et_gunner: Py<PyAny>,
    et_sentinel: Py<PyAny>,
    et_breach: Py<PyAny>,
    et_launcher: Py<PyAny>,
    et_conveyor: Py<PyAny>,
    et_splitter: Py<PyAny>,
    et_armoured_conveyor: Py<PyAny>,
    et_bridge: Py<PyAny>,
    et_harvester: Py<PyAny>,
    et_foundry: Py<PyAny>,
    et_road: Py<PyAny>,
    et_barrier: Py<PyAny>,
    et_marker: Py<PyAny>,
    // Position class (callable to construct instances)
    position_cls: Py<PyAny>,
    // GameError class
    game_error_cls: Py<PyAny>,
}

impl PyTypeCache {
    fn init(py: Python) -> PyResult<Self> {
        let bc = py.import("cambc")?;

        let team = bc.getattr("Team")?;
        let dir = bc.getattr("Direction")?;
        let env = bc.getattr("Environment")?;
        let res = bc.getattr("ResourceType")?;
        let et = bc.getattr("EntityType")?;

        Ok(Self {
            team_a: team.getattr("A")?.unbind(),
            team_b: team.getattr("B")?.unbind(),

            dir_north: dir.getattr("NORTH")?.unbind(),
            dir_northeast: dir.getattr("NORTHEAST")?.unbind(),
            dir_east: dir.getattr("EAST")?.unbind(),
            dir_southeast: dir.getattr("SOUTHEAST")?.unbind(),
            dir_south: dir.getattr("SOUTH")?.unbind(),
            dir_southwest: dir.getattr("SOUTHWEST")?.unbind(),
            dir_west: dir.getattr("WEST")?.unbind(),
            dir_northwest: dir.getattr("NORTHWEST")?.unbind(),
            dir_centre: dir.getattr("CENTRE")?.unbind(),

            env_empty: env.getattr("EMPTY")?.unbind(),
            env_wall: env.getattr("WALL")?.unbind(),
            env_ore_titanium: env.getattr("ORE_TITANIUM")?.unbind(),
            env_ore_axionite: env.getattr("ORE_AXIONITE")?.unbind(),

            res_titanium: res.getattr("TITANIUM")?.unbind(),
            res_raw_axionite: res.getattr("RAW_AXIONITE")?.unbind(),
            res_refined_axionite: res.getattr("REFINED_AXIONITE")?.unbind(),

            et_builder_bot: et.getattr("BUILDER_BOT")?.unbind(),
            et_core: et.getattr("CORE")?.unbind(),
            et_gunner: et.getattr("GUNNER")?.unbind(),
            et_sentinel: et.getattr("SENTINEL")?.unbind(),
            et_breach: et.getattr("BREACH")?.unbind(),
            et_launcher: et.getattr("LAUNCHER")?.unbind(),
            et_conveyor: et.getattr("CONVEYOR")?.unbind(),
            et_splitter: et.getattr("SPLITTER")?.unbind(),
            et_armoured_conveyor: et.getattr("ARMOURED_CONVEYOR")?.unbind(),
            et_bridge: et.getattr("BRIDGE")?.unbind(),
            et_harvester: et.getattr("HARVESTER")?.unbind(),
            et_foundry: et.getattr("FOUNDRY")?.unbind(),
            et_road: et.getattr("ROAD")?.unbind(),
            et_barrier: et.getattr("BARRIER")?.unbind(),
            et_marker: et.getattr("MARKER")?.unbind(),

            position_cls: bc.getattr("Position")?.unbind(),
            game_error_cls: bc.getattr("GameError")?.unbind(),
        })
    }
}

use pyo3_ffi::PyThreadState;
use std::collections::HashMap;

thread_local! {
    static PY_TYPE_CACHES: RefCell<HashMap<*mut PyThreadState, PyTypeCache>> =
        RefCell::new(HashMap::new());
}

fn current_tstate() -> *mut PyThreadState {
    unsafe { pyo3_ffi::PyThreadState_Get() }
}

/// Remove the cache entry for the currently active interpreter when it is being torn down.
/// Must be called while that interpreter is still active (before Py_EndInterpreter).
pub fn remove_type_cache() {
    PY_TYPE_CACHES.with(|m| m.borrow_mut().remove(&current_tstate()));
}

/// Must be called once per interpreter (after cambc is importable) while that
/// interpreter is active.
pub fn init_type_cache(py: Python) -> PyResult<()> {
    let cache = PyTypeCache::init(py)?;
    PY_TYPE_CACHES.with(|m| m.borrow_mut().insert(current_tstate(), cache));
    Ok(())
}

fn with_cache<F, R>(f: F) -> R
where
    F: FnOnce(&PyTypeCache) -> R,
{
    PY_TYPE_CACHES.with(|m| {
        let borrow = m.borrow();
        let cache = borrow
            .get(&current_tstate())
            .expect("PyTypeCache not initialized for active interpreter");
        f(cache)
    })
}

// ---------------------------------------------------------------------------
// IntoPyObject impls — all use the cache, never call py.import()
// ---------------------------------------------------------------------------

impl<'py> IntoPyObject<'py> for Team {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(with_cache(|c| match self {
            Team::A => c.team_a.clone_ref(py),
            Team::B => c.team_b.clone_ref(py),
        })
        .into_bound(py))
    }
}

impl<'py> IntoPyObject<'py> for Direction {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(with_cache(|c| match self {
            Direction::North => c.dir_north.clone_ref(py),
            Direction::Northeast => c.dir_northeast.clone_ref(py),
            Direction::East => c.dir_east.clone_ref(py),
            Direction::Southeast => c.dir_southeast.clone_ref(py),
            Direction::South => c.dir_south.clone_ref(py),
            Direction::Southwest => c.dir_southwest.clone_ref(py),
            Direction::West => c.dir_west.clone_ref(py),
            Direction::Northwest => c.dir_northwest.clone_ref(py),
            Direction::Centre => c.dir_centre.clone_ref(py),
        })
        .into_bound(py))
    }
}

impl<'py> IntoPyObject<'py> for Environment {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(with_cache(|c| match self {
            Environment::Empty => c.env_empty.clone_ref(py),
            Environment::Wall => c.env_wall.clone_ref(py),
            Environment::OreTitanium => c.env_ore_titanium.clone_ref(py),
            Environment::OreAxionite => c.env_ore_axionite.clone_ref(py),
        })
        .into_bound(py))
    }
}

impl<'py> IntoPyObject<'py> for ResourceType {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(with_cache(|c| match self {
            ResourceType::Titanium => c.res_titanium.clone_ref(py),
            ResourceType::RawAxionite => c.res_raw_axionite.clone_ref(py),
            ResourceType::RefinedAxionite => c.res_refined_axionite.clone_ref(py),
        })
        .into_bound(py))
    }
}

impl<'py> IntoPyObject<'py> for Pos {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        with_cache(|c| c.position_cls.call1(py, (self.x, self.y))).map(|obj| obj.into_bound(py))
    }
}

impl<'py> FromPyObject<'py> for Pos {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let (Ok(x_any), Ok(y_any)) = (ob.getattr("x"), ob.getattr("y")) {
            let x: i32 = x_any.extract()?;
            let y: i32 = y_any.extract()?;
            return Ok(Pos { x, y });
        }
        let (x, y): (i32, i32) = ob.extract()?;
        Ok(Pos { x, y })
    }
}

impl<'py> FromPyObject<'py> for Direction {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let value: String = ob.getattr("value")?.extract()?;
        Direction::from_str(&value)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid direction"))
    }
}

impl<'py> IntoPyObject<'py> for Unit<'py> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(with_cache(|c| match self {
            Unit::BuilderBot(_) => c.et_builder_bot.clone_ref(py),
            Unit::Core(_) => c.et_core.clone_ref(py),
            Unit::Gunner(_) => c.et_gunner.clone_ref(py),
            Unit::Sentinel(_) => c.et_sentinel.clone_ref(py),
            Unit::Breach(_) => c.et_breach.clone_ref(py),
            Unit::Launcher(_) => c.et_launcher.clone_ref(py),
        })
        .into_bound(py))
    }
}

impl<'py> IntoPyObject<'py> for Entity {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(with_cache(|c| match self {
            Entity::BuilderBot(_) => c.et_builder_bot.clone_ref(py),
            Entity::Core(_) => c.et_core.clone_ref(py),
            Entity::Gunner(_) => c.et_gunner.clone_ref(py),
            Entity::Sentinel(_) => c.et_sentinel.clone_ref(py),
            Entity::Breach(_) => c.et_breach.clone_ref(py),
            Entity::Launcher(_) => c.et_launcher.clone_ref(py),
            Entity::Conveyor(_) => c.et_conveyor.clone_ref(py),
            Entity::Splitter(_) => c.et_splitter.clone_ref(py),
            Entity::ArmouredConveyor(_) => c.et_armoured_conveyor.clone_ref(py),
            Entity::Bridge(_) => c.et_bridge.clone_ref(py),
            Entity::Harvester(_) => c.et_harvester.clone_ref(py),
            Entity::Foundry(_) => c.et_foundry.clone_ref(py),
            Entity::Road(_) => c.et_road.clone_ref(py),
            Entity::Barrier(_) => c.et_barrier.clone_ref(py),
            Entity::Marker(_) => c.et_marker.clone_ref(py),
        })
        .into_bound(py))
    }
}

/// Create a GameError using the cached class (never calls py.import).
pub fn game_error(message: &str) -> PyErr {
    Python::with_gil(|py| {
        PY_TYPE_CACHES.with(|m| {
            let borrow = m.borrow();
            if let Some(cache) = borrow.get(&current_tstate()) {
                if let Ok(err_cls) = cache
                    .game_error_cls
                    .bind(py)
                    .clone()
                    .downcast_into::<pyo3::types::PyType>()
                {
                    return PyErr::from_type(err_cls, message.to_string());
                }
            }
            pyo3::exceptions::PyValueError::new_err(message.to_string())
        })
    })
}
