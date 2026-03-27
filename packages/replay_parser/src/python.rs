//! PyO3 bindings exposing replay parsing to Python.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::fs;
use std::path::Path;

use crate::protobuf::*;
use crate::replay::{self, N_CHANNELS, N_SCALARS};
use crate::sequential::{self, BC_ACTION_SIZE};
use crate::types::*;

/// Reinterpret a typed slice as raw bytes (for zero-copy PyBytes).
unsafe fn as_bytes<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * std::mem::size_of::<T>())
}

/// Parse a replay into contiguous buffers for zero-copy tensor construction.
#[pyfunction]
#[pyo3(signature = (path, team=0))]
fn parse_replay_full<'py>(py: Python<'py>, path: &str, team: u8) -> PyResult<PyObject> {
    let data = fs::read(Path::new(path)).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e))
    })?;

    let rd = replay::process_replay(&data, team);
    let result = PyDict::new(py);

    result.set_item("states", PyBytes::new(py, unsafe { as_bytes(&rd.states) }))?;
    result.set_item("scalars", PyBytes::new(py, unsafe { as_bytes(&rd.scalars) }))?;
    result.set_item(
        "working_harvesters",
        PyBytes::new(py, unsafe { as_bytes(&rd.working_harvesters) }),
    )?;
    result.set_item(
        "resource_transfers",
        PyBytes::new(py, unsafe { as_bytes(&rd.resource_transfers) }),
    )?;
    result.set_item(
        "titanium_delivered",
        PyBytes::new(py, unsafe { as_bytes(&rd.titanium_delivered) }),
    )?;
    result.set_item(
        "axionite_delivered",
        PyBytes::new(py, unsafe { as_bytes(&rd.axionite_delivered) }),
    )?;

    result.set_item("map_w", rd.map_w)?;
    result.set_item("map_h", rd.map_h)?;
    result.set_item("n_turns", rd.n_turns)?;
    result.set_item("n_channels", N_CHANNELS)?;
    result.set_item("n_scalars", N_SCALARS)?;
    result.set_item("state_stride", N_CHANNELS * rd.map_h * rd.map_w)?;
    result.set_item("winner", rd.winner)?;

    Ok(result.into())
}

/// Minimal parse: just extract working harvesters and winner.
#[pyfunction]
fn parse_replay_working(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let data = fs::read(Path::new(path)).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e))
    })?;

    let rd = replay::process_replay(&data, 0);
    let result = PyDict::new(py);

    result.set_item(
        "working_harvesters",
        PyBytes::new(py, unsafe { as_bytes(&rd.working_harvesters) }),
    )?;
    result.set_item("n_turns", rd.n_turns)?;
    result.set_item("winner", rd.winner)?;

    Ok(result.into())
}

/// Parse replay and return full Python dict (compatible with Python replay_parser.parse_replay).
#[pyfunction]
fn parse_replay(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let data = fs::read(Path::new(path)).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e))
    })?;

    let top = decode_fields(&data, 0, data.len());
    let result = PyDict::new(py);

    if let Some(w) = get_varint(&top, 4) {
        result.set_item("winner", w)?;
    }

    if let Some(map_data) = get_submsg(&top, 1, &data) {
        let map = decode_map(map_data);
        let map_dict = PyDict::new(py);
        map_dict.set_item("width", map.width)?;
        map_dict.set_item("height", map.height)?;

        let rows_list = PyList::empty(py);
        for y in 0..map.height {
            let row_dict = PyDict::new(py);
            let tiles: Vec<u8> = (0..map.width)
                .map(|x| map.env_grid[y * map.width + x])
                .collect();
            row_dict.set_item("tiles", tiles)?;
            rows_list.append(row_dict)?;
        }
        map_dict.set_item("rows", rows_list)?;

        let cores_list = PyList::empty(py);
        for (id, team, pos) in &map.cores {
            let core_dict = PyDict::new(py);
            core_dict.set_item("id", *id)?;
            core_dict.set_item("team", *team)?;
            let pos_dict = PyDict::new(py);
            pos_dict.set_item("x", pos.x)?;
            pos_dict.set_item("y", pos.y)?;
            core_dict.set_item("position", pos_dict)?;
            cores_list.append(core_dict)?;
        }
        map_dict.set_item("cores", cores_list)?;

        result.set_item("map", map_dict)?;
    }

    let turns_data = get_repeated_submsg(&top, 3, &data);
    let turns_list = PyList::empty(py);
    for turn_data in &turns_data {
        let turn_fields = decode_fields(turn_data, 0, turn_data.len());
        let updates_data = get_repeated_submsg(&turn_fields, 1, turn_data);
        let updates_list = PyList::empty(py);

        for update_data in &updates_data {
            let update_dict = PyDict::new(py);
            let fields = decode_fields(update_data, 0, update_data.len());

            if let Some(sub) = get_submsg(&fields, 1, update_data) {
                let pf = decode_fields(sub, 0, sub.len());
                if let Some(ent_data) = get_submsg(&pf, 1, sub) {
                    let ent = decode_entity(ent_data);
                    let ent_dict = entity_to_pydict(py, &ent)?;
                    let pe_dict = PyDict::new(py);
                    pe_dict.set_item("entity", ent_dict)?;
                    update_dict.set_item("placeEntity", pe_dict)?;
                }
            } else if let Some(sub) = get_submsg(&fields, 2, update_data) {
                let mf = decode_fields(sub, 0, sub.len());
                let mv_dict = PyDict::new(py);
                mv_dict.set_item("id", get_i32(&mf, 1))?;
                if let Some(to_data) = get_submsg(&mf, 2, sub) {
                    let to = decode_pos(to_data);
                    let to_dict = PyDict::new(py);
                    to_dict.set_item("x", to.x)?;
                    to_dict.set_item("y", to.y)?;
                    mv_dict.set_item("to", to_dict)?;
                }
                update_dict.set_item("moveBuilderBot", mv_dict)?;
            } else if let Some(sub) = get_submsg(&fields, 3, update_data) {
                let rf = decode_fields(sub, 0, sub.len());
                let re_dict = PyDict::new(py);
                re_dict.set_item("id", get_i32(&rf, 1))?;
                update_dict.set_item("removeEntity", re_dict)?;
            } else if let Some(sub) = get_submsg(&fields, 4, update_data) {
                let df = decode_fields(sub, 0, sub.len());
                let dr_dict = PyDict::new(py);
                let moves_list = PyList::empty(py);
                for mv_data in get_repeated_submsg(&df, 1, sub) {
                    let mvf = decode_fields(mv_data, 0, mv_data.len());
                    let mv_dict = PyDict::new(py);
                    if let Some(from_data) = get_submsg(&mvf, 1, mv_data) {
                        let from = decode_pos(from_data);
                        let d = PyDict::new(py);
                        d.set_item("x", from.x)?;
                        d.set_item("y", from.y)?;
                        mv_dict.set_item("from", d)?;
                    }
                    if let Some(to_data) = get_submsg(&mvf, 2, mv_data) {
                        let to = decode_pos(to_data);
                        let d = PyDict::new(py);
                        d.set_item("x", to.x)?;
                        d.set_item("y", to.y)?;
                        mv_dict.set_item("to", d)?;
                    }
                    moves_list.append(mv_dict)?;
                }
                dr_dict.set_item("moves", moves_list)?;
                update_dict.set_item("distributeResources", dr_dict)?;
            } else if let Some(sub) = get_submsg(&fields, 5, update_data) {
                let hf = decode_fields(sub, 0, sub.len());
                let hp_dict = PyDict::new(py);
                hp_dict.set_item("id", get_i32(&hf, 1))?;
                hp_dict.set_item("delta", get_i32(&hf, 2))?;
                update_dict.set_item("updateHp", hp_dict)?;
            } else if let Some(sub) = get_submsg(&fields, 6, update_data) {
                let uf = decode_fields(sub, 0, sub.len());
                let up_dict = PyDict::new(py);
                if let Some(players_data) = get_submsg(&uf, 1, sub) {
                    let pf = decode_fields(players_data, 0, players_data.len());
                    let players_dict = PyDict::new(py);
                    if let Some(a_data) = get_submsg(&pf, 1, players_data) {
                        let a = decode_player_res(a_data);
                        let d = PyDict::new(py);
                        d.set_item("titanium", a.titanium)?;
                        d.set_item("axionite", a.axionite)?;
                        d.set_item("resourcesCollected", a.resources_collected)?;
                        d.set_item("titaniumCollected", a.titanium_collected)?;
                        d.set_item("axioniteCollected", a.axionite_collected)?;
                        players_dict.set_item("a", d)?;
                    }
                    if let Some(b_data) = get_submsg(&pf, 2, players_data) {
                        let b = decode_player_res(b_data);
                        let d = PyDict::new(py);
                        d.set_item("titanium", b.titanium)?;
                        d.set_item("axionite", b.axionite)?;
                        d.set_item("resourcesCollected", b.resources_collected)?;
                        d.set_item("titaniumCollected", b.titanium_collected)?;
                        d.set_item("axioniteCollected", b.axionite_collected)?;
                        players_dict.set_item("b", d)?;
                    }
                    up_dict.set_item("players", players_dict)?;
                }
                update_dict.set_item("updatePlayers", up_dict)?;
            } else if let Some(sub) = get_submsg(&fields, 7, update_data) {
                let cf = decode_fields(sub, 0, sub.len());
                let cd_dict = PyDict::new(py);
                cd_dict.set_item("id", get_i32(&cf, 1))?;
                cd_dict.set_item("value", get_i32(&cf, 2))?;
                update_dict.set_item("setActionCooldown", cd_dict)?;
            } else if let Some(sub) = get_submsg(&fields, 8, update_data) {
                let cf = decode_fields(sub, 0, sub.len());
                let cd_dict = PyDict::new(py);
                cd_dict.set_item("id", get_i32(&cf, 1))?;
                cd_dict.set_item("value", get_i32(&cf, 2))?;
                update_dict.set_item("setMoveCooldown", cd_dict)?;
            }

            updates_list.append(update_dict)?;
        }

        let turn_dict = PyDict::new(py);
        turn_dict.set_item("updates", updates_list)?;
        turns_list.append(turn_dict)?;
    }
    result.set_item("turns", turns_list)?;

    Ok(result.into())
}

/// Parse replay sequentially for behavioral cloning.
#[pyfunction]
#[pyo3(signature = (path, team=0))]
fn parse_replay_sequential<'py>(py: Python<'py>, path: &str, team: u8) -> PyResult<PyObject> {
    let data = fs::read(Path::new(path)).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e))
    })?;

    let bc = sequential::process_replay_sequential(&data, team);
    let result = PyDict::new(py);

    result.set_item("obs", PyBytes::new(py, unsafe { as_bytes(&bc.obs) }))?;
    result.set_item("actions", PyBytes::new(py, &bc.actions))?;
    result.set_item("turns", PyBytes::new(py, unsafe { as_bytes(&bc.turns) }))?;
    result.set_item("bot_ids", PyBytes::new(py, unsafe { as_bytes(&bc.bot_ids) }))?;

    result.set_item("n_samples", bc.n_samples)?;
    result.set_item("obs_size", bc.obs_size)?;
    result.set_item("action_size", BC_ACTION_SIZE)?;
    result.set_item("winner", bc.winner)?;
    result.set_item("map_w", bc.map_w)?;
    result.set_item("map_h", bc.map_h)?;

    Ok(result.into())
}

fn entity_to_pydict(py: Python<'_>, ent: &Entity) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("id", ent.id)?;
    d.set_item("team", ent.team)?;
    let pos = PyDict::new(py);
    pos.set_item("x", ent.x)?;
    pos.set_item("y", ent.y)?;
    d.set_item("position", pos)?;
    d.set_item("hp", ent.hp)?;
    d.set_item("maxHp", ent.max_hp)?;

    let kind_dict = PyDict::new(py);
    match ent.kind {
        EntityKind::BuilderBot => {
            kind_dict.set_item("actionCooldown", ent.action_cooldown)?;
            kind_dict.set_item("moveCooldown", ent.move_cooldown)?;
        }
        EntityKind::Conveyor | EntityKind::Splitter | EntityKind::ArmouredConveyor => {
            kind_dict.set_item("direction", ent.direction)?;
            kind_dict.set_item("stored", ent.stored)?;
        }
        EntityKind::Harvester => {
            kind_dict.set_item("cooldown", ent.cooldown)?;
            kind_dict.set_item("resourceType", ent.stored)?;
        }
        EntityKind::Core => {
            kind_dict.set_item("actionCooldown", ent.action_cooldown)?;
        }
        EntityKind::Gunner | EntityKind::Sentinel | EntityKind::Breach | EntityKind::Launcher => {
            kind_dict.set_item("direction", ent.direction)?;
            kind_dict.set_item("ammoType", ent.stored)?;
        }
        _ => {}
    }
    d.set_item(ent.kind.proto_name(), kind_dict)?;

    Ok(d.into())
}

/// Parse multiple replays in parallel (releases GIL, uses rayon thread pool).
///
/// Args:
///   jobs: list of (path, team) tuples
///
/// Returns: list of dicts, same format as parse_replay_sequential per entry.
#[pyfunction]
fn parse_replays_sequential_batch(py: Python<'_>, jobs: Vec<(String, u8)>) -> PyResult<PyObject> {
    use rayon::prelude::*;

    // Read files and parse in parallel (GIL released)
    let results: Vec<Result<(Vec<u8>, sequential::BCSamples), String>> =
        py.allow_threads(|| {
            jobs.par_iter()
                .map(|(path, team)| {
                    let data = fs::read(Path::new(path))
                        .map_err(|e| format!("Failed to read {}: {}", path, e))?;
                    let bc = sequential::process_replay_sequential(&data, *team);
                    Ok((data, bc))
                })
                .collect()
        });

    // Convert to Python dicts (needs GIL)
    let out = PyList::empty(py);
    for res in results {
        match res {
            Ok((_data, bc)) => {
                let d = PyDict::new(py);
                d.set_item("obs", PyBytes::new(py, unsafe { as_bytes(&bc.obs) }))?;
                d.set_item("actions", PyBytes::new(py, &bc.actions))?;
                d.set_item("turns", PyBytes::new(py, unsafe { as_bytes(&bc.turns) }))?;
                d.set_item("bot_ids", PyBytes::new(py, unsafe { as_bytes(&bc.bot_ids) }))?;
                d.set_item("n_samples", bc.n_samples)?;
                d.set_item("obs_size", bc.obs_size)?;
                d.set_item("action_size", BC_ACTION_SIZE)?;
                d.set_item("winner", bc.winner)?;
                d.set_item("map_w", bc.map_w)?;
                d.set_item("map_h", bc.map_h)?;
                out.append(d)?;
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyIOError::new_err(e));
            }
        }
    }

    Ok(out.into())
}

/// Parse multiple replays for working harvester data in parallel.
///
/// Args:
///   paths: list of file paths
///
/// Returns: list of dicts with {working_harvesters, n_turns, winner}
#[pyfunction]
fn parse_replays_working_batch(py: Python<'_>, paths: Vec<String>) -> PyResult<PyObject> {
    use rayon::prelude::*;

    let results: Vec<Result<replay::ReplayData, String>> = py.allow_threads(|| {
        paths
            .par_iter()
            .map(|path| {
                let data = fs::read(Path::new(path))
                    .map_err(|e| format!("Failed to read {}: {}", path, e))?;
                Ok(replay::process_replay(&data, 0))
            })
            .collect()
    });

    let out = PyList::empty(py);
    for res in results {
        match res {
            Ok(rd) => {
                let d = PyDict::new(py);
                d.set_item(
                    "working_harvesters",
                    PyBytes::new(py, unsafe { as_bytes(&rd.working_harvesters) }),
                )?;
                d.set_item("n_turns", rd.n_turns)?;
                d.set_item("winner", rd.winner)?;
                out.append(d)?;
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyIOError::new_err(e));
            }
        }
    }

    Ok(out.into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_replay_full, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replay_working, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replay, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replay_sequential, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replays_sequential_batch, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replays_working_batch, m)?)?;
    Ok(())
}
