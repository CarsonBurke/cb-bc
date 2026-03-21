mod watchdog;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3_ffi::{
    Py_EndInterpreter, Py_NewInterpreterFromConfig, PyInterpreterConfig,
    PyInterpreterConfig_SHARED_GIL, PyThreadState, PyThreadState_Swap,
};

use crate::bindings as rustlib;
use crate::bindings::controller::Controller;
use crate::bindings::py_convert;
use crate::cli::Args;
use crate::common::game_constants::MAX_TURNS;
use crate::common::Team;
use crate::game::Game;
use crate::map_loader;
use crate::replay::recorder::GameDiff;

/// Returns the calling thread's cumulative CPU time in nanoseconds.
/// Uses CLOCK_THREAD_CPUTIME_ID — immune to noisy neighbors and OS scheduling.
/// Returns 0 when the `tle` feature is disabled.
#[cfg(feature = "tle")]
pub fn thread_cpu_time_ns() -> u64 {
    cpu_time_ns_for_clock(libc::CLOCK_THREAD_CPUTIME_ID)
}

#[cfg(feature = "tle")]
fn cpu_time_ns_for_clock(clock_id: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(clock_id, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(not(feature = "tle"))]
pub fn thread_cpu_time_ns() -> u64 {
    0
}

/// CPU clock ID for the main thread, obtained via pthread_getcpuclockid.
/// Stored at startup so the watchdog thread can measure main-thread CPU time.
#[cfg(feature = "tle")]
static MAIN_THREAD_CLOCK_ID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Call once on the main thread at startup to capture its CPU clock ID.
#[cfg(feature = "tle")]
fn init_main_thread_clock_id() {
    let mut clock_id: libc::clockid_t = 0;
    unsafe {
        libc::pthread_getcpuclockid(libc::pthread_self(), &mut clock_id);
    }
    MAIN_THREAD_CLOCK_ID.store(clock_id, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(feature = "tle"))]
fn init_main_thread_clock_id() {}

/// Returns the main thread's elapsed CPU time in nanoseconds.
/// Safe to call from any thread.
#[cfg(feature = "tle")]
pub fn main_thread_cpu_time_ns() -> u64 {
    let clock_id = MAIN_THREAD_CLOCK_ID.load(std::sync::atomic::Ordering::Relaxed);
    cpu_time_ns_for_clock(clock_id)
}

#[cfg(not(feature = "tle"))]
pub fn main_thread_cpu_time_ns() -> u64 {
    0
}

/// CPU deadline (absolute ns, in main-thread CPU time) written before arming the watchdog.
static CPU_DEADLINE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Python-callable that returns the main thread's CPU time remaining in the current budget (ns).
/// Positive means budget not yet reached; negative means over budget.
/// Safe to call from the watchdog thread.
#[pyfunction]
fn py_cpu_time_remaining_ns() -> i64 {
    let deadline = CPU_DEADLINE_NS.load(std::sync::atomic::Ordering::Relaxed);
    deadline as i64 - main_thread_cpu_time_ns() as i64
}

/// Rust-backed `PyThreadState_SetAsyncExc` wrapper — called by the watchdog
/// to fire a SystemExit on the main thread.  Bypasses ctypes entirely so the
/// audit-hook sandbox never sees a `ctypes.call_function` event.
#[pyfunction]
fn py_fire_system_exit(tid: std::ffi::c_ulong) {
    extern "C" {
        fn PyThreadState_SetAsyncExc(
            id: std::ffi::c_ulong,
            exc: *mut pyo3_ffi::PyObject,
        ) -> std::ffi::c_int;
    }
    unsafe {
        PyThreadState_SetAsyncExc(tid, pyo3_ffi::PyExc_SystemExit);
    }
}

/// Clear any pending async exception on the given thread.
#[pyfunction]
fn py_clear_async_exc_for(tid: std::ffi::c_ulong) {
    extern "C" {
        fn PyThreadState_SetAsyncExc(
            id: std::ffi::c_ulong,
            exc: *mut pyo3_ffi::PyObject,
        ) -> std::ffi::c_int;
    }
    unsafe {
        PyThreadState_SetAsyncExc(tid, std::ptr::null_mut());
    }
}

/// Percent of the per-turn time limit that can be banked as extra time.
const ADAPTIVE_TIME_PERCENT: u64 = 5;

struct UnitRunner {
    player: Py<PyAny>,
    tstate: *mut PyThreadState,
    watchdog: watchdog::Watchdog,
}

struct GameRunner {
    game: Rc<RefCell<Game>>,
    /// Per-team bot source paths, used when initialising a new subinterpreter.
    bot_paths: [String; 2],
    /// Engine root path, used when initialising a new subinterpreter's sys.path.
    engine_root: std::path::PathBuf,
    /// The main interpreter's thread state, captured at construction time.
    /// We swap back to this after entering/leaving each unit's subinterpreter.
    main_tstate: *mut PyThreadState,
    /// Per-unit runner: player object + subinterpreter thread state + watchdog.
    unit_runners: HashMap<i32, UnitRunner>,
    turn_timeout_ms: u64,
    /// Banked extra time per bot (nanoseconds), used to absorb CPU jitter.
    bot_extra_time_ns: HashMap<i32, u64>,
    // Cached module references to avoid py.import() during gameplay
    sys_mod: Py<PyModule>,
    io_mod: Py<PyModule>,
    gc_mod: Py<PyModule>,
}

impl GameRunner {
    fn run(&mut self, py: Python) -> PyResult<()> {
        let sys = self.sys_mod.bind(py);
        sys.call_method1("setswitchinterval", (0.0001,))?;
        let gc = self.gc_mod.clone_ref(py).into_bound(py);
        gc.call_method0("disable")?;
        for i in 0..MAX_TURNS {
            self.run_turn(py, &gc)?;
            if i % 100 == 0 {
                println!("Completed turn {}", i);
            }
            if self.game.borrow_mut().winner_team().is_some() {
                break;
            }
        }
        gc.call_method0("enable")?;
        Ok(())
    }

    fn run_turn(&mut self, py: Python, gc: &Bound<'_, PyModule>) -> PyResult<()> {
        self.game.borrow_mut().new_turn();

        self.perform_unit_actions(py)?;
        let mut game = self.game.borrow_mut();
        game.distribute_resources();
        game.update_cooldowns();
        let players = game.players.clone();
        game.replay_recorder
            .append(GameDiff::UpdatePlayers { players });
        game.turn += 1;
        drop(game);
        self.cleanup_unit_runners();
        gc.call_method0("collect")?;
        Ok(())
    }

    fn perform_unit_actions(&mut self, py: Python) -> PyResult<()> {
        let cpu_budget_ns = self.turn_timeout_ms * 1_000_000;
        let max_extra_ns = cpu_budget_ns * ADAPTIVE_TIME_PERCENT / 100;
        let units = { self.game.borrow().unit_order.clone() };
        for unit_id in units {
            let game = self.game.borrow();
            let Some(entity) = game.entity(unit_id) else {
                continue;
            };
            let team = entity.team;
            drop(game);
            self.ensure_unit_runner(py, unit_id, team);

            let runner = &self.unit_runners[&unit_id];
            let player = runner.player.clone_ref(py);
            let wd_arm = runner.watchdog.arm.clone_ref(py);
            let wd_disarm = runner.watchdog.disarm.clone_ref(py);
            let wd_clear = runner.watchdog.clear_async_exc.clone_ref(py);

            let extra_ns = *self
                .bot_extra_time_ns
                .entry(unit_id)
                .or_insert(max_extra_ns);
            let effective_budget_ns = cpu_budget_ns + extra_ns;

            let wall_timeout = effective_budget_ns as f64 / 1_000_000_000.0 * 1.05;

            loop {
                match wd_clear.call0(py) {
                    Ok(_) => break,
                    Err(ref e) if e.is_instance_of::<pyo3::exceptions::PySystemExit>(py) => {
                        continue
                    }
                    Err(e) => panic!("clear_async_exc failed: {e}"),
                }
            }

            // Create a StringIO buffer for capturing this unit's stdout.
            let io = self.io_mod.bind(py);
            let stdout_buf = io.call_method0("StringIO")?.unbind();

            loop {
                match wd_arm.call1(py, (wall_timeout,)) {
                    Ok(_) => break,
                    Err(ref e) if e.is_instance_of::<pyo3::exceptions::PySystemExit>(py) => {
                        continue
                    }
                    Err(e) => panic!("arm failed: {e}"),
                }
            }

            // Swap into the unit's subinterpreter and redirect its sys.stdout
            let sub_tstate = self.unit_runners[&unit_id].tstate;
            unsafe { PyThreadState_Swap(sub_tstate); }
            let sub_sys = PyModule::import(py, "sys")?;
            sub_sys.setattr("stdout", stdout_buf.bind(py))?;

            // Create the Controller inside the subinterpreter so its type
            // is resolved in the same interpreter where bot code runs.
            // Creating it in the main interpreter caused intermittent
            // AttributeError on method access due to cross-interpreter
            // type cache mismatches.
            let controller = Py::new(py, Controller::new(self.game.clone(), unit_id))
                .expect("Py::new controller");

            // Set the deadline as late as possible so setup overhead isn't
            // counted against the bot's CPU budget.
            let cpu_start = thread_cpu_time_ns();
            let cpu_deadline_ns = cpu_start + effective_budget_ns;
            CPU_DEADLINE_NS.store(cpu_deadline_ns, std::sync::atomic::Ordering::Relaxed);
            controller.borrow_mut(py).set_cpu_deadline(cpu_start, cpu_deadline_ns);

            let result = player.call_method1(py, "run", (controller,));
            let cpu_elapsed_ns = thread_cpu_time_ns() - cpu_start;

            unsafe { PyThreadState_Swap(self.main_tstate); }

            // Disarm the watchdog so no new async exceptions can fire
            loop {
                match wd_disarm.call0(py) {
                    Ok(_) => break,
                    Err(ref e) if e.is_instance_of::<pyo3::exceptions::PySystemExit>(py) => {
                        continue
                    }
                    Err(e) => panic!("disarm failed: {e}"),
                }
            }
            // Clear any already-injected exception before touching stdout
            loop {
                match wd_clear.call0(py) {
                    Ok(_) => break,
                    Err(ref e) if e.is_instance_of::<pyo3::exceptions::PySystemExit>(py) => {
                        continue
                    }
                    Err(e) => panic!("clear_async_exc failed: {e}"),
                }
            }

            let stdout: String = stdout_buf
                .call_method0(py, "getvalue")
                .and_then(|s| s.extract(py))
                .unwrap_or_default();

            self.game
                .borrow_mut()
                .replay_recorder
                .append(GameDiff::BotOutput {
                    id: unit_id,
                    stdout,
                    exec_time_us: (cpu_elapsed_ns / 1000) as u32,
                    tled: cpu_elapsed_ns > effective_budget_ns,
                });

            let cpu_tle = cpu_elapsed_ns > effective_budget_ns;

            if let Err(ref err) = result {
                if err.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                    return Err(err.clone_ref(py));
                } else if err.is_instance_of::<pyo3::exceptions::PySystemExit>(py) {
                    // Watchdog or check_deadline fired — skip this unit's turn.
                } else {
                    unsafe { PyThreadState_Swap(sub_tstate); }
                    err.print(py);
                    unsafe { PyThreadState_Swap(self.main_tstate); }
                    let mut game = self.game.borrow_mut();
                    if game.entity(unit_id).is_some() {
                        game.destroy_entity(unit_id);
                    }
                }
            } else if cpu_tle {
                // Bot returned normally but exceeded CPU budget (tight loop
                // that finished just before the watchdog fired). Skip turn.
            }

            // Update banked extra time: bank unused time, debit overuse.
            let new_extra = if cpu_elapsed_ns <= cpu_budget_ns + extra_ns {
                (extra_ns + cpu_budget_ns).saturating_sub(cpu_elapsed_ns)
            } else {
                0
            };
            self.bot_extra_time_ns
                .insert(unit_id, new_extra.min(max_extra_ns));
        }
        Ok(())
    }

    fn ensure_unit_runner(&mut self, py: Python, unit: i32, team: Team) {
        if self.unit_runners.contains_key(&unit) {
            return;
        }

        // Grab the already-registered cambc module from the main interpreter
        // before creating the subinterpreter (which swaps away from main).
        // We inject it into the subinterpreter's sys.modules so the PyO3
        // extension module isn't re-initialized (it doesn't support subinterpreters).
        let cambc_mod = py.import("cambc").unwrap().unbind();

        // Create a new subinterpreter for this unit (SHARED_GIL — compatible
        // with extension modules like our Rust bindings).
        // NOTE: Py_NewInterpreterFromConfig swaps the current thread state to
        // the new subinterpreter on success, so all Python calls after this
        // point run inside the subinterpreter until we swap back.
        let sub_tstate: *mut PyThreadState = unsafe {
            let config = PyInterpreterConfig {
                use_main_obmalloc: 1,
                allow_fork: 0,
                allow_exec: 0,
                allow_threads: 1,
                allow_daemon_threads: 1,
                check_multi_interp_extensions: 0,
                gil: PyInterpreterConfig_SHARED_GIL,
            };
            let mut tstate: *mut PyThreadState = std::ptr::null_mut();
            let status = Py_NewInterpreterFromConfig(&mut tstate, &config);
            if pyo3_ffi::PyStatus_IsError(status) != 0 || tstate.is_null() {
                panic!("Py_NewInterpreterFromConfig failed");
            }
            tstate
        };

        // sub_tstate is now the current thread state (Py_NewInterpreterFromConfig
        // swaps in the new subinterpreter's thread state on success).
        // Set up sys.path, inject the shared cambc module, init the type cache,
        // then load the bot and instantiate the player.
        // Also create the watchdog here, inside the subinterpreter, so that
        // PyThreadState_SetAsyncExc targets the correct interpreter's thread state.
        let result: PyResult<(Py<PyAny>, watchdog::Watchdog)> = (|| {
            ensure_sys_path(py, &self.engine_root)?;
            // Inject the shared cambc module rather than re-initialising it.
            let sys = PyModule::import(py, "sys")?;
            sys.getattr("modules")?.set_item("cambc", cambc_mod.bind(py))?;
            py_convert::init_type_cache(py)?;
            let player_cls = load_player_class(py, &self.bot_paths[team.index()])?;
            let player = player_cls.call0(py)?;
            // Create a watchdog inside this subinterpreter so async exceptions
            // are injected into the correct interpreter's execution context.
            let cpu_time_fn: Py<PyAny> = wrap_pyfunction!(py_cpu_time_remaining_ns, py)?
                .unbind()
                .into();
            let fire_fn: Py<PyAny> = wrap_pyfunction!(py_fire_system_exit, py)?
                .unbind()
                .into();
            let clear_fn: Py<PyAny> = wrap_pyfunction!(py_clear_async_exc_for, py)?
                .unbind()
                .into();
            let wd = watchdog::Watchdog::new(py, cpu_time_fn, fire_fn, clear_fn);
            Ok((player, wd))
        })();

        // Swap back to the main interpreter's thread state.
        let main_tstate = self.main_tstate;
        unsafe { PyThreadState_Swap(main_tstate); }

        let (player, wd) = result.expect("failed to init unit runner");

        self.unit_runners.insert(unit, UnitRunner { player, tstate: sub_tstate, watchdog: wd });
    }

    fn cleanup_unit_runners(&mut self) {
        let game = self.game.borrow();
        let dead: Vec<i32> = self
            .unit_runners
            .keys()
            .copied()
            .filter(|id| game.entity(*id).is_none())
            .collect();
        drop(game);

        self.end_subinterpreters(&dead);
    }

    /// End subinterpreters for the given unit IDs and remove their runners.
    fn end_subinterpreters(&mut self, ids: &[i32]) {
        let main_tstate = self.main_tstate;
        for &id in ids {
            let UnitRunner { player, tstate, watchdog } = self.unit_runners.remove(&id).unwrap();
            self.bot_extra_time_ns.remove(&id);
            unsafe {
                PyThreadState_Swap(tstate);
                Python::with_gil(|py| {
                    // Signal the watchdog thread to exit and wait for it to
                    // finish before calling Py_EndInterpreter.  The thread
                    // must be gone before we destroy the interpreter or CPython
                    // aborts with "Py_EndInterpreter: not the last thread".
                    let _ = watchdog.shutdown.call0(py);
                    let _ = watchdog.thread.call_method0(py, "join");
                    drop(watchdog);

                    // Drop the player Py<PyAny> while the subinterpreter is still
                    // active so Py_DECREF (and any __del__) runs in the correct
                    // interpreter context.
                    drop(player);
                    // Remove the shared cambc module from this subinterpreter's
                    // sys.modules BEFORE Py_EndInterpreter.  CPython's interpreter
                    // cleanup clears the __dict__ of every module in sys.modules,
                    // which would set Team/Direction/etc. to None on the shared
                    // module object, corrupting it for all other subinterpreters.
                    if let Ok(sys) = PyModule::import(py, "sys") {
                        if let Ok(modules) = sys.getattr("modules") {
                            let _ = modules.call_method1("pop", ("cambc",));
                        }
                    }
                    py_convert::remove_type_cache();
                });
                Py_EndInterpreter(tstate);
                PyThreadState_Swap(main_tstate);
            }
        }
    }

    /// Destroy all remaining subinterpreters. Must be called before process
    /// exit to avoid "PyInterpreterState_Delete: remaining subinterpreters".
    fn destroy_all_subinterpreters(&mut self) {
        let all: Vec<i32> = self.unit_runners.keys().copied().collect();
        self.end_subinterpreters(&all);
    }
}

pub fn run(args: Args) -> PyResult<()> {
    Python::with_gil(|py| {
        ensure_sys_path(py, &args.engine_root)?;
        register_rust_module(py)?;

        // --- Phase 1: Initialize type cache (before sandbox, while imports work) ---
        py_convert::init_type_cache(py)?;

        // Capture the main thread's CPU clock ID so the watchdog can measure it.
        init_main_thread_clock_id();

        // Pre-import modules that CPython may lazily load during exception
        // handling or traceback formatting. Without this, the audit hook's
        // filesystem lock can trigger "Sandbox violation" when PyO3 normalizes
        // a SystemExit from the watchdog — intermittent (~7% of TLE bot matches).
        py.run(c"import traceback, linecache, tokenize, reprlib, random", None, None)?;

        // --- Phase 3: Install audit hook sandbox ---
        // Blocks ctypes, subprocess, socket, os.fork, gc introspection, etc.
        // for all subsequent Python code. The watchdog's fire/clear functions
        // are Rust-backed (injected above) so they bypass ctypes entirely and
        // trigger zero audit events.
        // Returns a lock function to tighten file access after bot loading.
        let sandbox_lock = if args.sandboxed {
            Some(install_sandbox(py)?)
        } else {
            None
        };

        // --- Phase 4: Validate bot code (runs under sandbox) ---
        // Bot imports/file reads still work here (fs lock not yet engaged).
        // Reject forbidden except handlers before loading bot code.
        check_except_handlers(py, &args.player_a)?;
        check_except_handlers(py, &args.player_b)?;
        // Exit codes: 10 = bot A failed, 11 = bot B failed, 12 = both failed.
        // Do a trial load in the main interpreter to validate both bots.
        let a_result = load_player_class(py, &args.player_a);
        let b_result = load_player_class(py, &args.player_b);
        match (&a_result, &b_result) {
            (Err(e), Ok(_)) => {
                eprintln!("Bot A failed to load: {e}");
                std::process::exit(10);
            }
            (Ok(_), Err(e)) => {
                eprintln!("Bot B failed to load: {e}");
                std::process::exit(11);
            }
            (Err(ea), Err(eb)) => {
                eprintln!("Both bots failed to load: A={ea}, B={eb}");
                std::process::exit(12);
            }
            _ => {}
        }

        // --- Phase 5: Lock filesystem access ---
        // After bots are loaded, block file opens to bot directories so
        // bots can't read each other's source code during gameplay.
        if let Some(lock_fn) = sandbox_lock {
            lock_fn.call0(py)?;
        }

        let (env, cores) = map_loader::load_map(&args.map).map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to load map {}: {}",
                args.map, err
            ))
        })?;
        let game = Game::new(env, cores, args.seed, args.suppress_indicators);

        // Seed Python's random module so bot RNG is reproducible.
        let random_mod = py.import("random")?;
        random_mod.call_method1("seed", (args.seed,))?;

        let sys_mod = PyModule::import(py, "sys")?.unbind();
        let io_mod = PyModule::import(py, "io")?.unbind();
        let gc_mod = PyModule::import(py, "gc")?.unbind();

        // Freeze all current objects into the permanent GC generation so
        // gc.collect() during gameplay only scans game objects, not the
        // thousands of objects from CLI imports (click, rich, etc.).
        // Without this, the CLI is ~50% slower than calling run_game directly.
        gc_mod.bind(py).call_method0("freeze")?;

        // Capture the main interpreter's thread state so subinterpreter swaps can restore it.
        let main_tstate = unsafe { pyo3_ffi::PyThreadState_Get() };

        let mut runner = GameRunner {
            game: Rc::new(RefCell::new(game)),
            bot_paths: [args.player_a.clone(), args.player_b.clone()],
            engine_root: args.engine_root.to_path_buf(),
            main_tstate,
            unit_runners: HashMap::new(),
            turn_timeout_ms: args.turn_timeout_ms,
            bot_extra_time_ns: HashMap::new(),
            sys_mod,
            io_mod,
            gc_mod,
        };
        runner.run(py)?;
        runner.destroy_all_subinterpreters();
        let mut game = runner.game.borrow_mut();
        let winner = game.winner_team();
        game.replay_recorder
            .write_to_path(&args.replay, winner)
            .map_err(|err| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "failed to write replay {}: {}",
                    args.replay, err
                ))
            })?;

        Ok(())
    })
}

/// Install the Python audit hook sandbox. Blocks dangerous operations
/// (ctypes memory manipulation, process spawning, network, gc introspection,
/// etc.) and removes ctypes modules from sys.modules.
///
/// Returns a lock function that, when called, blocks ALL Python-level file
/// opens (preventing /proc/self/mem reads, weight file loads, lazy imports,
/// etc.). Must be called AFTER bot loading so import-time reads still work.
fn install_sandbox(py: Python) -> PyResult<Py<PyAny>> {
    let locals = PyDict::new(py);
    py.run(
        c"
import sys

def _make_sandbox():
    blocked = frozenset({
        # ctypes memory manipulation
        'ctypes.addressof', 'ctypes.call_function', 'ctypes.cdata',
        'ctypes.cdata/buffer', 'ctypes.create_string_buffer',
        'ctypes.create_unicode_buffer', 'ctypes.dlopen', 'ctypes.dlsym',
        'ctypes.dlsym/handle', 'ctypes.memoryview_at', 'ctypes.string_at',
        'ctypes.wstring_at', 'ctypes.PyObj_FromPtr', 'ctypes._CData',
        # Process creation
        'subprocess.Popen', 'os.system', 'os.exec', 'os.fork', 'os.forkpty',
        'os.spawn', 'os.posix_spawn', 'os.kill', 'os.killpg',
        '_posixsubprocess.fork_exec',
        # Network
        'socket.__new__', 'socket.bind', 'socket.connect',
        'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr',
        'socket.sendmsg', 'socket.sendto',
        # GC introspection (could be used to find and tamper with sandbox state)
        'gc.get_objects', 'gc.get_referrers', 'gc.get_referents',
        # Threading (would bypass per-thread CPU time metering)
        '_thread.start_new_thread',
    })

    # fs_locked is a mutable list inside the closure — inaccessible to bot
    # code because the hook function is registered via sys.addaudithook()
    # (no public API to list hooks) and gc introspection is blocked above.
    fs_locked = [False]

    def hook(event, args):
        if event in blocked:
            raise RuntimeError('Sandbox violation: ' + event)
        if fs_locked[0] and event == 'open':
            raise RuntimeError('Sandbox violation: file access blocked during gameplay')

    def lock():
        fs_locked[0] = True

    return hook, lock

_hook, _lock_sandbox = _make_sandbox()
sys.addaudithook(_hook)

# Remove ctypes modules so bots can't re-import them.
for _mod_name in list(sys.modules):
    if 'ctypes' in _mod_name:
        del sys.modules[_mod_name]

# Remove _watchdog from sys.modules so bots can't access the watchdog.
sys.modules.pop('_watchdog', None)
",
        None,
        Some(&locals),
    )?;
    let lock_fn = locals
        .get_item("_lock_sandbox")?
        .expect("_lock_sandbox not found")
        .unbind();
    Ok(lock_fn)
}

/// Validate except handlers in all bot .py files.
///
/// Only allows handlers of the form `except Name:` or `except (Name, ...):`
/// where every name is in the whitelist of known-safe builtin exceptions plus
/// GameError. This guarantees the type expression is always a valid exception
/// type, preventing the TypeError side-channel that could be used to evade TLE.
///
/// Excluded from the whitelist: BaseException, SystemExit, KeyboardInterrupt
/// (used for TLE/engine control)
fn check_except_handlers(py: Python, main_py_path: &str) -> PyResult<()> {
    let locals = PyDict::new(py);
    locals.set_item("main_py_path", main_py_path)?;
    py.run(
        c"
import ast, os

ALLOWED = {
    # All builtin exceptions minus BaseException, SystemExit, KeyboardInterrupt
    'ArithmeticError', 'AssertionError', 'AttributeError',
    'BaseExceptionGroup', 'BlockingIOError', 'BrokenPipeError', 'BufferError',
    'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError',
    'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError',
    'DeprecationWarning', 'EOFError', 'EncodingWarning', 'EnvironmentError',
    'Exception', 'ExceptionGroup', 'FileExistsError', 'FileNotFoundError',
    'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError',
    'ImportError', 'ImportWarning', 'IndentationError', 'IndexError',
    'InterruptedError', 'IsADirectoryError', 'KeyError', 'LookupError',
    'MemoryError', 'ModuleNotFoundError', 'NameError', 'NotADirectoryError',
    'NotImplementedError', 'OSError', 'OverflowError',
    'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError',
    'RecursionError', 'ReferenceError', 'ResourceWarning', 'RuntimeError',
    'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError',
    'SyntaxWarning', 'SystemError', 'TabError', 'TimeoutError', 'TypeError',
    'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError',
    'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning',
    'ValueError', 'Warning', 'ZeroDivisionError',
    # Game-specific
    'GameError',
}

bot_dir = os.path.dirname(os.path.abspath(main_py_path))

for root, dirs, files in os.walk(bot_dir):
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        with open(fpath, encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=fpath)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            lineno = node.lineno
            ty = node.type
            if ty is None:
                raise ValueError(f'{fpath}:{lineno}: bare `except:` is not allowed; use a specific exception type')
            # Only Name or Tuple-of-Names are permitted — anything else
            # (Attribute, Call, Constant, List, etc.) is rejected.
            if isinstance(ty, ast.Name):
                names = [ty.id]
            elif isinstance(ty, ast.Tuple):
                if not all(isinstance(e, ast.Name) for e in ty.elts):
                    raise ValueError(f'{fpath}:{lineno}: except handler types must be plain names')
                names = [e.id for e in ty.elts]
            else:
                raise ValueError(f'{fpath}:{lineno}: except handler types must be plain names')
            for name in names:
                if name not in ALLOWED:
                    raise ValueError(f'{fpath}:{lineno}: `{name}` is not an allowed exception type')
",
        None,
        Some(&locals),
    )?;
    Ok(())
}

fn load_player_class(py: Python, path: &str) -> PyResult<PyObject> {
    // Add the bot's directory to sys.path so sibling imports work
    // (e.g. `import core` from main.py finds core.py in the same dir).
    let bot_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let sys = PyModule::import(py, "sys")?;
    let sys_path = sys.getattr("path")?;
    let bot_dir_str = bot_dir.to_str().unwrap();
    if !sys_path.contains(bot_dir_str)? {
        sys_path.call_method1("insert", (0, bot_dir_str))?;
    }

    let importlib = PyModule::import(py, "importlib.util")?;
    let spec = importlib
        .call_method1("spec_from_file_location", ("player_mod", path))?
        .unbind();
    let module = importlib
        .call_method1("module_from_spec", (spec.clone_ref(py),))?
        .unbind();
    spec.getattr(py, "loader")?
        .call_method1(py, "exec_module", (module.clone_ref(py),))?;
    let player_cls = module.getattr(py, "Player")?;
    Ok(player_cls)
}

fn ensure_sys_path(py: Python, engine_root: &Path) -> PyResult<()> {
    let sys = PyModule::import(py, "sys")?;
    let path = sys.getattr("path")?;
    let py_dir = engine_root.join("py");
    path.call_method1("insert", (0, py_dir.to_str().unwrap()))?;
    path.call_method1("insert", (1, engine_root.to_str().unwrap()))?;
    Ok(())
}

fn register_rust_module(py: Python) -> PyResult<()> {
    let rust_mod = pyo3::wrap_pymodule!(rustlib::controller::controller_mod)(py);
    let controller = rust_mod.bind(py).getattr("Controller")?;
    let cambc = PyModule::import(py, "cambc")?;
    cambc.setattr("Controller", controller)?;
    Ok(())
}
