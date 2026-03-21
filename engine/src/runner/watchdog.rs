use pyo3::prelude::*;
use pyo3::types::PyAny;

pub struct Watchdog {
    pub arm: Py<PyAny>,
    pub disarm: Py<PyAny>,
    pub clear_async_exc: Py<PyAny>,
    pub shutdown: Py<PyAny>,
    pub thread: Py<PyAny>,
}

#[cfg(feature = "tle")]
mod tle_impl {
    use super::*;

    impl Watchdog {
        /// Create a new watchdog.
        ///
        /// Single long-lived watchdog thread that reuses a Condition variable.
        /// `arm(wall_timeout)` wakes it; `disarm()` cancels before it fires.
        /// No thread spawn/join per unit turn.
        ///
        /// The watchdog sleeps for the wall-clock timeout, then fires
        /// a SystemExit async exception on the main thread. CPU time
        /// enforcement is handled separately by `check_deadline` in the
        /// controller, which runs on the main thread at every API call.
        pub fn new(
            py: Python,
            cpu_time_fn: Py<PyAny>,
            fire_fn: Py<PyAny>,
            clear_fn: Py<PyAny>,
        ) -> Watchdog {
            let watchdog_mod = pyo3::types::PyModule::from_code(
                py,
                c"
import sys
import threading
import time

# Capture references before bot code can monkey-patch them.
_monotonic = time.monotonic
# _cpu_time_remaining_ns, _fire_system_exit, _clear_async_exc_for are
# injected by Rust after module creation — no ctypes needed.

tid = threading.get_ident()
cond = threading.Condition()

_wall_deadline = None
shutdown_flag = False

def watchdog_loop():
    global _wall_deadline, shutdown_flag
    with cond:
        while True:
            while _wall_deadline is None and not shutdown_flag:
                cond.wait()
            if shutdown_flag:
                return
            deadline = _wall_deadline
            remaining = deadline - _monotonic()
            if remaining > 0:
                cond.wait(timeout=remaining)
            # If deadline changed (disarmed or re-armed) while we were waiting, skip
            if _wall_deadline is not deadline:
                continue
            # Check whether the main thread has actually reached its CPU budget.
            # If budget remains, sleep for that duration + 50 us and re-check.
            cpu_resleeps = 0
            remaining_ns = _cpu_time_remaining_ns()
            while remaining_ns > 0:
                cpu_resleeps += 1
                cond.wait(timeout=remaining_ns / 1e9 + 50e-6)
                if _wall_deadline is not deadline:
                    break
                remaining_ns = _cpu_time_remaining_ns()
            if _wall_deadline is not deadline:
                continue
            if cpu_resleeps >= 1:
                print(f'[watchdog] fired after {cpu_resleeps} cpu re-sleep(s)', file=sys.stderr, flush=True)
            _wall_deadline = None
            cond.release()
            try:
                _fire_system_exit(tid)
            finally:
                cond.acquire()

thread = threading.Thread(target=watchdog_loop, daemon=True)
thread.start()

def arm(wall_timeout):
    global _wall_deadline
    with cond:
        _wall_deadline = _monotonic() + wall_timeout
        cond.notify()

def disarm():
    global _wall_deadline
    with cond:
        _wall_deadline = None
        cond.notify()

def shutdown():
    global shutdown_flag
    with cond:
        shutdown_flag = True
        cond.notify()

def clear_async_exc():
    while True:
        try:
            _clear_async_exc_for(tid)
            return
        except SystemExit:
            pass
",
                c"_watchdog",
                c"_watchdog",
            )
            .expect("watchdog module");

            // Inject Rust-backed functions — these bypass ctypes entirely,
            // so they never trigger ctypes.* audit events.
            watchdog_mod
                .setattr("_cpu_time_remaining_ns", cpu_time_fn)
                .expect("inject _cpu_time_remaining_ns");
            watchdog_mod
                .setattr("_fire_system_exit", fire_fn)
                .expect("inject _fire_system_exit");
            watchdog_mod
                .setattr("_clear_async_exc_for", clear_fn)
                .expect("inject _clear_async_exc_for");

            Watchdog {
                arm: watchdog_mod.getattr("arm").expect("arm attr").unbind(),
                disarm: watchdog_mod
                    .getattr("disarm")
                    .expect("disarm attr")
                    .unbind(),
                clear_async_exc: watchdog_mod
                    .getattr("clear_async_exc")
                    .expect("clear_async_exc attr")
                    .unbind(),
                shutdown: watchdog_mod
                    .getattr("shutdown")
                    .expect("shutdown attr")
                    .unbind(),
                thread: watchdog_mod
                    .getattr("thread")
                    .expect("thread attr")
                    .unbind(),
            }
        }
    }
}

#[cfg(not(feature = "tle"))]
impl Watchdog {
    /// No-op watchdog when TLE enforcement is disabled.
    pub fn new(py: Python, _cpu_time_fn: Py<PyAny>, _fire_fn: Py<PyAny>, _clear_fn: Py<PyAny>) -> Watchdog {
        let noop = py
            .eval(c"lambda *a, **kw: None", None, None)
            .expect("noop lambda")
            .unbind();
        let none = py.None();
        Watchdog {
            arm: noop.clone_ref(py),
            disarm: noop.clone_ref(py),
            clear_async_exc: noop.clone_ref(py),
            shutdown: noop,
            thread: none,
        }
    }
}
