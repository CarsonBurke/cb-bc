#[cfg(all(feature = "tle", not(target_os = "linux")))]
compile_error!("the 'tle' feature requires Linux (depends on libc clock_gettime)");

use pyo3::prelude::*;

fn main() -> PyResult<()> {
    let args = match battlecode_titan::cli::parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{err}\nUsage: titan_runner --player-a PATH --player-b PATH [--map PATH] [--replay PATH]");
            std::process::exit(2);
        }
    };
    battlecode_titan::runner::run(args)
}
