use std::env;
use std::path::PathBuf;

pub struct Args {
    pub player_a: String,
    pub player_b: String,
    pub replay: String,
    pub map: String,
    pub turn_timeout_ms: u64,
    pub seed: u64,
    pub sandboxed: bool,
    pub suppress_indicators: bool,
    pub engine_root: PathBuf,
}

pub fn parse_args() -> Result<Args, String> {
    let mut player_a: Option<String> = None;
    let mut player_b: Option<String> = None;
    let mut replay: String = "replay.replay26".to_string();
    let mut map: Option<String> = None;
    let mut turn_timeout_ms: u64 = 2;
    let mut seed: u64 = 1;
    let mut sandboxed: bool = false;
    let mut suppress_indicators: bool = false;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--player-a" => {
                player_a = iter.next();
                if player_a.is_none() {
                    return Err("--player-a requires a path".to_string());
                }
            }
            "--player-b" => {
                player_b = iter.next();
                if player_b.is_none() {
                    return Err("--player-b requires a path".to_string());
                }
            }
            "--replay" => {
                replay = iter
                    .next()
                    .ok_or_else(|| "--replay requires a path".to_string())?;
            }
            "--map" => {
                map = Some(
                    iter.next()
                        .ok_or_else(|| "--map requires a path".to_string())?,
                );
            }
            "--turn-timeout-ms" => {
                turn_timeout_ms = iter
                    .next()
                    .ok_or_else(|| "--turn-timeout-ms requires a value".to_string())?
                    .parse()
                    .map_err(|_| "--turn-timeout-ms must be a positive integer".to_string())?;
            }
            "--seed" => {
                seed = iter
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_string())?
                    .parse()
                    .map_err(|_| "--seed must be a non-negative integer".to_string())?;
            }
            "--sandboxed" => {
                sandboxed = true;
            }
            "--suppress-indicators" => {
                suppress_indicators = true;
            }
            _ => {
                return Err(format!("Unknown argument: {arg}"));
            }
        }
    }

    let player_a = player_a.ok_or_else(|| "Missing --player-a".to_string())?;
    let player_b = player_b.ok_or_else(|| "Missing --player-b".to_string())?;
    let map = map.ok_or_else(|| "Missing --map".to_string())?;

    let engine_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine root")
        .to_path_buf();

    Ok(Args {
        player_a,
        player_b,
        replay,
        map,
        turn_timeout_ms,
        seed,
        sandboxed,
        suppress_indicators,
        engine_root,
    })
}
