use std::fs;
use std::io;
use std::path::Path;

use prost::Message;

use crate::common::{Environment, Pos, Team};
use crate::proto;

pub fn load_map(path: &str) -> io::Result<(Vec<Vec<Environment>>, Vec<(Pos, Team)>)> {
    let bytes = fs::read(path)?;
    let map = proto::Map::decode(&*bytes)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

    let environment = map
        .rows
        .iter()
        .map(|row| {
            row.tiles
                .iter()
                .map(|&tile| match proto::Environment::try_from(tile) {
                    Ok(proto::Environment::EnvEmpty) => Environment::Empty,
                    Ok(proto::Environment::EnvWall) => Environment::Wall,
                    Ok(proto::Environment::EnvOreTitanium) => Environment::OreTitanium,
                    Ok(proto::Environment::EnvOreAxionite) => Environment::OreAxionite,
                    Err(_) => panic!("unknown environment value: {}", tile),
                })
                .collect()
        })
        .collect();

    let cores: Vec<(Pos, Team)> = map
        .cores
        .iter()
        .map(|core| {
            let pos = core
                .position
                .as_ref()
                .map(|p| Pos { x: p.x, y: p.y })
                .unwrap_or_else(|| panic!("core missing position"));
            let team = match proto::Team::try_from(core.team) {
                Ok(proto::Team::A) => Team::A,
                Ok(proto::Team::B) => Team::B,
                _ => panic!("unknown team value: {}", core.team),
            };
            (pos, team)
        })
        .collect();

    let has_team_a = cores.iter().any(|(_, t)| *t == Team::A);
    let has_team_b = cores.iter().any(|(_, t)| *t == Team::B);
    if !has_team_a || !has_team_b {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "map must have a core for each team",
        ));
    }

    Ok((environment, cores))
}

pub fn save_map(
    path: &str,
    environment: &[Vec<Environment>],
    cores: &[(Pos, Team)],
) -> io::Result<()> {
    use crate::replay::proto_conversions::build_proto_map;

    let map = build_proto_map(environment, cores);
    let mut buf = Vec::new();
    map.encode(&mut buf)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, buf)
}
