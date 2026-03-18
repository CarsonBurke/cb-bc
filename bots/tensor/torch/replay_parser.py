"""Parse .replay26 protobuf files to extract per-turn game state.

Key data extracted:
- Per-harvester cooldown each turn (cooldown > 0 = actively harvesting)
- Resource distribution events (which tiles had resource flow)
- Entity placements, removals, HP changes
- Player resource totals per turn
"""
import json
from pathlib import Path

# --- Protobuf schema (extracted from visualiser JS) ---

PROTO_JSON = {
    "nested": {
        "Replay": {"fields": {
            "map": {"type": "Map", "id": 1},
            "turns": {"rule": "repeated", "type": "Turn", "id": 3},
            "winner": {"type": "Team", "id": 4, "options": {"proto3_optional": True}},
        }},
        "Map": {"fields": {
            "width": {"type": "int32", "id": 1},
            "height": {"type": "int32", "id": 2},
            "rows": {"rule": "repeated", "type": "TileRow", "id": 3},
            "cores": {"rule": "repeated", "type": "CorePosition", "id": 4},
        }},
        "TileRow": {"fields": {
            "tiles": {"rule": "repeated", "type": "Environment", "id": 1},
        }},
        "Players": {"fields": {
            "a": {"type": "Player", "id": 1},
            "b": {"type": "Player", "id": 2},
        }},
        "Player": {"fields": {
            "titanium": {"type": "int32", "id": 1},
            "axionite": {"type": "int32", "id": 2},
            "resourcesCollected": {"type": "int32", "id": 3},
            "titaniumCollected": {"type": "int32", "id": 4},
            "axioniteCollected": {"type": "int32", "id": 5},
        }},
        "Turn": {"fields": {
            "updates": {"rule": "repeated", "type": "Update", "id": 1},
        }},
        "Update": {
            "oneofs": {"kind": {"oneof": [
                "placeEntity", "moveBuilderBot", "removeEntity",
                "distributeResources", "updateHp", "updatePlayers",
                "setActionCooldown", "setMoveCooldown", "botOutput",
                "indicatorLine", "indicatorDot", "fireTurret",
            ]}},
            "fields": {
                "placeEntity": {"type": "PlaceEntity", "id": 1},
                "moveBuilderBot": {"type": "MoveBuilderBot", "id": 2},
                "removeEntity": {"type": "RemoveEntity", "id": 3},
                "distributeResources": {"type": "DistributeResources", "id": 4},
                "updateHp": {"type": "UpdateHp", "id": 5},
                "updatePlayers": {"type": "UpdatePlayers", "id": 6},
                "setActionCooldown": {"type": "SetActionCooldown", "id": 7},
                "setMoveCooldown": {"type": "SetMoveCooldown", "id": 8},
                "botOutput": {"type": "BotOutput", "id": 9},
                "indicatorLine": {"type": "IndicatorLine", "id": 10},
                "indicatorDot": {"type": "IndicatorDot", "id": 11},
                "fireTurret": {"type": "FireTurret", "id": 12},
            },
        },
        "PlaceEntity": {"fields": {"entity": {"type": "Entity", "id": 1}}},
        "MoveBuilderBot": {"fields": {
            "id": {"type": "int32", "id": 1},
            "to": {"type": "Pos", "id": 2},
        }},
        "RemoveEntity": {"fields": {"id": {"type": "int32", "id": 1}}},
        "DistributeResources": {"fields": {
            "moves": {"rule": "repeated", "type": "ResourceMove", "id": 1},
        }},
        "ResourceMove": {"fields": {
            "from": {"type": "Pos", "id": 1},
            "to": {"type": "Pos", "id": 2},
        }},
        "UpdateHp": {"fields": {
            "id": {"type": "int32", "id": 1},
            "delta": {"type": "int32", "id": 2},
        }},
        "UpdatePlayers": {"fields": {
            "players": {"type": "Players", "id": 1},
        }},
        "SetActionCooldown": {"fields": {
            "id": {"type": "int32", "id": 1},
            "value": {"type": "int32", "id": 2},
        }},
        "SetMoveCooldown": {"fields": {
            "id": {"type": "int32", "id": 1},
            "value": {"type": "int32", "id": 2},
        }},
        "BotOutput": {"fields": {
            "id": {"type": "int32", "id": 1},
            "stdout": {"type": "string", "id": 2},
            "execTimeUs": {"type": "uint32", "id": 3},
            "tled": {"type": "bool", "id": 4},
        }},
        "IndicatorLine": {"fields": {
            "id": {"type": "int32", "id": 1},
            "posA": {"type": "Pos", "id": 2},
            "posB": {"type": "Pos", "id": 3},
            "r": {"type": "int32", "id": 4},
            "g": {"type": "int32", "id": 5},
            "b": {"type": "int32", "id": 6},
        }},
        "IndicatorDot": {"fields": {
            "id": {"type": "int32", "id": 1},
            "pos": {"type": "Pos", "id": 2},
            "r": {"type": "int32", "id": 3},
            "g": {"type": "int32", "id": 4},
            "b": {"type": "int32", "id": 5},
        }},
        "FireTurret": {"fields": {
            "from": {"type": "Pos", "id": 1},
            "to": {"type": "Pos", "id": 2},
        }},
        "Pos": {"fields": {
            "x": {"type": "int32", "id": 1},
            "y": {"type": "int32", "id": 2},
        }},
        "CorePosition": {"fields": {
            "id": {"type": "int32", "id": 1},
            "team": {"type": "Team", "id": 2},
            "position": {"type": "Pos", "id": 3},
        }},
        "Entity": {
            "oneofs": {"kind": {"oneof": [
                "builderBot", "conveyor", "splitter", "armouredConveyor",
                "bridge", "harvester", "foundry", "road", "barrier",
                "marker", "core", "gunner", "sentinel", "breach", "launcher",
            ]}},
            "fields": {
                "id": {"type": "int32", "id": 1},
                "team": {"type": "Team", "id": 2},
                "position": {"type": "Pos", "id": 3},
                "hp": {"type": "int32", "id": 4},
                "maxHp": {"type": "int32", "id": 5},
                "builderBot": {"type": "BuilderBot", "id": 10},
                "conveyor": {"type": "Conveyor", "id": 11},
                "splitter": {"type": "Splitter", "id": 12},
                "armouredConveyor": {"type": "ArmouredConveyor", "id": 13},
                "bridge": {"type": "Bridge", "id": 14},
                "harvester": {"type": "Harvester", "id": 15},
                "foundry": {"type": "Foundry", "id": 16},
                "road": {"type": "Road", "id": 17},
                "barrier": {"type": "Barrier", "id": 18},
                "marker": {"type": "Marker", "id": 19},
                "core": {"type": "Core", "id": 20},
                "gunner": {"type": "Gunner", "id": 21},
                "sentinel": {"type": "Sentinel", "id": 22},
                "breach": {"type": "Breach", "id": 23},
                "launcher": {"type": "Launcher", "id": 24},
            },
        },
        "BuilderBot": {"fields": {
            "actionCooldown": {"type": "int32", "id": 1},
            "moveCooldown": {"type": "int32", "id": 2},
        }},
        "Conveyor": {"fields": {
            "direction": {"type": "Direction", "id": 1},
            "stored": {"type": "ResourceType", "id": 2},
        }},
        "Splitter": {"fields": {
            "direction": {"type": "Direction", "id": 1},
            "stored": {"type": "ResourceType", "id": 2},
        }},
        "ArmouredConveyor": {"fields": {
            "direction": {"type": "Direction", "id": 1},
            "stored": {"type": "ResourceType", "id": 2},
        }},
        "Bridge": {"fields": {
            "target": {"type": "Pos", "id": 1},
            "stored": {"type": "ResourceType", "id": 2},
        }},
        "Harvester": {"fields": {
            "cooldown": {"type": "int32", "id": 1},
            "resourceType": {"type": "ResourceType", "id": 2},
        }},
        "Foundry": {"fields": {
            "stored": {"type": "ResourceType", "id": 2},
        }},
        "Road": {"fields": {}},
        "Barrier": {"fields": {}},
        "Marker": {"fields": {"value": {"type": "uint32", "id": 1}}},
        "Core": {"fields": {
            "actionCooldown": {"type": "int32", "id": 1},
        }},
        "Gunner": {"fields": {
            "direction": {"type": "Direction", "id": 1},
            "ammoType": {"type": "ResourceType", "id": 2},
            "ammoAmount": {"type": "int32", "id": 3},
        }},
        "Sentinel": {"fields": {
            "direction": {"type": "Direction", "id": 1},
            "ammoType": {"type": "ResourceType", "id": 2},
            "ammoAmount": {"type": "int32", "id": 3},
        }},
        "Breach": {"fields": {
            "direction": {"type": "Direction", "id": 1},
            "ammoType": {"type": "ResourceType", "id": 2},
            "ammoAmount": {"type": "int32", "id": 3},
        }},
        "Launcher": {"fields": {
            "direction": {"type": "Direction", "id": 1},
            "ammoType": {"type": "ResourceType", "id": 2},
            "ammoAmount": {"type": "int32", "id": 3},
        }},
        "Team": {"values": {"TEAM_A": 0, "TEAM_B": 1}},
        "Direction": {"values": {
            "DIR_CENTRE": 0, "DIR_NORTH": 1, "DIR_NORTHEAST": 2,
            "DIR_EAST": 3, "DIR_SOUTHEAST": 4, "DIR_SOUTH": 5,
            "DIR_SOUTHWEST": 6, "DIR_WEST": 7, "DIR_NORTHWEST": 8,
        }},
        "ResourceType": {"values": {
            "RESOURCE_NONE": 0, "RESOURCE_TITANIUM": 1,
            "RESOURCE_RAW_AXIONITE": 2, "RESOURCE_REFINED_AXIONITE": 3,
        }},
        "Environment": {"values": {
            "ENV_EMPTY": 0, "ENV_WALL": 1,
            "ENV_ORE_TITANIUM": 2, "ENV_ORE_AXIONITE": 3,
        }},
    }
}

# --- Pure-Python protobuf decoder (no dependencies) ---

# Wire types
VARINT = 0
FIXED64 = 1
LENGTH_DELIMITED = 2
FIXED32 = 5


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a varint, return (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _decode_zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _decode_message(data: bytes, start: int, end: int) -> dict[int, list]:
    """Decode raw protobuf fields into {field_id: [values]}."""
    fields: dict[int, list] = {}
    pos = start
    while pos < end:
        tag, pos = _decode_varint(data, pos)
        field_id = tag >> 3
        wire_type = tag & 0x07
        if wire_type == VARINT:
            value, pos = _decode_varint(data, pos)
        elif wire_type == FIXED64:
            value = data[pos:pos + 8]
            pos += 8
        elif wire_type == LENGTH_DELIMITED:
            length, pos = _decode_varint(data, pos)
            value = data[pos:pos + length]
            pos += length
        elif wire_type == FIXED32:
            value = data[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unknown wire type {wire_type} at pos {pos}")
        fields.setdefault(field_id, []).append((wire_type, value))
    return fields


# Type mapping for schema interpretation
_SCALAR_TYPES = {
    "int32", "int64", "uint32", "uint64", "sint32", "sint64",
    "bool", "fixed32", "fixed64", "sfixed32", "sfixed64",
    "float", "double", "string", "bytes",
}


def _is_enum(type_name: str, schema: dict) -> bool:
    return type_name in schema and "values" in schema[type_name]


def _is_message(type_name: str, schema: dict) -> bool:
    return type_name in schema and "fields" in schema[type_name]


def _decode_typed(field_type: str, wire_type: int, value, schema: dict):
    """Decode a single field value according to its schema type."""
    if field_type in ("int32", "int64", "uint32", "uint64"):
        if wire_type == VARINT:
            return value  # already decoded
        return int.from_bytes(value, "little")
    elif field_type in ("sint32", "sint64"):
        return _decode_zigzag(value)
    elif field_type == "bool":
        return bool(value)
    elif field_type == "string":
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    elif field_type == "bytes":
        return value
    elif field_type == "float":
        import struct
        return struct.unpack("<f", value)[0]
    elif field_type == "double":
        import struct
        return struct.unpack("<d", value)[0]
    elif _is_enum(field_type, schema):
        return value if wire_type == VARINT else int.from_bytes(value, "little")
    elif _is_message(field_type, schema):
        if isinstance(value, bytes):
            return decode_proto(field_type, value, schema)
        return value
    return value


def decode_proto(msg_type: str, data: bytes, schema: dict) -> dict:
    """Decode a protobuf message using the schema."""
    msg_schema = schema[msg_type]
    field_defs = msg_schema.get("fields", {})
    oneofs = msg_schema.get("oneofs", {})

    # Build id -> field_name, field_type mapping
    id_to_field = {}
    for fname, fdef in field_defs.items():
        id_to_field[fdef["id"]] = (fname, fdef.get("type", "bytes"), fdef.get("rule", ""))

    raw = _decode_message(data, 0, len(data))
    result = {}

    for fid, entries in raw.items():
        if fid not in id_to_field:
            continue
        fname, ftype, frule = id_to_field[fid]
        if frule == "repeated":
            result[fname] = [_decode_typed(ftype, wt, val, schema) for wt, val in entries]
        else:
            wt, val = entries[-1]  # last wins for non-repeated
            result[fname] = _decode_typed(ftype, wt, val, schema)

    return result


# --- High-level replay parsing ---

SCHEMA = PROTO_JSON["nested"]


def parse_replay(path: str | Path) -> dict:
    """Parse a .replay26 file and return the decoded Replay message."""
    data = Path(path).read_bytes()
    return decode_proto("Replay", data, SCHEMA)


def extract_turn_states(replay: dict) -> list[dict]:
    """Process replay turns into per-turn game state snapshots.

    Returns list of dicts, one per turn, with:
    - entities: {id: entity_dict} — current entity states
    - working_harvesters: {team: count} — harvesters with cooldown > 0
    - resource_moves: [(from_pos, to_pos)] — resource flow events
    - player_resources: {team: {ti, ax, collected}} — player totals
    - core_hp: {team: hp}
    """
    entities = {}  # id -> entity state
    turn_states = []

    # Initialize cores from map data
    if "map" in replay:
        for core in replay["map"].get("cores", []):
            eid = core["id"]
            team = core.get("team", 0)
            pos = core.get("position", {})
            entities[eid] = {
                "id": eid, "kind": "core", "team": team,
                "position": pos, "hp": 500, "maxHp": 500,
                "actionCooldown": 0,
            }

    for turn_idx, turn in enumerate(replay.get("turns", [])):
        resource_moves = []
        player_resources = {}

        for update in turn.get("updates", []):
            if "placeEntity" in update:
                ent = update["placeEntity"].get("entity", {})
                eid = ent.get("id", 0)
                kind = None
                extra = {}
                for k in ("builderBot", "conveyor", "splitter", "armouredConveyor",
                          "bridge", "harvester", "foundry", "road", "barrier",
                          "marker", "core", "gunner", "sentinel", "breach", "launcher"):
                    if k in ent:
                        kind = k
                        extra = ent[k] if isinstance(ent[k], dict) else {}
                        break
                entities[eid] = {
                    "id": eid,
                    "kind": kind,
                    "team": ent.get("team", 0),
                    "position": ent.get("position", {}),
                    "hp": ent.get("hp", 0),
                    "maxHp": ent.get("maxHp", 0),
                    **extra,
                }

            elif "moveBuilderBot" in update:
                mv = update["moveBuilderBot"]
                eid = mv.get("id", 0)
                if eid in entities:
                    entities[eid]["position"] = mv.get("to", {})

            elif "removeEntity" in update:
                eid = update["removeEntity"].get("id", 0)
                entities.pop(eid, None)

            elif "updateHp" in update:
                hp_update = update["updateHp"]
                eid = hp_update.get("id", 0)
                if eid in entities:
                    entities[eid]["hp"] = entities[eid].get("hp", 0) + hp_update.get("delta", 0)

            elif "setActionCooldown" in update:
                cd = update["setActionCooldown"]
                eid = cd.get("id", 0)
                if eid in entities:
                    entities[eid]["actionCooldown"] = cd.get("value", 0)

            elif "setMoveCooldown" in update:
                cd = update["setMoveCooldown"]
                eid = cd.get("id", 0)
                if eid in entities:
                    entities[eid]["moveCooldown"] = cd.get("value", 0)

            elif "distributeResources" in update:
                dist = update["distributeResources"]
                for mv in dist.get("moves", []):
                    resource_moves.append((mv.get("from", {}), mv.get("to", {})))

            elif "updatePlayers" in update:
                players = update["updatePlayers"].get("players", {})
                for team_key in ("a", "b"):
                    if team_key in players:
                        p = players[team_key]
                        player_resources[team_key] = {
                            "titanium": p.get("titanium", 0),
                            "axionite": p.get("axionite", 0),
                            "resourcesCollected": p.get("resourcesCollected", 0),
                            "titaniumCollected": p.get("titaniumCollected", 0),
                            "axioniteCollected": p.get("axioniteCollected", 0),
                        }

        # Count working harvesters per team
        working = {0: 0, 1: 0}
        for ent in entities.values():
            if ent.get("kind") == "harvester":
                cd = ent.get("cooldown", ent.get("actionCooldown", 0))
                if cd and cd > 0:
                    working[ent.get("team", 0)] += 1

        # Core HP per team
        core_hp = {}
        for ent in entities.values():
            if ent.get("kind") == "core":
                team_key = "a" if ent.get("team", 0) == 0 else "b"
                core_hp[team_key] = ent.get("hp", 0)

        turn_states.append({
            "turn": turn_idx,
            "working_harvesters": working,
            "resource_moves": resource_moves,
            "player_resources": player_resources,
            "core_hp": core_hp,
        })

    return turn_states


def extract_working_harvesters_from_replay(replay: dict) -> dict[int, dict[int, int]]:
    """Extract per-turn working harvester counts per team from a parsed replay.

    Uses the same logic as the visualiser:
    - When distributeResources moves FROM a harvester position, set its cooldown to 3
    - Each turn, decrement all harvester cooldowns
    - A harvester with cooldown > 0 is "working"

    Returns {turn_number: {team_0_count, team_1_count}}.
    """
    entities = {}  # id -> {kind, team, position, cooldown}
    harv_positions = {}  # (x, y) -> entity

    # Init cores from map
    if "map" in replay:
        for core in replay["map"].get("cores", []):
            entities[core["id"]] = {
                "kind": "core", "team": core.get("team", 0),
                "position": core.get("position", {}),
            }

    result = {}
    for turn_idx, turn in enumerate(replay.get("turns", [])):
        # Decrement harvester cooldowns (start of turn, like visualiser)
        for ent in entities.values():
            if ent.get("kind") == "harvester":
                cd = ent.get("cooldown", 0)
                if cd > 0:
                    ent["cooldown"] = cd - 1

        # Process updates
        for update in turn.get("updates", []):
            if "placeEntity" in update:
                ent = update["placeEntity"].get("entity", {})
                eid = ent.get("id", 0)
                kind = None
                for k in ("builderBot", "conveyor", "splitter", "armouredConveyor",
                          "bridge", "harvester", "foundry", "road", "barrier",
                          "marker", "core", "gunner", "sentinel", "breach", "launcher"):
                    if k in ent:
                        kind = k
                        break
                entity = {
                    "kind": kind,
                    "team": ent.get("team", 0),
                    "position": ent.get("position", {}),
                    "cooldown": 0,
                }
                entities[eid] = entity
                if kind == "harvester":
                    pos = entity["position"]
                    harv_positions[(pos.get("x", -1), pos.get("y", -1))] = entity

            elif "removeEntity" in update:
                eid = update["removeEntity"].get("id", 0)
                ent = entities.pop(eid, None)
                if ent and ent.get("kind") == "harvester":
                    pos = ent.get("position", {})
                    harv_positions.pop((pos.get("x", -1), pos.get("y", -1)), None)

            elif "moveBuilderBot" in update:
                mv = update["moveBuilderBot"]
                eid = mv.get("id", 0)
                if eid in entities:
                    ent = entities[eid]
                    if ent.get("kind") == "harvester":
                        old_pos = ent.get("position", {})
                        harv_positions.pop((old_pos.get("x", -1), old_pos.get("y", -1)), None)
                        ent["position"] = mv.get("to", {})
                        new_pos = ent["position"]
                        harv_positions[(new_pos.get("x", -1), new_pos.get("y", -1))] = ent
                    else:
                        ent["position"] = mv.get("to", {})

            elif "distributeResources" in update:
                for mv in update["distributeResources"].get("moves", []):
                    from_pos = mv.get("from", {})
                    key = (from_pos.get("x", -1), from_pos.get("y", -1))
                    ent = harv_positions.get(key)
                    if ent is not None:
                        ent["cooldown"] = 3

        # Count working harvesters per team
        counts = {0: 0, 1: 0}
        for ent in harv_positions.values():
            if ent.get("cooldown", 0) > 0:
                counts[ent.get("team", 0)] += 1
        result[turn_idx] = counts

    return result


def extract_working_harvesters(replay_path: str | Path) -> dict[int, dict[int, int]]:
    """Extract per-turn working harvester counts per team from a replay file."""
    replay = parse_replay(replay_path)
    return extract_working_harvesters_from_replay(replay)


# Obs vector layout indices
_GLOBAL_START = 25 * 17  # GRID_SIZE^2 * TILE_FEATURES = 425
_ROUND_OBS_IDX = _GLOBAL_START  # first global feature = rnd / 2000.0


def augment_trajectories_with_working_counts(
    trajectories: list[list],
    working: dict[int, dict[int, int]],
    team: int = 0,
) -> list[list]:
    """Override n_working in trajectory steps with precomputed replay ground truth.

    Each trajectory step is a tuple with n_working at index 6.
    Round number is extracted from obs[425] (= rnd / 2000.0).
    """
    if not working:
        return trajectories  # fallback: leave unchanged

    for traj in trajectories:
        for i, step in enumerate(traj):
            if not isinstance(step, (list, tuple)) or len(step) < 7:
                continue
            # Convert to list if tuple
            if isinstance(step, tuple):
                step = list(step)
                traj[i] = step
            # Extract round from obs
            obs = step[0]
            rnd = int(round(obs[_ROUND_OBS_IDX] * 2000.0))
            rnd = max(0, min(rnd, 1999))
            # Get working harvester count for this team at this round
            counts = working.get(rnd, {})
            n_working = counts.get(team, 0)
            step[6] = n_working

    return trajectories


def augment_trajectories_with_replay(
    trajectories: list[list],
    replay_path: str | Path,
    team: int = 0,
) -> list[list]:
    """Override n_working in trajectory steps with replay ground truth."""
    try:
        working = extract_working_harvesters(replay_path)
    except Exception:
        return trajectories
    return augment_trajectories_with_working_counts(trajectories, working, team)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python replay_parser.py <replay_file>")
        sys.exit(1)

    replay = parse_replay(sys.argv[1])
    print(f"Map: {replay.get('map', {}).get('width', '?')}x{replay.get('map', {}).get('height', '?')}")
    print(f"Turns: {len(replay.get('turns', []))}")
    print(f"Winner: {'A' if replay.get('winner', 0) == 0 else 'B'}")

    states = extract_turn_states(replay)
    for s in states[:5]:
        print(f"  Turn {s['turn']}: working_harv={s['working_harvesters']} "
              f"moves={len(s['resource_moves'])} core_hp={s['core_hp']}")
    if len(states) > 10:
        print("  ...")
        for s in states[-3:]:
            print(f"  Turn {s['turn']}: working_harv={s['working_harvesters']} "
                  f"moves={len(s['resource_moves'])} core_hp={s['core_hp']}")
