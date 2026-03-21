use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use rand::Rng;

use super::*;
use crate::common::ResourceType;
use crate::game_map::{ArmouredConveyor, Bridge, Conveyor, Foundry, Splitter};

impl Game {
    pub fn distribute_resources(&mut self) {
        let mut moves = Vec::new();
        #[derive(Clone, Debug)]
        struct Edge {
            priority: f64,
            source: Pos,
            sink: Pos,
        }
        impl Eq for Edge {}
        impl PartialEq for Edge {
            fn eq(&self, other: &Self) -> bool {
                self.priority == other.priority
            }
        }
        impl Ord for Edge {
            fn cmp(&self, other: &Self) -> Ordering {
                self.priority
                    .partial_cmp(&other.priority)
                    .unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for Edge {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        // incoming[sink] = list of sources that have a valid output edge pointing to sink
        let mut incoming: BTreeMap<Pos, Vec<Pos>> = BTreeMap::new();
        // outgoing_count[source] = number of valid output edges from source to buildings
        let mut outgoing_count: HashMap<Pos, usize> = HashMap::new();
        let mut processed: HashSet<Pos> = HashSet::new();

        for row in &self.game_map.tiles {
            for tile in row {
                if let Some(id) = tile.building {
                    let entity = self
                        .entities
                        .get(&id)
                        .unwrap_or_else(|| panic!("tile building id missing entity {}", id));
                    let no_output = matches!(
                        entity,
                        Entity::Conveyor(Conveyor { stored: None, .. })
                            | Entity::Splitter(Splitter { stored: None, .. })
                            | Entity::ArmouredConveyor(ArmouredConveyor { stored: None, .. })
                            | Entity::Bridge(Bridge { stored: None, .. })
                            | Entity::Foundry(Foundry {
                                stored: None
                                    | Some(ResourceType::Titanium)
                                    | Some(ResourceType::RawAxionite),
                                ..
                            })
                    );
                    if no_output {
                        processed.insert(tile.position);
                    }
                    let mut count = 0;
                    for sink_pos in entity.output_targets() {
                        if !self.game_map.in_bounds(sink_pos) {
                            continue;
                        }
                        let sink_tile = self.game_map.tile(sink_pos);
                        if sink_tile.building.is_some() {
                            count += 1;
                            incoming
                                .entry(sink_pos)
                                .or_default()
                                .push(tile.position);
                        }
                    }
                    outgoing_count.insert(tile.position, count);
                }
            }
        }

        // Compute edge priority: forced edges (source has 1 output, sink has 1 input) get i32::MAX.
        // Otherwise use global LRU: -(last turn used), so never-used edges get 0 and
        // more-recently-used edges get more negative values (lower priority in max-heap).
        let edge_priority = |source: Pos, sink: Pos| -> i32 {
            let src_out = outgoing_count.get(&source).copied().unwrap_or(0);
            let sink_in = incoming.get(&sink).map(|v| v.len()).unwrap_or(0);
            if src_out == 1 && sink_in == 1 {
                i32::MAX
            } else {
                -self.edge_last_used.get(&(source, sink)).copied().unwrap_or(0)
            }
        };

        let mut heap: BinaryHeap<Edge> = BinaryHeap::new();
        for (sink_pos, sources) in &incoming {
            let sink_id = self
                .game_map
                .tile(*sink_pos)
                .building
                .unwrap_or_else(|| panic!("incoming sink missing building at {:?}", sink_pos));
            for source_pos in sources {
                let source_id = self.game_map.tile(*source_pos).building.unwrap_or_else(|| {
                    panic!("incoming source missing building at {:?}", source_pos)
                });
                let source = self
                    .entities
                    .get(&source_id)
                    .unwrap_or_else(|| panic!("unknown building id {}", source_id));
                let resource = match source.resource_to_feed() {
                    Some(resource) => resource,
                    None => continue,
                };
                let sink = self
                    .entities
                    .get(&sink_id)
                    .unwrap_or_else(|| panic!("unknown building id {}", sink_id));
                let sink_can_accept = sink.can_accept_from(resource, *source_pos, matches!(source, Entity::Bridge(_)));
                if sink_can_accept {
                    let priority = edge_priority(*source_pos, *sink_pos, );
                    let jitter = self.rng.gen::<f64>();
                    heap.push(Edge {
                        priority: priority as f64 + jitter,
                        source: *source_pos,
                        sink: *sink_pos,
                    });
                }
            }
        }

        while let Some(edge) = heap.pop() {
            if processed.contains(&edge.source) {
                continue;
            }
            let source_id = self
                .game_map
                .tile(edge.source)
                .building
                .unwrap_or_else(|| panic!("edge source missing building at {:?}", edge.source));
            let sink_id = self
                .game_map
                .tile(edge.sink)
                .building
                .unwrap_or_else(|| panic!("edge sink missing building at {:?}", edge.sink));

            let resource = {
                let source = self
                    .entities
                    .get(&source_id)
                    .unwrap_or_else(|| panic!("unknown building id {}", source_id));
                source.resource_to_feed()
            };
            let resource = match resource {
                Some(r) => r,
                None => continue,
            };
            let sink_can_accept = {
                let source = self
                    .entities
                    .get(&source_id)
                    .unwrap_or_else(|| panic!("unknown building id {}", source_id));
                let sink = self
                    .entities
                    .get(&sink_id)
                    .unwrap_or_else(|| panic!("unknown building id {}", sink_id));
                sink.can_accept_from(resource, edge.source, matches!(source, Entity::Bridge(_)))
            };
            if !sink_can_accept {
                continue;
            }

            {
                let sink = self
                    .entities
                    .get_mut(&sink_id)
                    .unwrap_or_else(|| panic!("unknown building id {}", sink_id));
                sink.receive_resource(resource);
            }
            {
                let source = self
                    .entities
                    .get_mut(&source_id)
                    .unwrap_or_else(|| panic!("unknown building id {}", source_id));
                source.consume_feed();
            }
            moves.push((edge.source, edge.sink));
            processed.insert(edge.source);

            for upstream_pos in incoming.get(&edge.source).cloned().unwrap_or_default() {
                if processed.contains(&upstream_pos) {
                    continue;
                }
                let upstream_id = self
                    .game_map
                    .tile(upstream_pos)
                    .building
                    .unwrap_or_else(|| {
                        panic!("upstream source missing building at {:?}", upstream_pos)
                    });
                let upstream_resource = {
                    let upstream = self
                        .entities
                        .get(&upstream_id)
                        .unwrap_or_else(|| panic!("unknown building id {}", upstream_id));
                    upstream.resource_to_feed()
                };
                if let Some(up_resource) = upstream_resource {
                    let source_can_accept = {
                        let upstream = self
                            .entities
                            .get(&upstream_id)
                            .unwrap_or_else(|| panic!("unknown building id {}", upstream_id));
                        let source = self
                            .entities
                            .get(&source_id)
                            .unwrap_or_else(|| panic!("unknown building id {}", source_id));
                        source.can_accept_from(up_resource, upstream_pos, matches!(upstream, Entity::Bridge(_)))
                    };
                    if source_can_accept {
                        let priority = edge_priority(upstream_pos, edge.source, );
                        let jitter = self.rng.gen::<f64>();
                        heap.push(Edge {
                            priority: priority as f64 + jitter,
                            source: upstream_pos,
                            sink: edge.source,
                        });
                    }
                }
            }
        }

        for entity in self.entities.values_mut() {
            if let Entity::Core(core) = entity {
                let team_idx = core.team.index();
                for resource in core.received.drain(..) {
                    self.players[team_idx].add_resource(resource);
                }
            }
        }
        for &(source, sink) in &moves {
            self.edge_last_used.insert((source, sink), self.turn);
        }
        if !moves.is_empty() {
            self.replay_recorder
                .append(GameDiff::DistributeResources { moves });
        }
    }
}
