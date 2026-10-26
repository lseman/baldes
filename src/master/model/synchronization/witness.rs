//! Joint earliest schedules and exact pinned-start certificates.
use super::*;

/// Find a joint schedule by monotone earliest-start propagation, then certify
/// every physical path with all common starts pinned simultaneously. Failure to
/// certify only triggers branching; it is never an infeasibility proof.
pub(in crate::master::model) fn witness<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
) -> Result<Option<Vec<FixedRoute>>> {
    if state.synchronized_services.is_empty() {
        return Ok(Some(routes.to_vec()));
    }
    let services = &state.synchronized_services;
    let mut starts: Vec<_> = services.iter().map(|s| s.domain[0] as f64).collect();
    let mut used = vec![0.; services.len()];
    let mut domains = HashMap::new();
    for oracle in &state.oracles {
        domains.insert(
            oracle.network.id,
            resource_groups::domain_graph(oracle)?.into_owned(),
        );
    }
    let membership = |network: i32, vertex: usize| -> Option<usize> {
        services.iter().position(|s| {
            s.definition
                .members
                .iter()
                .any(|m| m.network == network && m.vertex == vertex)
        })
    };
    for fixed in routes {
        for &v in &fixed.route.route.nodes {
            if let Some(i) = usize::try_from(v)
                .ok()
                .and_then(|v| membership(fixed.route.network, v))
            {
                used[i] += fixed.quantity;
            }
        }
    }
    if used
        .iter()
        .zip(services)
        .any(|(&n, s)| n != 0. && n != f64::from(s.definition.vehicles_required))
    {
        return Ok(None);
    }
    if services
        .iter()
        .zip(&used)
        .all(|(s, &n)| n == 0. || s.domain[0] == s.domain[1])
    {
        return schedule_at(state, routes, domains, &used, &starts);
    }
    // A bounded heuristic avoids spending the node's whole budget on a
    // positive temporal cycle. The finite time-domain tree remains complete.
    for _ in 0..1024 {
        if Instant::now() >= state.deadline {
            return Ok(None);
        }
        let mut changed = false;
        for fixed in routes {
            let graph = &domains[&fixed.route.network];
            let axes: HashSet<_> = services
                .iter()
                .filter(|s| {
                    s.definition
                        .members
                        .iter()
                        .any(|m| m.network == fixed.route.network)
                })
                .map(|s| s.definition.resource)
                .collect();
            for r in axes {
                let Some(initial) = graph.root_resource(r, true) else {
                    return Ok(None);
                };
                let mut value = crate::master::numerics::ExactSum::default();
                if value.add(initial).is_none() {
                    return Ok(None);
                }
                for &id in &fixed.route.route.arcs {
                    let Some(arc) = graph.arcs.iter().find(|a| a.id == id) else {
                        return Ok(None);
                    };
                    let i = membership(fixed.route.network, arc.head)
                        .filter(|&i| services[i].definition.resource == r);
                    let bounds = graph.vertices[arc.head].bounds[r];
                    let mut lower = bounds.lower;
                    let mut upper = bounds.upper;
                    if let Some(w) = arc.bounds[r] {
                        lower = lower.max(w.lower);
                        upper = upper.min(w.upper);
                    }
                    if let Some(i) = i {
                        lower = lower.max(starts[i]);
                        upper = upper.min(services[i].domain[1] as f64);
                    }
                    if value.add(arc.consumption[r]).is_none() {
                        return Ok(None);
                    }
                    let windows = vec![Window { lower, upper }];
                    #[cfg(feature = "multi-time-windows")]
                    let windows = if graph.time_windows(arc.head, r).is_empty() {
                        windows
                    } else {
                        graph
                            .time_windows(arc.head, r)
                            .iter()
                            .map(|w| Window {
                                lower: lower.max(w.lower),
                                upper: upper.min(w.upper),
                            })
                            .collect()
                    };
                    let mut feasible = None;
                    for window in windows {
                        if window.lower > window.upper {
                            continue;
                        }
                        let mut next = value.clone();
                        let mut difference = next.clone();
                        if difference.add(-window.lower).is_none() {
                            return Ok(None);
                        }
                        if difference.sign() < 0. {
                            next = crate::master::numerics::ExactSum::default();
                            if next.add(window.lower).is_none() {
                                return Ok(None);
                            }
                        }
                        let Some(mut candidate) = next.upper() else {
                            return Ok(None);
                        };
                        if i.is_some() {
                            candidate = candidate.ceil();
                            // An upward floating bound can cross an integer;
                            // compare the exact expansion before choosing it.
                            let mut previous = next.clone();
                            if previous.add(-(candidate - 1.)).is_none() {
                                return Ok(None);
                            }
                            if previous.sign() <= 0. {
                                candidate -= 1.;
                            }
                            next = crate::master::numerics::ExactSum::default();
                            if next.add(candidate).is_none() {
                                return Ok(None);
                            }
                        }
                        let mut difference = next.clone();
                        if difference.add(-window.upper).is_none() {
                            return Ok(None);
                        }
                        if difference.sign() <= 0. {
                            feasible = Some((next, candidate));
                            break;
                        }
                    }
                    let Some((next, candidate)) = feasible else {
                        return Ok(None);
                    };
                    value = next;
                    if let Some(i) = i {
                        if candidate > starts[i] {
                            starts[i] = candidate;
                            changed = true;
                        }
                    }
                }
            }
        }
        if changed {
            continue;
        }
        return schedule_at(state, routes, domains, &used, &starts);
    }
    Ok(None)
}

fn schedule_at<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
    mut domains: HashMap<i32, Graph<D>>,
    used: &[f64],
    starts: &[f64],
) -> Result<Option<Vec<FixedRoute>>> {
    for (i, service) in state.synchronized_services.iter().enumerate() {
        if used[i] == 0. {
            continue;
        }
        for member in &service.definition.members {
            let graph = domains.get_mut(&member.network).unwrap();
            let old = graph.vertices[member.vertex].bounds[service.definition.resource];
            if starts[i] < old.lower || starts[i] > old.upper {
                graph.arcs.retain(|a| a.head != member.vertex);
                continue;
            }
            graph.set_bounds(
                member.vertex,
                service.definition.resource,
                starts[i],
                starts[i],
            )?;
        }
    }
    let mut scheduled = routes.to_vec();
    for fixed in &mut scheduled {
        let problem = crate::master::routing::RoutingContext::set_partitioning(
            domains[&fixed.route.network].clone(),
            1.,
        )?;
        if crate::master::routing::exact_route_feasible(&problem, &fixed.route.route).is_none() {
            return Ok(None);
        }
        let Some(route) = problem.elementary_route(&fixed.route.route.arcs) else {
            return Ok(None);
        };
        fixed.route.route.resources = route.resources;
        fixed.route.route.resource_profile = route.resource_profile;
    }
    Ok(Some(scheduled))
}
