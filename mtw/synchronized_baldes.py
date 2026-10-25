"""Native BALDES models for DSN requests requiring simultaneous antennas.

Unlike the Python-master adapter, this runs coverage, synchronization checking,
start-domain branching and pricing in the native shared-master tree.
"""
from collections import Counter
from dataclasses import replace
import math

import dsn_instances
from pybaldes import SynchronizedService, SynchronizedServiceMember
from run_pybaldes_main import make_model


def make_synchronized_model(data, ng_size=8):
    """Return (pricing model, service definitions), with one reward per request."""
    ordinary = dict(data, jobs=[dict(j, antennas_required=1) for j in data['jobs']])
    model = make_model(ordinary, ng_size=ng_size)
    jobs, _, _ = dsn_instances.to_jobs(data)
    synchronized = {v for v in range(1, len(jobs) - 1) if jobs[v].antennas_required > 1}
    networks = tuple(replace(n, arcs=tuple(
        replace(a, cost=0., entries=()) if a.head in synchronized else a
        for a in n.arcs)) for n in model.networks)
    services = []
    for v in sorted(synchronized):
        members = tuple(SynchronizedServiceMember(n.id, v) for n in networks
                        if n.resources[0].time_windows[v])
        if members:
            services.append(SynchronizedService(
                id=v, resource=0, vehicles_required=jobs[v].antennas_required,
                reward=1., start=(0, math.floor(data['horizon'] - jobs[v].duration)),
                members=members))
    return replace(model, networks=networks), tuple(services)


def validate_synchronized_witness(data, result):
    """Replay full-service windows, fleet capacity, coverage and common starts."""
    if result.incumbent is None:
        return
    model, _ = make_synchronized_model(data)
    jobs, dist, _ = dsn_instances.to_jobs(data)
    networks = {n.id: n for n in model.networks}
    counts, visits, starts = Counter(), Counter(), {}
    route_cost = 0.
    for route in result.routes + result.fixed_routes:
        if route.value < 0 or not math.isfinite(route.value) or route.value != round(route.value):
            raise ValueError('noninteger route quantity')
        network = networks[route.network]
        counts[route.network] += route.value
        if counts[route.network] > network.multiplicity[1]:
            raise ValueError('too many routes for an antenna group')
        if route.nodes[0] != 0 or route.nodes[-1] != len(jobs) - 1:
            raise ValueError('invalid depots')
        if len(set(route.nodes)) != len(route.nodes):
            raise ValueError('non-elementary route')
        finish, ordinary = 0., 0
        for pos, (u, v) in enumerate(zip(route.nodes, route.nodes[1:]), 1):
            arrival = finish + dist[u][v]
            if v == len(jobs) - 1:
                if arrival > data['horizon'] + 1e-6:
                    raise ValueError('route finishes after horizon')
                continue
            windows = network.resources[0].time_windows[v] or ()
            if jobs[v].antennas_required > 1:
                start = route.resource_profile[pos][0]
                if abs(start - round(start)) > 1e-6 or start < arrival - 1e-6:
                    raise ValueError('invalid synchronized start')
                if v in starts and abs(starts[v] - start) > 1e-6:
                    raise ValueError('different starts for synchronized request')
                starts[v] = start
            else:
                start = next((max(arrival, e) for e, l in windows
                              if max(arrival, e) <= l + 1e-6), None)
                ordinary += 1
            if start is None or not any(e - 1e-6 <= start <= l + 1e-6 for e, l in windows):
                raise ValueError('service does not fit a full-service window')
            visits[v] += round(route.value)
            finish = start + jobs[v].duration
        if abs(route.cost + ordinary) > 1e-6:
            raise ValueError('route cost counts a synchronized reward')
        route_cost += route.cost * route.value
    if any(n != jobs[v].antennas_required for v, n in visits.items()):
        raise ValueError('incorrect request multiplicity')
    objective = route_cost - len(starts)
    if abs(objective - result.incumbent) > 1e-6 or abs(len(visits) + result.incumbent) > 1e-6:
        raise ValueError('objective differs from served request count')


def solve_synchronized(data, **options):
    """Solve natively and independently check every returned integer witness."""
    model, services = make_synchronized_model(data)
    result = model.solve(tree=True, synchronized_services=services, **options)
    validate_synchronized_witness(data, result)
    return result
