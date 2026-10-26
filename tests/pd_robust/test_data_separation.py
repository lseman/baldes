"""Independent scenario and cut-validity checks on exhaustively enumerated routes."""
from dataclasses import replace
from itertools import combinations, permutations

import pytest

from pd_robust.data import Settings, extract_routes, parse_instance
from pd_robust.separators import minimal_infeasible_path, ripei, rrci, separate_ripei, separate_rrci


def fixture(capacity=10, close=100, gamma_q=1, gamma_t=1):
    text = f"""n 2 Q {capacity} r_mod 2
0 0 0 0 0 0 100
1 1 0 0 4 0 50
2 2 0 0 4 0 50
3 3 0 0 -4 0 {close}
4 4 0 0 -4 0 {close}
5 0 0 0 0 0 100
"""
    return parse_instance(text, Settings(gamma_q, gamma_t, 0.5, 0.5, adjust_demands=False))


def scenario_feasible(p, route):
    """Enumerate extreme points; do not use either robust recurrence."""
    for qcount in range(min(p.n, p.settings.gamma_q) + 1):
        for qset in combinations(p.pickups, qcount):
            for tcount in range(min(len(route) - 1, p.settings.gamma_t) + 1):
                for tset in combinations(range(len(route) - 1), tcount):
                    time, load = p.nodes[0].open, 0
                    onboard = set()
                    for k, (i, j) in enumerate(zip(route, route[1:])):
                        time = max(p.nodes[j].open, time + p.nodes[i].service + p.t[i][j]
                                   + (p.dt[i][j] if k in tset else 0))
                        if time > p.nodes[j].close + 1e-7:
                            return False
                        if j <= p.n:
                            onboard.add(j)
                        elif j != p.sink:
                            if j - p.n not in onboard:
                                return False
                            onboard.remove(j - p.n)
                        elif onboard:
                            return False
                        load += p.q[j] + (p.dq[j] if (j if j <= p.n else j - p.n) in qset else 0)
                        if load > p.capacity + 1e-7 or load < -1e-7:
                            return False
    return True


def feasible_routes(p):
    routes = []
    for count in range(2, 2 * p.n + 1):
        for sequence in permutations(p.customers, count):
            if count % 2 or any((i in sequence) != (i + p.n in sequence) for i in p.pickups):
                continue
            route = (0,) + sequence + (p.sink,)
            if scenario_feasible(p, route):
                routes.append(route)
    return routes


def solutions(p):
    all_routes = feasible_routes(p)
    for count in range(1, p.n + 1):
        for selected in combinations(all_routes, count):
            coverage = sorted(v for route in selected for v in route[1:-1])
            if coverage == list(p.customers):
                yield selected


def arc_values(routes):
    return {arc: 1.0 for route in routes for arc in zip(route, route[1:])}


@pytest.mark.parametrize("gq,gt", [(0, 0), (1, 0), (0, 1), (1, 2), (2, 2)])
def test_replay_matches_explicit_extreme_scenarios(gq, gt):
    p = fixture(capacity=10, close=6, gamma_q=gq, gamma_t=gt)
    for sequence in permutations(p.customers):
        route = (0,) + sequence + (p.sink,)
        try:
            p.replay(route)
            feasible = True
        except ValueError:
            feasible = False
        assert feasible == scenario_feasible(p, route), route


def test_rrci_replaces_delivered_top_deviations_and_is_valid_for_every_solution():
    p = fixture(capacity=9, gamma_q=2, gamma_t=0)
    bad = arc_values([(0, 1, 2, 3, 4, 5)])
    cuts = separate_rrci(p, bad)
    assert cuts
    assert any(cut.rhs == 2 and cut.violation(bad) > 0 for cut in cuts)
    for selected in solutions(p):
        values = arc_values(selected)
        for cut in cuts:
            assert cut.violation(values) <= 1e-7
    # Pairing cancellation: {pickup1, delivery1} has no unpaired load.
    assert rrci(p, [1, 3]).rhs == 1


def test_ripei_minimality_depot_tournament_and_validity_on_every_solution():
    p = fixture(capacity=12, close=6, gamma_q=0, gamma_t=2)
    bad_route = (0, 2, 1, 3, 4, 5)
    bad = arc_values([bad_route])
    cuts = separate_ripei(p, bad, integer=True)
    assert cuts
    for cut in cuts:
        assert cut.violation(bad) > 0
        assert p.time_feasible(cut.nodes[1:])
        assert p.time_feasible(cut.nodes[:-1])
        if cut.nodes[0] == 0:
            assert [a for a in cut.arcs if a[0] == 0] == [(0, cut.nodes[1])]
    for selected in solutions(p):
        for cut in cuts:
            assert cut.violation(arc_values(selected)) <= 1e-7
    minimal = minimal_infeasible_path(p, bad_route)
    assert not p.time_feasible(minimal)
    assert p.time_feasible(minimal[1:]) and p.time_feasible(minimal[:-1])


def test_separators_detect_fractional_violations():
    p = fixture(capacity=9, gamma_q=2, gamma_t=0)
    bad = arc_values([(0, 1, 2, 3, 4, 5)])
    good = arc_values([(0, 1, 3, 2, 4, 5)])
    fractional = {a: 0.7 * bad.get(a, 0) + 0.3 * good.get(a, 0) for a in p.arcs}
    assert separate_rrci(p, fractional)
    p = fixture(capacity=12, close=6, gamma_q=0, gamma_t=2)
    bad = arc_values([(0, 2, 1, 3, 4, 5)])
    fractional = {a: 0.9 * bad.get(a, 0) for a in p.arcs}
    assert separate_ripei(p, fractional)


def test_every_nominal_integer_violation_is_separated():
    for p in (fixture(capacity=9, gamma_q=2, gamma_t=2),
              fixture(capacity=12, close=6, gamma_q=1, gamma_t=2)):
        nominal = replace(p, settings=replace(p.settings, gamma_q=0, gamma_t=0))
        for selected in solutions(nominal):
            values = arc_values(selected)
            cuts = separate_rrci(p, values) + separate_ripei(p, values, integer=True)
            if all(scenario_feasible(p, route) for route in selected):
                assert not cuts
            else:
                assert cuts, selected


def test_invalid_input_and_integer_topology_are_rejected():
    for budget in (-1, 0.5, True):
        with pytest.raises(ValueError):
            Settings(gamma_q=budget)
    with pytest.raises(ValueError):
        parse_instance("n 2 Q 10 r_mod 2")
    p = fixture()
    with pytest.raises(ValueError):
        extract_routes(p, {(1, 2): 1, (2, 1): 1})


def test_python_distance_rounding_matches_positive_rust_rounding():
    p = parse_instance("n 1 Q 10 r_mod 2\n0 0 0 0 0 0 100\n1 0.125 0 0 1 0 100\n2 1 0 0 -1 0 100\n3 0 0 0 0 0 100")
    assert p.t[0][1] == 0.13


@pytest.mark.parametrize("settings", [Settings(sigma_q=1e308), Settings(sigma_t=1e308)])
def test_derived_numeric_overflow_is_rejected(settings):
    with pytest.raises(ValueError, match="overflow"):
        parse_instance("n 1 Q 10 r_mod 2\n0 0 0 0 0 0 100\n1 1 0 0 4 0 100\n2 4 0 0 -4 0 100\n3 0 0 0 0 0 100", settings)
