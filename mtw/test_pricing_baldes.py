"""MTW pricing checks using BALDES instead of the pure Python bucket labeler.

Build pybaldes with multi-time-windows, then run from the repository root:
    python -m pytest mtw/test_pricing_baldes.py -q

The independent DFS uses full-service windows and finish times, matching
mtw/test_pricing.py. No Gurobi installation or license is needed. These tests
cover cut-free pricing; rank-one cut states and Python bucket-arc elimination
are not exposed by the low-level Python solver.
"""
import math
import random
from types import SimpleNamespace

import pytest

pybaldes = pytest.importorskip("pybaldes")
from baldes_pricing import price_group

pytestmark = pytest.mark.skipif(
    not hasattr(pybaldes.Solver, "set_vertex_time_windows"),
    reason="requires pybaldes built with multi-time-windows",
)


def instance(durations, windows, dist, arcs=None, horizon=80, initial=0):
    n = len(durations)
    if arcs is None:
        arcs = {(u, v) for u in range(n - 1) for v in range(1, n)
                if u != v and (u, v) != (0, n - 1)}
    group = SimpleNamespace(windows=windows, arcs=set(arcs))
    return SimpleNamespace(n=n, s=durations, dist=dist, start=0, end=n - 1,
                           job_ids=list(range(1, n - 1)), t0=initial, T=horizon,
                           groups=[group], copies={})


def random_instance(seed):
    rnd = random.Random(seed)
    n = rnd.randint(4, 7)
    durations = [0] + [rnd.randint(8, 16) for _ in range(n - 2)] + [0]
    windows = {}
    for v in range(1, n - 1):
        windows[v] = sorted((e, min(80, e + durations[v] + rnd.randint(0, 25)))
                            for e in [rnd.randint(0, 80 - durations[v])
                                      for _ in range(rnd.randint(1, 3))])
    pts = [(rnd.uniform(0, 12), rnd.uniform(0, 12)) for _ in range(n)]
    pts[-1] = pts[0]
    dist = [[math.dist(p, q) for q in pts] for p in pts]
    inst = instance(durations, windows, dist)
    size = rnd.choice([1, 2, n - 2])
    ng = [0] * n
    for v in inst.job_ids:
        neighbors = sorted(inst.job_ids, key=lambda w: (dist[v][w], w))[:size]
        ng[v] = sum(1 << w for w in neighbors)
    pi = [0.] + [rnd.uniform(0., 1.8) for _ in inst.job_ids] + [0.]
    return inst, pi, ng, rnd.uniform(0., 1.)


def enumerate_routes(inst, pi, ng, *, mu=0., removed=frozenset(),
                     banned=frozenset(), windows=None, arc_cost=None):
    """Independent exhaustive NG route enumeration, without BALDES transitions."""
    group = inst.groups[0]
    windows = group.windows if windows is None else windows
    arc_cost = {} if arc_cost is None else arc_cost
    out = {}

    def visit(path, finish, memory, cost):
        u = path[-1]
        for v in inst.job_ids + [inst.end]:
            arc = (u, v)
            if arc not in group.arcs or arc in removed or v in banned:
                continue
            arrival = finish + inst.dist[u][v]
            if v == inst.end:
                if u != inst.start and arrival <= inst.T:
                    out[tuple(path + [v])] = cost + mu + arc_cost.get(arc, 0.)
                continue
            if memory & (1 << v):
                continue
            start = next((max(arrival, e) for e, l in windows[v]
                          if max(arrival, e) + inst.s[v] <= l), None)
            if start is not None:
                visit(path + [v], start + inst.s[v], (memory & ng[v]) | (1 << v),
                      cost + pi[v] - 1. + arc_cost.get(arc, 0.))
    visit([inst.start], inst.t0, 0, 0.)
    return out


def compare(inst, pi, ng, **options):
    routes = enumerate_routes(inst, pi, ng, **{k: v for k, v in options.items()
                                              if k not in {"mode", "buckets", "maximum"}})
    result = price_group(inst, inst.groups[0], pi, ng, **options)
    if not routes:
        assert result.minimum is None and not result.columns
    else:
        assert result.minimum == pytest.approx(min(routes.values()), abs=1e-6)
    assert len({path for _, path in result.columns}) == len(result.columns)
    for cost, path in result.columns:
        assert path in routes
        assert cost == pytest.approx(routes[path], abs=1e-6)
        assert cost < 0.
    if routes and min(routes.values()) < -1e-6:
        assert result.columns
        assert result.columns[0][0] == pytest.approx(min(routes.values()), abs=1e-6)
    return result


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_random_ng_pricing_matches_exhaustive_routes(seed, mode):
    inst, pi, ng, mu = random_instance(seed)
    compare(inst, pi, ng, mu=mu, mode=mode, buckets=[3, 10, 25][seed % 3])


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_full_service_must_fit_and_waiting_crosses_a_gap(mode):
    inst = instance([0, 3, 0], {1: [(0, 5), (10, 13)]},
                    [[0, 4, 0], [0, 0, 2], [0, 0, 0]], horizon=15)
    result = compare(inst, [0., 0., 0.], [0, 2, 0], mode=mode)
    assert result.columns == ((-1., (0, 1, 2)),)
    # Starting at 4 fits the first raw interval but does not fit full service;
    # the return takes until 15. An earlier horizon therefore forbids the route.
    inst.T = 14
    compare(inst, [0., 0., 0.], [0, 2, 0], mode=mode)


@pytest.mark.parametrize("seed", range(12))
def test_branch_exclusions_window_intersections_and_arc_duals(seed):
    inst, pi, ng, mu = random_instance(seed)
    windows = {v: [(max(e, 20.), min(l, 60.)) for e, l in ws
                   if max(e, 20.) + inst.s[v] <= min(l, 60.)]
               for v, ws in inst.groups[0].windows.items()}
    compare(inst, pi, ng, mu=mu, removed={(0, 1)}, banned={2}, windows=windows,
            arc_cost={(1, inst.end): .35}, maximum=3)


def test_multiple_columns_and_positive_minimum():
    inst = instance([0, 1, 1, 0], {1: [(0, 10)], 2: [(0, 10)]},
                    [[0.] * 4 for _ in range(4)],
                    arcs={(0, 1), (1, 3), (0, 2), (2, 3)}, horizon=10)
    result = compare(inst, [0., 0., .2, 0.], [0, 2, 4, 0], maximum=10)
    assert len(result.columns) == 2
    result = compare(inst, [0., 2., 3., 0.], [0, 2, 4, 0], maximum=10)
    assert result.minimum == pytest.approx(1.) and not result.columns


def test_ng_memory_allows_a_repeat_but_full_memory_forbids_it():
    inst = instance([0, 2, 2, 0], {1: [(0, 6)], 2: [(0, 6)]},
                    [[0.] * 4 for _ in range(4)],
                    arcs={(0, 1), (1, 2), (2, 1), (1, 3)}, horizon=6)
    relaxed = compare(inst, [0.] * 4, [0, 2, 4, 0])
    elementary = compare(inst, [0.] * 4, [0, 6, 6, 0])
    assert relaxed.minimum == -3.
    assert elementary.minimum == -1.


def test_nonzero_depot_time_and_zero_width_start_window():
    inst = instance([0, 3, 0], {1: [(10, 13)]}, [[0.] * 3 for _ in range(3)],
                    horizon=13, initial=7)
    compare(inst, [0.] * 3, [0, 2, 0])


@pytest.mark.parametrize("seed", range(80))
@pytest.mark.parametrize("mode,split_fraction", [(0, math.inf), (1, .5), (2, -math.inf)])
def test_previous_python_labeler_and_baldes_agree(seed, mode, split_fraction):
    # The historical MTW sources are local reference files, intentionally ignored.
    # Keep the new suite runnable without them while exercising parity when present.
    old = pytest.importorskip("bucket_graph")
    inst, pi, ng, mu = random_instance(seed)
    group = inst.groups[0]
    static = old.build_static(
        inst.n, inst.s, inst.dist, inst.t0, inst.T,
        [(group.windows, group.arcs)], [3., 10., 40.][seed % 3], 1,
    )
    static["groups"][0]["ng"] = ng
    cuts = dict(sigma=[], den=[], cuts_of=[[] for _ in range(inst.n)], keep={})
    split = split_fraction * inst.T
    empty = frozenset()
    labels = [old.LabelSet(static, 0, direction, old.run_labeling(
        static, 0, direction, pi, cuts, empty, empty, empty, {}, split, True,
    )) for direction in (old.FWD, old.BWD)]
    _, previous = old.concatenate(
        static, 0, *labels, cuts, empty, empty, math.inf, 30,
    )
    # The old forward-only and backward-only splits are compared with BALDES
    # forward and halfway modes respectively; each must find the same optimum.
    result = compare(inst, pi, ng, mu=mu, mode=mode)
    if previous is None:
        assert result.minimum is None
    else:
        assert result.minimum == pytest.approx(previous + mu, abs=1e-6)


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_reused_pricer_tracks_duals_and_restrictions_without_stale_labels(mode):
    from baldes_pricing import GroupPricer
    inst, _, ng, _ = random_instance(12)
    pricer = GroupPricer(inst, inst.groups[0], capacity=2)
    for pi in ([0.] * inst.n, [0., 2., .1, .2, 0.] if inst.n == 5 else [1.] * inst.n,
               [.3] * inst.n):
        reference = enumerate_routes(inst, pi, ng, mu=.2)
        result = pricer.price(pi, ng, mu=.2, mode=mode)
        assert result.exact
        expected = min(reference.values(), default=None)
        if expected is None:
            assert result.minimum is None
        else:
            assert result.minimum == pytest.approx(expected, abs=1e-6)
        for cost, path in result.columns:
            assert cost == pytest.approx(reference[path], abs=1e-6)
    assert pricer.preparations == 1
    assert pricer.native_calls == 3
    pi = [.3] * inst.n
    changed = pricer.price(pi, ng, mu=.2, mode=mode, banned={1})
    assert all(1 not in path for _, path in changed.columns)
    parent = pricer.price(pi, ng, mu=.2, mode=mode)
    assert parent == result
    assert pricer.preparations == 2
    assert pricer.cached_results == 1


def test_reused_pricer_distinguishes_ng_windows_arcs_and_restores_parent():
    from baldes_pricing import GroupPricer
    inst = instance([0, 2, 2, 0], {1: [(0, 6)], 2: [(0, 6)]},
                    [[0.] * 4 for _ in range(4)],
                    arcs={(0, 1), (1, 2), (2, 1), (1, 3)}, horizon=6)
    p = GroupPricer(inst, inst.groups[0])
    parent = p.price([0.] * 4, [0, 2, 4, 0])
    assert parent.minimum == -3.
    assert p.price([0.] * 4, [0, 6, 6, 0]).minimum == -1.
    assert p.price([0.] * 4, [0, 2, 4, 0], removed={(1, 2)}).minimum == -1.
    assert p.price([0.] * 4, [0, 2, 4, 0], windows={1: [(0, 2)], 2: []}).minimum == -1.
    assert p.price([0.] * 4, [0, 2, 4, 0]) == parent
    assert p.preparations == 4 and p.cached_results == 1
    p.price([0.] * 4, [0, 2, 4, 0], banned={2})
    assert len(p.cache) == p.capacity


def test_heuristic_columns_are_feasible_and_never_expose_an_unproved_bound():
    from baldes_pricing import GroupPricer
    inst, _, ng, _ = random_instance(3)
    p = GroupPricer(inst, inst.groups[0])
    pi = [0.] * inst.n
    truth = enumerate_routes(inst, pi, ng)
    heuristic = p.price(pi, ng, exact=False)
    for cost, path in heuristic.columns:
        assert cost == pytest.approx(truth[path], abs=1e-6)
    if not heuristic.exact:
        assert heuristic.minimum is None
    exact = p.price(pi, ng, exact=True)
    assert exact.exact and exact.minimum == pytest.approx(min(truth.values()), abs=1e-6)
    calls = p.native_calls
    assert p.price(pi, ng, exact=False) == exact
    assert p.native_calls == calls


def test_reduced_cost_updates_preserve_original_costs_and_validate_atomically():
    solver = pybaldes.Solver(3, 0)
    solver.new_arc(0, 1, 2.)
    solver.new_arc(1, 2, 3.)
    solver.set_reduced_costs([-4., -2.])
    minimum, routes, exact = solver.price_with_status(10, exact=True)
    assert exact and minimum == -6.
    assert routes[0].original_cost == 5.
    for invalid in ([1.], [math.nan, 0.], [0., math.inf]):
        with pytest.raises(ValueError):
            solver.set_reduced_costs(invalid)
        assert solver.price_with_status(10, exact=True)[0] == -6.
    solver.set_reduced_costs([1., 2.])
    minimum, routes, exact = solver.price_with_status(10, exact=True)
    assert exact and minimum == 3.
    assert routes[0].original_cost == 5.
    for invalid in [0, -1]:
        with pytest.raises(ValueError):
            solver.price_with_status(invalid, exact=True)


def test_explicit_exact_pricing_overrides_and_restores_heuristic_schedule():
    solver = pybaldes.Solver(3, 0)
    solver.new_arc(0, 1, -2.)
    solver.new_arc(1, 2, 0.)
    solver.set_option('BALDESheurLabelingStrategy', 2.)
    before = solver.get_phase(0)
    assert before.labels_per_bucket == 16
    minimum, routes, exact = solver.price_with_status(10, exact=True)
    assert exact and minimum == -2.
    assert routes[0].is_certified
    after = solver.get_phase(0)
    assert (after.check_ng, after.allow_jump_arcs, after.labels_per_bucket) == (
        before.check_ng, before.allow_jump_arcs, before.labels_per_bucket,
    )
