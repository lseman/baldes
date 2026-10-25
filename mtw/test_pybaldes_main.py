"""Verify the native benchmark formulation against exhaustive small schedules."""
from functools import lru_cache

import pytest

pytest.importorskip('pybaldes')
from pybaldes import Solver
import dsn_instances
from run_pybaldes_main import make_model, validate_witness

pytestmark = pytest.mark.skipif(
    not hasattr(Solver, 'set_vertex_time_windows'), reason='multi-time-windows build required',
)


def exhaustive_value(data):
    jobs, dist, antennas = dsn_instances.to_jobs(data)
    n = len(jobs) - 2
    choices = []
    for a in range(antennas):
        masks = {0}
        def visit(u, finish, mask):
            for v in range(1, n + 1):
                bit = 1 << (v - 1)
                if mask & bit:
                    continue
                arrival = finish + dist[u][v]
                start = next((max(arrival, e) for e, l in jobs[v].antenna_windows[a]
                              if max(arrival, e) + jobs[v].duration <= l), None)
                if start is not None:
                    new_finish = start + jobs[v].duration
                    if new_finish + dist[v][-1] <= data['horizon']:
                        masks.add(mask | bit)
                        visit(v, new_finish, mask | bit)
        visit(0, 0., 0)
        choices.append(tuple(masks))
    @lru_cache(None)
    def pack(a, served):
        if a == antennas:
            return served.bit_count()
        return max(pack(a + 1, served | mask) for mask in choices[a] if not mask & served)
    return pack(0, 0)


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('per', [1, 2])
def test_native_tree_matches_independent_schedule_packing(seed, per):
    data = dsn_instances.generate(7, seed, days=1, antennas_per_complex=per, dur=(120, 300))
    result = make_model(data).solve(tree=True, time_limit=10.)
    validate_witness(data, result)
    assert result.status == 'Optimal'
    assert -result.incumbent == exhaustive_value(data)
    assert result.bound == pytest.approx(result.incumbent)


def test_arraying_is_rejected():
    data = dsn_instances.generate(6, 1, days=1, arrayed=.5)
    with pytest.raises(ValueError, match='ordinary'):
        make_model(data)


@pytest.mark.parametrize('n', [30, 60, 90])
def test_benchmark_networks_match_original_mtw_instance(n):
    pytest.importorskip('gurobipy')
    from bp_mtw import Instance
    data = dsn_instances.generate(n, 1, days=1, antennas_per_complex=2, dur=(120, 300))
    original = Instance(*dsn_instances.to_jobs(data))
    model = make_model(data)
    assert len(model.networks) == len(original.groups)
    for network, group in zip(model.networks, original.groups):
        assert network.multiplicity == (0., group.count)
        assert {(a.tail, a.head) for a in network.arcs} == group.arcs
        for v in original.job_ids:
            starts = tuple((e, l - original.s[v]) for e, l in group.windows[v])
            assert network.resources[0].time_windows[v] == (starts or None)


@pytest.mark.parametrize('seed', [1, 2])
def test_original_bcp_search_with_baldes_pricing_matches_python(seed):
    pytest.importorskip('gurobipy')
    from bp_mtw import solve_bcp as original_solve
    from bp_mtw_baldes import solve_bcp as baldes_solve
    data = dsn_instances.generate(10, seed, days=1, antennas_per_complex=1, dur=(300, 450))
    inputs = dsn_instances.to_jobs(data)
    options = dict(verbose=False, parallel=False, time_limit=20., root_cut_rounds=0,
                   node_cut_rounds=0, arc_elimination=False)
    old = original_solve(*inputs, **options)
    new = baldes_solve(*inputs, **options)
    assert old['optimal'] and new['optimal']
    assert old['unresolved_nodes'] == new['unresolved_nodes'] == 0
    assert old['value'] == new['value']
    assert old['bound'] == new['bound']
    assert new['baldes_calls'] > 0


def test_baldes_bcp_rejects_ng_growth_beyond_native_capacity():
    pytest.importorskip('gurobipy')
    from bp_mtw_baldes import NG_CAPACITY, BaldesBranchPrice
    with pytest.raises(ValueError, match=f'ng_max <= {NG_CAPACITY}'):
        BaldesBranchPrice(None, ng_max=NG_CAPACITY + 1)
