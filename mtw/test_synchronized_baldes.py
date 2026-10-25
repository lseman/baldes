"""Native synchronization compared with independent tiny DSN schedules."""
import pytest

pytest.importorskip('pybaldes')
from synchronized_baldes import make_synchronized_model, solve_synchronized


def instance(windows, required=2):
    return dict(horizon=20, antennas=len(windows), setup=0, jobs=[
        dict(id=1, duration=2, antennas_required=required,
             antenna_windows={a: w for a, w in enumerate(windows)})])


@pytest.mark.parametrize('windows,expected', [
    ([[(0, 10)], [(5, 15)]], 1),
    ([[(0, 4)], [(6, 10)]], 0),
    ([[(0, 4), (10, 15)], [(6, 12)]], 1),
    ([[(0, 10)], [(0, 10)]], 1),
    ([[(0, 10)]], 0),
])
def test_native_service_coverage_and_windows(windows, expected):
    result = solve_synchronized(instance(windows), time_limit=10)
    assert result.status == 'Optimal'
    assert result.incumbent == result.bound == -expected


def test_model_does_not_reward_each_antenna_copy():
    model, services = make_synchronized_model(instance([[(0, 10)], [(0, 10)]]))
    assert len(services) == 1
    assert services[0].reward == 1
    assert all(a.cost == 0 and not a.entries for n in model.networks for a in n.arcs)


def exhaustive(data):
    """Enumerate integer schedules directly on each physical antenna."""
    from collections import Counter
    from itertools import product
    schedules = []
    for a in range(data['antennas']):
        feasible = {()}
        def extend(schedule, finish):
            for j in data['jobs']:
                if any(v == j['id'] for v, _ in schedule):
                    continue
                for t in range(data['horizon'] - j['duration'] + 1):
                    arrival = finish + (data['setup'] if schedule else 0)
                    if t < arrival or not any(e <= t and t + j['duration'] <= l
                                             for e, l in j['antenna_windows'][a]):
                        continue
                    path = schedule + ((j['id'], t),)
                    feasible.add(path)
                    extend(path, t + j['duration'])
        extend((), 0)
        schedules.append(feasible)
    required = {j['id']: j['antennas_required'] for j in data['jobs']}
    best = 0
    for candidate in product(*schedules):
        visits = Counter(v for route in candidate for v, _ in route)
        if any(n != required[v] for v, n in visits.items()):
            continue
        times = {v: {t for route in candidate for job, t in route if job == v}
                 for v in visits}
        if any(len(ts) != 1 for ts in times.values()):
            continue
        best = max(best, len(visits))
    return best


@pytest.mark.parametrize('seed', range(12))
def test_mixed_requests_match_exhaustive_physical_antenna_schedules(seed):
    import random
    rnd = random.Random(seed)
    data = dict(horizon=8, antennas=2, setup=1, jobs=[])
    for v, required in [(1, 2), (2, 1)]:
        windows = {}
        for a in range(2):
            e = rnd.randrange(5)
            windows[a] = [(e, min(8, e + rnd.randrange(2, 6)))]
        data['jobs'].append(dict(id=v, duration=2, antennas_required=required,
                                 antenna_windows=windows))
    result = solve_synchronized(data, time_limit=10)
    assert result.status == 'Optimal'
    assert -result.incumbent == exhaustive(data)
