"""Reusable BALDES pricing for the MTW instance interface in bp_mtw.py.

Rust uses service-start times; MTW stores full service windows and finish times.
Limited-memory rank-1 cuts with arc memory are passed to the native solver;
Python bucket-arc elimination is not represented here.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

from pybaldes import Solver

try:  # pybaldes builds before the NG memory option always had 15 elements
    from pybaldes import ng_memory_capacity as NG_CAPACITY
except ImportError:
    NG_CAPACITY = 15


@dataclass(frozen=True)
class PricingResult:
    # Only exact pricing results expose a minimum usable as a bound.
    minimum: Optional[float]
    columns: Tuple[Tuple[float, Tuple[int, ...]], ...]
    exact: bool = True


class PreparedGroup:
    """One native solver for immutable windows, restrictions and NG memory."""

    def __init__(self, inst, group, ng, windows, removed, banned, mode, buckets):
        solver = Solver(inst.n, 1)
        solver.set_source(inst.start)
        solver.set_sink(inst.end)
        solver.set_initial_resource(0, inst.t0)
        solver.set_bidirectional_mode(mode)
        solver.set_option('BALDESnumberOfBucketsPerVertex', buckets)
        active = {inst.start, inst.end}
        for vertex in range(inst.n):
            solver.set_vertex_bounds(vertex, 0, inst.t0, inst.T)
            neighbors = sorted({v for v in range(inst.n) if ng[vertex] & (1 << v)} | {vertex})
            if len(neighbors) > NG_CAPACITY:
                raise ValueError(f'BALDES native local NG memory supports at most {NG_CAPACITY} elements')
            solver.set_ng_neighborhood(vertex, neighbors)
        for vertex in inst.job_ids:
            if vertex in banned:
                continue
            starts = [(e, l - inst.s[vertex]) for e, l in windows[vertex]
                      if e + inst.s[vertex] <= l]
            if starts:
                solver.set_vertex_time_windows(vertex, 0, starts)
                active.add(vertex)
        self.arcs = []
        self.arc_ids = {}
        self.base_costs = []
        for tail, head in sorted(group.arcs):
            if (tail, head) in removed or tail not in active or head not in active:
                continue
            if (tail, head) == (inst.start, inst.end):
                continue
            cost = -1. if head != inst.end else 0.
            arc = solver.new_arc(tail, head, cost)
            solver.set_arc_consumption(arc, 0, inst.s[tail] + inst.dist[tail][head])
            self.arc_ids[tail, head] = arc
            self.arcs.append((tail, head))
            self.base_costs.append(cost)
        for siblings in getattr(inst, 'copies', {}).values():
            solver.add_elementarity_set(list(siblings))
        # The first native pricing call prepares the buckets. Further calls
        # retain them because only reduced costs change, not original costs.
        self.solver = solver
        self.inst = inst
        self.last_key = None
        self.last_result = None
        self.last_cuts = ()

    def price(self, pi, mu, arc_cost, maximum, exact, cuts=()):
        costs = tuple(base + (mu if head == self.inst.end else pi[head])
                      + arc_cost.get((tail, head), 0.)
                      for (tail, head), base in zip(self.arcs, self.base_costs))
        key = (costs, maximum, cuts)
        if self.last_key == key and (not exact or self.last_result.exact):
            return self.last_result, False
        self.solver.set_reduced_costs(list(costs))
        if cuts != self.last_cuts:
            # The native master convention: a packing row dual -sigma <= 0 adds
            # sigma per unit of the limited-memory coefficient to a route.
            self.solver.set_rank_one_cuts([
                (list(support), den, rhs,
                 sorted(self.arc_ids[a] for a in memory if a in self.arc_ids), -sigma)
                for support, den, rhs, memory, sigma in cuts])
            self.last_cuts = cuts
        minimum, routes, completed = self.solver.price_with_status(maximum, exact=exact)
        if exact and not completed:
            raise RuntimeError('BALDES did not complete exact MTW pricing')
        paths = {}
        for route in routes:
            if route.cost < -1e-6:
                path = tuple(route.nodes())
                paths[path] = min(paths.get(path, float('inf')), route.cost)
        columns = tuple(sorted((cost, path) for path, cost in paths.items()))
        minimum = minimum if completed and minimum < 1e12 else None
        result = PricingResult(minimum, columns, completed)
        self.last_key, self.last_result = key, result
        return result, True


class GroupPricer:
    """Bounded cache of native solvers, scoped to one immutable MTW group.

    Branch/window restrictions and every NG mask are in the cache key. Returning
    to a parent therefore cannot retain a child's restrictions or NG state.
    """

    def __init__(self, inst, group, capacity=4):
        if capacity < 1:
            raise ValueError('cache capacity must be positive')
        self.inst, self.group, self.capacity = inst, group, capacity
        self.cache = OrderedDict()
        self.preparations = 0
        self.native_calls = 0
        self.cached_results = 0

    def price(self, pi, ng, *, mu=0., maximum=30, mode=1, buckets=10,
              removed=frozenset(), banned=frozenset(), windows=None,
              arc_cost=None, cuts=(), exact=True):
        """cuts: tuple of (((vertex, multiplier), ...), den, rhs, frozenset of arcs, sigma)
        with sigma >= 0, as built by rank_one_cuts."""
        if not hasattr(Solver, 'price_with_status'):
            raise RuntimeError('rebuild pybaldes with MTW and status-aware pricing support')
        if maximum <= 0:
            raise ValueError('maximum must be positive')
        if len(pi) != self.inst.n or len(ng) != self.inst.n:
            raise ValueError('pi and ng must contain one entry per vertex')
        windows = self.group.windows if windows is None else windows
        window_key = tuple(tuple(tuple(w) for w in windows[v]) for v in self.inst.job_ids)
        key = (tuple(ng), frozenset(removed), frozenset(banned), window_key, mode, buckets)
        prepared = self.cache.get(key)
        if prepared is None:
            prepared = PreparedGroup(self.inst, self.group, ng, windows, removed, banned, mode, buckets)
            self.cache[key] = prepared
            self.preparations += 1
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        result, ran = prepared.price(pi, mu, {} if arc_cost is None else arc_cost, maximum, exact,
                                     tuple(cuts))
        self.native_calls += ran
        self.cached_results += not ran
        return result


def rank_one_cuts(resources):
    """Active cuts of bp_mtw's dual dict R (sigma, den, cuts_of, keep) in GroupPricer form."""
    sigma, den = resources.get('sigma', ()), resources.get('den', ())
    support = [[] for _ in sigma]
    for vertex, entries in enumerate(resources.get('cuts_of', ())):
        for k, p in entries:
            support[k].append((vertex, p))
    memory = [set() for _ in sigma]
    for arc, mask in resources.get('keep', {}).items():
        for k in range(len(sigma)):
            if mask >> k & 1:
                memory[k].add(arc)
    return tuple((tuple(support[k]), den[k], sum(p for _, p in support[k]) // den[k],
                  frozenset(memory[k]), sigma[k]) for k in range(len(sigma)))


def price_group(inst, group, pi, ng, **options):
    """One-shot pricing; retain a GroupPricer instead for column generation."""
    return GroupPricer(inst, group).price(pi, ng, **options)
