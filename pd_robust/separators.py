"""RRCI (35), RIPEI tournament (36), and the greedy separation of §5."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .data import Arc, Problem, TOLERANCE


@dataclass(frozen=True)
class Cut:
    kind: str
    nodes: tuple[int, ...]
    arcs: tuple[Arc, ...]
    rhs: int

    def violation(self, values: Mapping[Arc, float]) -> float:
        lhs = math.fsum(values.get(a, 0.0) for a in self.arcs)
        return self.rhs - lhs if self.kind == "rrci" else lhs - self.rhs


def rrci(problem: Problem, nodes: Sequence[int]) -> Cut:
    members = frozenset(nodes)
    if len(members) < 2 or not members.issubset(problem.customers):
        raise ValueError("RRCI requires at least two distinct customers")
    leaving = [i for i in members if i <= problem.n and i + problem.n not in members]
    entering = [i - problem.n for i in members if i > problem.n and i - problem.n not in members]
    worst = max(problem.load(leaving), problem.load(entering))
    rhs = max(1, math.ceil(worst / problem.capacity - 1e-12))
    return Cut("rrci", tuple(sorted(members)),
               tuple((i, j) for i, j in problem.arcs if i in members and j not in members), rhs)


def ripei(problem: Problem, path: Sequence[int]) -> Cut:
    if len(path) < 2 or problem.time_feasible(path):
        raise ValueError("RIPEI requires a robust time-infeasible path")
    available = set(problem.arcs)
    arcs = []
    for i, tail in enumerate(path[:-1]):
        for j in range(i + 1, len(path)):
            if tail == 0 and j != i + 1:
                continue
            if path[-1] == problem.sink and i < len(path) - 2 and j == len(path) - 1:
                continue
            if (tail, path[j]) in available:
                arcs.append((tail, path[j]))
    return Cut("ripei", tuple(path), tuple(arcs), len(path) - 2)


def minimal_infeasible_path(problem: Problem, path: Sequence[int]) -> tuple[int, ...]:
    """Direct recurrence replay establishes minimality without an offset shortcut."""
    path = tuple(path)
    if problem.time_feasible(path):
        raise ValueError("path is feasible")
    while len(path) > 2 and not problem.time_feasible(path[:-1]):
        path = path[:-1]
    while len(path) > 2 and not problem.time_feasible(path[1:]):
        path = path[1:]
    return path


def separate_rrci(problem: Problem, values: Mapping[Arc, float], limit: int = 50) -> list[Cut]:
    """Algorithm 1: grow each customer seed by maximum flow from the set.

    Recompute the two unpaired loads from (33), avoiding mutable top-Γ sets.
    """
    cuts = {}
    for seed in problem.customers:
        members = {seed}
        while len(members) < len(problem.customers):
            candidates = [(math.fsum(values.get((i, j), 0) for i in members), -j, j)
                          for j in problem.customers if j not in members]
            flow, _, head = max(candidates)
            if flow <= TOLERANCE:
                break
            members.add(head)
            cut = rrci(problem, tuple(members))
            if cut.violation(values) > TOLERANCE:
                cuts[cut.nodes] = cut
                break
    return sorted(cuts.values(), key=lambda c: (-c.violation(values), c.nodes))[:limit]


def _tournament_flow(problem: Problem, path: Sequence[int], values: Mapping[Arc, float]) -> float:
    # Same L_i(u) depot restrictions as (36); no feasibility assumption needed.
    return math.fsum(values.get((tail, path[j]), 0) for i, tail in enumerate(path[:-1])
                     for j in range(i + 1, len(path))
                     if not (tail == 0 and j != i + 1)
                     and not (path[-1] == problem.sink and i < len(path) - 2 and j == len(path) - 1))


def separate_ripei(problem: Problem, values: Mapping[Arc, float], limit: int = 50,
                   *, integer: bool = False) -> list[Cut]:
    """Algorithm 2: maximum-outgoing-flow paths, DP and minimal infeasible suffixes.

    Redundant integer seeds may be skipped. Fractional seeds are all examined.
    """
    cuts = {}
    excluded: set[int] = set()
    for seed in problem.customers:
        if integer and seed in excluded:
            continue
        path = [0, seed] if values.get((0, seed), 0) > TOLERANCE else [seed]
        while True:
            if not problem.time_feasible(path):
                minimal = minimal_infeasible_path(problem, path)
                cut = ripei(problem, minimal)
                if cut.violation(values) > TOLERANCE:
                    cuts[cut.nodes] = cut
                excluded.update(path[:len(path) - len(minimal) + 1])
                break
            if path[-1] == problem.sink:
                excluded.update(path)
                break
            choices = [(values.get((path[-1], j), 0), -j, j)
                       for j in range(1, problem.sink + 1) if j not in path]
            if not choices:
                break
            flow, _, head = max(choices)
            if flow <= TOLERANCE:
                break
            path.append(head)
            if _tournament_flow(problem, path, values) <= len(path) - 2 + TOLERANCE:
                break
    return sorted(cuts.values(), key=lambda c: (-c.violation(values), c.nodes))[:limit]
