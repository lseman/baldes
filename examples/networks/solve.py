"""Run the shared JSON example through Python and inspect an assignment child."""
from pathlib import Path
from pybaldes import NetworkProblem

problem = NetworkProblem.from_file(Path(__file__).with_name("triangle.json"))
root = problem.solve()
print(f"root: status={root.status} lp={root.lp}")
candidate = root.snapshot.fractional_assignments()[0]
for assigned in (False, True):
    child = root.snapshot.branch_assignment(candidate.shared_row, candidate.network, assigned)
    result = child.solve()
    print(f"assignment {assigned}: status={result.status} lp={result.lp}")
integer = problem.solve(tree=True)
print(f"tree: status={integer.status} incumbent={integer.incumbent} nodes={integer.nodes}")
