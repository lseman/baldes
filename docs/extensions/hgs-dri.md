# Customer decomposition for HGS

`baldes hgs INSTANCE --dri` enables a customer-based decomposition inspired by
Kerscher and Minner, *Decompose-route-improve framework for solving large-scale
vehicle routing problems with time windows* (2025), `papers/decomposition.pdf`.

```sh
cargo build --release --features clp --bin baldes
target/release/baldes hgs instances/Solomon/C101.txt --get-bks --dri --time 60 \
  --workers 4 --subproblem-customers 200 --dri-vicinity 3 --dri-neighbors 20 \
  --dri-fuzzy 0.8 \
  --solution-output /tmp/C101-dri.sol
```

`--get-bks` or `--initial-solution FILE` supplies an incumbent. The incumbent is
validated and saved first; it is returned if the cluster solution is infeasible,
exceeds the fleet, or fails to improve it. Without an incumbent, failure of a
cluster or of the fleet check is reported as an error. `--dri` and `--intensify`
select separate workflows. Only the distance objective is currently supported.
For downloaded Solomon/Homberger BKSs, `--get-bks` selects truncated tenths;
for a local BKS, select its metric explicitly with `--rounding`.

The symmetric dissimilarity takes the minimum of the two directed values from
equations (11)–(14). Each directed value multiplies spatial distance by a factor
for time-window flexibility, waiting, and combined demand. It uses the instance
travel distance as its spatial measure; the paper's optional polar-angle term is
not included. When time windows or the depot horizon are infinite, the temporal
factor is set to one, an extension for CVRP.

The implementation computes fuzzy memberships using equation (18) with κ = 2
by default (`--dri-kappa` changes it),
and updates medoids using the weighted dissimilarity objective (19). Customers
are assigned to the highest-membership available cluster for routing. A size
cap bounds HGS subproblem memory without storing a full customer-pair matrix.
The fuzzy membership matrix is capped at 50 million entries; increase the
cluster-size limit if a finer partition would exceed that cap.
Customers whose maximum membership exceeds `--dri-fuzzy` (default 0.8) are
excluded as source vertices in the cross-cluster search. Set it to 1 to consider
every customer.

For bounded work, the implementation uses deterministic farthest-first medoid
initialization, samples at most 16 candidate medoids per cluster, and caps the
number of clustering iterations (`--dri-iterations`, default 5). Capacity-bounded hard assignment can differ
from the largest-membership assignment in the paper. It does not implement the
paper's silhouette-based selection of cluster count. Negative dissimilarities,
which can arise for windows extending beyond the depot horizon, are clamped to
zero for membership calculations.

Subproblems use the same depot, customer attributes, distance matrix and metric
as the original instance. They are dispatched through a bounded CPU worker pool;
when an incumbent is supplied, each cluster is seeded with its projected routes
if that projection passes independent feasibility checks.
the routing phase receives 80% of the time remaining after clustering by default
(`--dri-routing-fraction` changes this share), and the
remainder is used for improvement. The available per-cluster time is proportional
to its size and clipped by the shared routing deadline. Each cluster may use up
to one vehicle per customer as in the paper, but the merged plan must respect
the original instance fleet. If the combined clusters exceed the fleet, the
improvement pass first favors moves that reduce the excess; only a complete
plan that passes the original fleet bound can replace the incumbent. Failed or
infeasible subproblems never enter the merged plan.

The improvement pass explores first-improving cross-cluster relocate, swap, and
route-tail crossover moves, as well as single-route 2-opt. Candidate customers and clusters are restricted to
nearest neighborhoods using the same similarity score. Every proposed complete
plan is replayed against the original instance before acceptance; during fleet
repair only, the fleet limit is temporarily relaxed.
Clustering, HGS setup, and individual local-search operations check deadlines
cooperatively, so a single operation may overrun a time limit.
