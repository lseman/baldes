# Standalone AILS-II search

Run the heuristic independently with:

```sh
baldes ails-ii instances/cvrplib/E-n22-k4.vrp --time 60 --seed 0
```

`--iterations N` limits the number of perturbation and improvement iterations.
`--initial-solution FILE` loads a CVRPLIB route plan, `--get-bks` fetches a
starting plan, and `--solution-output FILE` writes the final plan. Without an
initial plan, the command constructs one subject to the instance fleet,
capacity, and time windows. Construction may fail even when a feasible plan
exists; supplying an initial plan can be useful for tightly constrained cases.

The search maintains a current plan and an elite set, applies adaptive
perturbation, repairs constraint violations where possible, and returns the best
independently validated plan. It can use SISR string removals and greedy
recreation. The implementation is inspired by AILS-II and supports time
windows; its repair and SWAP* neighborhoods are not the published granular
implementations.

The AILS-II module and search do not invoke HGS. HGS is a separate command;
`--population-size` and `--nb-iter` apply to HGS, not to AILS-II.
