# Python: pyBaldes

[Documentation index](../README.md)

pyBaldes now has a dedicated [documentation section](../pybaldes/README.md),
covering installation, tutorials, modeling, API signatures, and troubleshooting.
The distribution and import name are both `pybaldes`.

| Task | Guide |
| --- | --- |
| Install or select build features | [Installation](../pybaldes/installation.md) |
| Run your first Python solve | [Quickstart](../pybaldes/quickstart.md) |
| Construct CVRP/VRPTW data and configure a root solve | [Routing inputs and options](../pybaldes/routing.md) |
| Interpret bounds, fractional routes, and exported JSON | [Results and guarantees](../pybaldes/results.md) |
| Integrate a pricing oracle into your own master | [Stateful pricing](../pybaldes/pricing.md) |
| Add coupled resources or explicit function encodings | [Resource modeling](../pybaldes/resources.md) |
| Solve shared-master models and integer trees | [Network models](../pybaldes/networks.md) |
| Look up a public class or method | [API reference](../pybaldes/api.md) |
| Diagnose builds, errors, or objective differences | [Troubleshooting](../pybaldes/troubleshooting.md) |

`pybaldes.solve` accepts instances, paths, scalar pricing networks, arrays and
complete network models. `Instance.solve` and `NetworkProblem.solve` delegate to
it. See the [unified solve guide](solve.md) for normalization and shared defaults.
Routing input returns an LP relaxation; network model input also supports
`tree=True`. A root `Optimal` status describes LP convergence, not an integer
routing optimum. The [handbook](../pybaldes/README.md) documents each input contract.
