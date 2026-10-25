# Python: PyBaldes

[Documentation index](../README.md)

PyBaldes now has a dedicated [documentation section](../pybaldes/README.md),
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

`solve_root` solves a routing LP relaxation; its `Optimal` status does not prove
an integer VRP optimum. Shared-master models expose their own integer tree via
`solve_networks(..., tree=True)`. Choose the interface that matches your model.

This page remains at its original path so existing documentation links continue
to work. Start new Python documentation links at the dedicated section above.
