# Branching family registration order

`BranchingRegistration::FamilyOrder` declares the creation order of
registered branching families for candidate admission. JSON `SolveOptions` uses
the shared `branching_registrations` array. The networks CLI accepts that array
from a file with `--tree --branching-registrations registrations.json`.

Ordinary family registrations record their construction order, so they do not
need separate order entries. `family-order` entries are an explicit override,
useful when specifying the position of an implicit producer such as
packing-resource. When an override is present, declare each participating
family whose relative order is needed.

```json
[
  {"kind": "family-order", "config": {"mapped-family": "arcs"}},
  {"kind": "family-order", "config": "packing-assignment"},
  {"kind": "family-order", "config": "packing-resource"},
  {"kind": "family-order", "config": "packing-ryan-foster"},
  {"kind": "family-order", "config": {"component-generator": 0}},
  {"kind": "family-order", "config": {"component-class": 0}},
  {"kind": "family-order", "config": {"paths-per-network": 0}},
  {"kind": "family-order", "config": {"cluster": 0}}
]
```

A mapped family refers to the family name in a `selection` registration.
Alternatively, `{"mapped-registration": 0}` refers to the first mapped
registration; if that variable has shared family metadata, it refers to that
shared family. The indexed `component-generator`, `component-class`,
`paths-per-network`, and `cluster` entries likewise refer to registrations of
their own kind, starting at zero. Packing assignment and Ryan–Foster entries
require their corresponding family registrations. Packing resource is the
implicit resource separator registration, including at nodes where it produces
no candidate. Unknown references and duplicate family references, including
mapped aliases of the same family, are rejected before tree evaluation.

Both `component-class` and `component-generator` entries are accepted in JSON
`SolveOptions.branching_registrations` when `tree` is enabled. Rust callers can use
`BranchingRegistry::register` with `BranchingRegistration` variants
for these families, the ordering declarations, directions, selection metadata,
and the other branching controls. The tree validates and prepares the registry
before using it at each node.

`expression` and `aggregate-variable` registrations require original pricing
identities. JSON solves compile them using the model's pricing-variable registry;
Rust callers use `BranchingRegistry::register_with_pricing` for those entries.

For example, a JSON solve can register an original mapped variable together
with its selection rule and direction:

```json
{
  "tree": true,
  "branching_registrations": [
    {"kind": "policy", "config": "phased"},
    {"kind": "strong-branching", "config": {}},
    {"kind": "mapped", "config": {
      "name": "count42", "priority": 10,
      "networks": [{"network": 42, "per_route": 1}]
    }},
    {"kind": "selection", "config": {
      "name": "count42",
      "selection": {"family": "counts", "rule": "fraction-weighted-priority",
                    "variable_priority": 2, "default_factor": 1}
    }},
    {"kind": "direction", "config": {
      "name": "count42", "direction": "upper-first"
    }}
  ]
}
```

The corresponding CLI file contains the array itself. Python accepts the
array as `branching_registrations=`. Each entry has one `kind` and its `config`;
`strong-branching` with `{}` uses the default phase settings.

Each visited family receives the current candidate quota independently. The
coordinator retains the resulting family sets and divides the remaining fresh
budget between them after reserving history candidates. Construction order
therefore matters when the quota is smaller than the number of families.
Component generators retain their registration identity independently of the
ordinal used to rank published generator instances, and publish at the actual
adaptive quota.

The recorded construction sequence resolves the relative order of registered
producer kinds at a visited priority. If a `family-order` override is used, it
must include every participating family to resolve their relative order. A
missing producer retains the adapter admission policy for that selection.
Empty registrations retain their positions. Master, other activity,
and assignment producers currently lack a recovered shared
owner/order contract and retain the adapter policy when visited; an order array
does not invent their owner. Lower unvisited priorities do not disable resolved
family admission. This option does not change child constraints or history
score formulas.

`selection` registrations can name mapped variables, prepared cluster
generators, packing assignments, explicit component classes, or per-network
path counts.
Cluster names have the form `cluster:<network>:<group>` for a boundary and
`cluster:<network>:<group>:<group>` for a pair. Supply metadata for the
generators to compare with the same family name and rule; `variable_priority`
and `default_factor` then use the same selection rules as mapped variables.
The cluster registration's `priority` sets its family level. Names must resolve
to prepared generators, so explicit partitions make the group numbering easier
to predict. Component classes use their registered names, and path-count names
use `paths:<network>`; packing assignments use `ESA_<network>_<group>`.
Unconfigured generators retain their default ordering. When configured
generators tie under their selection rules, their registration sequence
determines the tie order across families.
