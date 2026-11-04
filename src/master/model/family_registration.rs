//! Explicit generic-family registration order, independent of generator names.
use super::*;

/// The single configuration and prepared lookup state for all branching families.
#[derive(Clone, Debug)]
pub struct BranchingRegistry {
    prepared: Option<CompiledBranchingRegistry>,
    /// Actual construction order recorded at the single registration entry point.
    creation_order: Vec<BranchFamilyRegistration>,
    pub(super) defaulted_aggregate_admission: HashSet<String>,
    pub(crate) branching: NetworkBranchingPolicy,
    pub(crate) branch_priorities: NetworkBranchPriorities,
    pub(crate) strong_branching: Option<crate::master::tree::StrongBranchingOptions>,
    pub(crate) use_branching_history: bool,
    pub(crate) mapped_variables: Vec<MappedBranchVariable>,
    pub(crate) branch_selection: BTreeMap<String, BranchSelection>,
    pub(crate) family_registration_order: Vec<BranchFamilyRegistration>,
    pub(crate) branch_directions: BTreeMap<String, NetworkBranchDirection>,
    pub(crate) aggregate_admission: BTreeMap<String, f64>,
    pub(crate) aggregate_projection: BTreeMap<String, AggregateProjection>,
    pub(crate) paths_per_network: Vec<PathsPerNetworkBranching>,
    pub(crate) component_classes: Vec<ComponentClassBranching>,
    pub(crate) component_generators: Vec<ComponentClassGenerator>,
    pub(crate) packing: Option<PackingBranching>,
    pub(crate) clusters: Vec<NetworkClusterBranching>,
    pub(crate) ryan_foster: Option<RyanFosterBranching>,
}

/// Validated derived lookups are published together and invalidated together.
#[derive(Clone, Debug)]
struct CompiledBranchingRegistry {
    declaration: HashMap<BranchFamilyRegistration, usize>,
    mapped_families: HashMap<String, (BranchFamilyRegistration, usize)>,
    selection_order: HashMap<String, usize>,
}

impl Default for BranchingRegistry {
    fn default() -> Self {
        Self {
            prepared: None,
            creation_order: vec![],
            defaulted_aggregate_admission: HashSet::new(),
            branching: NetworkBranchingPolicy::default(),
            branch_priorities: NetworkBranchPriorities::default(),
            strong_branching: None,
            use_branching_history: true,
            mapped_variables: vec![],
            branch_selection: BTreeMap::new(),
            family_registration_order: vec![],
            branch_directions: BTreeMap::new(),
            aggregate_admission: BTreeMap::new(),
            aggregate_projection: BTreeMap::new(),
            paths_per_network: vec![],
            component_classes: vec![],
            component_generators: vec![],
            packing: None,
            clusters: vec![],
            ryan_foster: None,
        }
    }
}

impl ModelTreeOptions {
    pub(super) fn invalidate_branching_cache(&mut self) {
        self.branching_registry.prepared = None;
    }

    pub(super) fn compile_branching_registry(&mut self) -> Result<()> {
        self.branching_registry.prepared = None;
        validate(self)?;
        let declaration = registration_order(self)?.into_owned();
        let mapped_families = mapped_families(self).into_owned();
        let selection_order = selection_order(self).into_owned();
        self.branching_registry.prepared = Some(CompiledBranchingRegistry {
            declaration,
            mapped_families,
            selection_order,
        });
        Ok(())
    }

    #[inline]
    pub(super) fn branching_registry(&self) -> &BranchingRegistry {
        &self.branching_registry
    }
}

/// A branching family or a setting attached to a named generator.
///
/// Register these through [`BranchingRegistry::register`]. The
/// registry owns every branching configuration and its prepared lookups.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "kebab-case")]
pub enum BranchingRegistration {
    Policy(NetworkBranchingPolicy),
    Priorities(NetworkBranchPriorities),
    StrongBranching(Option<crate::master::tree::StrongBranchingOptions>),
    History(bool),
    Mapped(MappedBranchVariable),
    Expression(BranchingExpression),
    AggregateVariable(AggregateVariableBranching),
    PathsPerNetwork(PathsPerNetworkBranching),
    ComponentClass(ComponentClassBranching),
    ComponentGenerator(ComponentClassGenerator),
    Packing(PackingBranching),
    Cluster(NetworkClusterBranching),
    RyanFoster(RyanFosterBranching),
    Selection {
        name: String,
        selection: BranchSelection,
    },
    Direction {
        name: String,
        direction: NetworkBranchDirection,
    },
    AggregateAdmission {
        name: String,
        threshold: f64,
    },
    #[serde(skip)]
    AggregateProjection {
        name: String,
        projection: AggregateProjection,
    },
    FamilyOrder(BranchFamilyRegistration),
}

impl BranchingRegistry {
    /// Policy applied to inferred family priorities and default child order.
    pub fn policy(&self) -> NetworkBranchingPolicy {
        self.branching
    }

    /// Priorities of inferred branching families.
    pub fn priorities(&self) -> &NetworkBranchPriorities {
        &self.branch_priorities
    }

    /// Phased strong-branching configuration, when enabled.
    pub fn strong_branching(&self) -> Option<&crate::master::tree::StrongBranchingOptions> {
        self.strong_branching.as_ref()
    }

    /// Whether retained branching history participates in selection.
    pub fn history_enabled(&self) -> bool {
        self.use_branching_history
    }

    /// Registered mapped variables in construction order.
    pub fn mapped_variables(&self) -> &[MappedBranchVariable] {
        &self.mapped_variables
    }

    /// Selection controls keyed by generator name.
    pub fn selections(&self) -> &BTreeMap<String, BranchSelection> {
        &self.branch_selection
    }

    /// Direction overrides keyed by generator name.
    pub fn directions(&self) -> &BTreeMap<String, NetworkBranchDirection> {
        &self.branch_directions
    }

    /// Registered per-network path-count families.
    pub fn paths_per_network(&self) -> &[PathsPerNetworkBranching] {
        &self.paths_per_network
    }

    /// Registered explicit component classes.
    pub fn component_classes(&self) -> &[ComponentClassBranching] {
        &self.component_classes
    }

    /// Registered automatic component-family generators.
    pub fn component_generators(&self) -> &[ComponentClassGenerator] {
        &self.component_generators
    }

    /// Registered cluster families.
    pub fn clusters(&self) -> &[NetworkClusterBranching] {
        &self.clusters
    }

    /// Registered packing-assignment family, if present.
    pub fn packing(&self) -> Option<&PackingBranching> {
        self.packing.as_ref()
    }

    /// Registered Ryan–Foster family, if present.
    pub fn ryan_foster(&self) -> Option<&RyanFosterBranching> {
        self.ryan_foster.as_ref()
    }

    /// Explicit family construction order used for candidate admission.
    pub fn family_order(&self) -> &[BranchFamilyRegistration] {
        &self.family_registration_order
    }

    /// Family construction events, before mapped entries are grouped by their
    /// configured selection family. Explicit family order overrides this list.
    pub fn construction_order(&self) -> &[BranchFamilyRegistration] {
        &self.creation_order
    }

    /// Register a batch atomically: on an error the options are unchanged.
    pub fn register_batch(
        &mut self,
        registrations: impl IntoIterator<Item = BranchingRegistration>,
    ) -> Result<()> {
        let mut updated = self.clone();
        for registration in registrations {
            updated.register(registration)?;
        }
        *self = updated;
        Ok(())
    }

    /// Add one branching registration. Registrations are applied in call
    /// order. Policy, priority, strong-branching and history controls replace
    /// their previous values; singleton families and named settings reject
    /// duplicates.
    pub fn register(&mut self, registration: BranchingRegistration) -> Result<()> {
        use BranchingRegistration::*;
        // A subsequent solve rebuilds the prepared view from these inputs.
        self.prepared = None;
        match registration {
            Policy(value) => self.branching = value,
            Priorities(value) => self.branch_priorities = value,
            StrongBranching(value) => self.strong_branching = value,
            History(value) => self.use_branching_history = value,
            Mapped(value) => {
                let index = self.mapped_variables.len();
                self.mapped_variables.push(value);
                self.creation_order
                    .push(BranchFamilyRegistration::MappedRegistration(index));
            }
            Expression(_) | AggregateVariable(_) => {
                return Err(invalid(
                    "original-variable branching requires a pricing-variable registry",
                ));
            }
            PathsPerNetwork(value) => {
                let index = self.paths_per_network.len();
                self.paths_per_network.push(value);
                self.creation_order
                    .push(BranchFamilyRegistration::PathsPerNetwork(index));
            }
            ComponentClass(value) => {
                let index = self.component_classes.len();
                self.component_classes.push(value);
                self.creation_order
                    .push(BranchFamilyRegistration::ComponentClass(index));
            }
            ComponentGenerator(value) => {
                let index = self.component_generators.len();
                self.component_generators.push(value);
                self.creation_order
                    .push(BranchFamilyRegistration::ComponentGenerator(index));
            }
            Cluster(value) => {
                let index = self.clusters.len();
                self.clusters.push(value);
                self.creation_order
                    .push(BranchFamilyRegistration::Cluster(index));
            }
            Packing(value) => {
                if self.packing.is_some() {
                    return Err(invalid("duplicate packing branching registration"));
                }
                self.packing = Some(value);
                self.creation_order
                    .push(BranchFamilyRegistration::PackingAssignment);
            }
            RyanFoster(value) => {
                if self.ryan_foster.is_some() {
                    return Err(invalid("duplicate Ryan–Foster branching registration"));
                }
                self.ryan_foster = Some(value);
                self.creation_order
                    .push(BranchFamilyRegistration::PackingRyanFoster);
            }
            Selection { name, selection } => {
                if self.branch_selection.contains_key(&name) {
                    return Err(invalid("duplicate branching selection registration"));
                }
                self.branch_selection.insert(name, selection);
            }
            Direction { name, direction } => {
                if self.branch_directions.contains_key(&name) {
                    return Err(invalid("duplicate branching direction registration"));
                }
                self.branch_directions.insert(name, direction);
            }
            AggregateAdmission { name, threshold } => {
                if self.aggregate_admission.contains_key(&name)
                    && !self.defaulted_aggregate_admission.remove(&name)
                {
                    return Err(invalid("duplicate aggregate admission registration"));
                }
                self.aggregate_admission.insert(name, threshold);
            }
            AggregateProjection { name, projection } => {
                if self.aggregate_projection.contains_key(&name) {
                    return Err(invalid("duplicate aggregate projection registration"));
                }
                self.aggregate_projection.insert(name, projection);
            }
            FamilyOrder(value) => self.family_registration_order.push(value),
        }
        Ok(())
    }
}

/// Declared construction order of registered branching families. Individual
/// mapped slots refer to their shared `BranchSelection.family` when supplied.
/// This order affects candidate admission; it does not change child constraints.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BranchFamilyRegistration {
    MappedFamily(String),
    MappedRegistration(usize),
    ComponentGenerator(usize),
    ComponentClass(usize),
    PathsPerNetwork(usize),
    Cluster(usize),
    PackingAssignment,
    PackingResource,
    PackingRyanFoster,
}

pub(super) fn canonical(
    options: &ModelTreeOptions,
    registration: &BranchFamilyRegistration,
) -> Result<BranchFamilyRegistration> {
    use BranchFamilyRegistration::*;
    match registration {
        MappedFamily(family) => {
            if family.trim().is_empty()
                || !options
                    .branching_registry()
                    .mapped_variables
                    .iter()
                    .any(|variable| {
                        options
                            .branching_registry()
                            .branch_selection
                            .get(&variable.name)
                            .is_some_and(|selection| selection.family == *family)
                    })
            {
                return Err(invalid("unknown branching registration family"));
            }
            Ok(registration.clone())
        }
        MappedRegistration(index) => {
            let variable = options
                .branching_registry()
                .mapped_variables
                .get(*index)
                .ok_or_else(|| invalid("unknown mapped branching registration"))?;
            Ok(options
                .branching_registry()
                .branch_selection
                .get(&variable.name)
                .map_or_else(
                    || registration.clone(),
                    |selection| MappedFamily(selection.family.clone()),
                ))
        }
        ComponentGenerator(index)
            if *index < options.branching_registry().component_generators.len() =>
        {
            Ok(registration.clone())
        }
        ComponentClass(index) if *index < options.branching_registry().component_classes.len() => {
            Ok(registration.clone())
        }
        PathsPerNetwork(index) if *index < options.branching_registry().paths_per_network.len() => {
            Ok(registration.clone())
        }
        Cluster(index) if *index < options.branching_registry().clusters.len() => {
            Ok(registration.clone())
        }
        PackingAssignment if options.branching_registry().packing.is_some() => {
            Ok(registration.clone())
        }
        PackingRyanFoster if options.branching_registry().ryan_foster.is_some() => {
            Ok(registration.clone())
        }
        // Resource policy is an implicit adapter registration even when its
        // separator produces no candidate at this node.
        PackingResource => Ok(registration.clone()),
        _ => Err(invalid("unknown or absent branching family registration")),
    }
}

pub(super) fn registration_order(
    options: &ModelTreeOptions,
) -> Result<std::borrow::Cow<'_, HashMap<BranchFamilyRegistration, usize>>> {
    if let Some(registry) = &options.branching_registry.prepared {
        return Ok(std::borrow::Cow::Borrowed(&registry.declaration));
    }
    let mut order = HashMap::new();
    let registry = options.branching_registry();
    let explicit = !registry.family_registration_order.is_empty();
    let registrations = if explicit {
        &registry.family_registration_order
    } else {
        &registry.creation_order
    };
    for (index, registration) in registrations.iter().enumerate() {
        let family = canonical(options, registration)?;
        if explicit && order.contains_key(&family) {
            return Err(invalid("duplicate branching family registration order"));
        }
        order.entry(family).or_insert(index);
    }
    Ok(std::borrow::Cow::Owned(order))
}

pub(super) fn validate(options: &ModelTreeOptions) -> Result<()> {
    let registry = options.branching_registry();
    if registry
        .branch_directions
        .keys()
        .any(|name| name.trim().is_empty())
    {
        return Err(invalid(
            "branch direction requires a nonempty generator name",
        ));
    }
    if let Some(strong) = &registry.strong_branching {
        model_strong::validate(strong)?;
    }
    registration_order(options)?;
    model_mapping::validate_aggregate_admission(
        &registry.mapped_variables,
        &registry.aggregate_admission,
    )?;
    if registry.aggregate_projection.keys().any(|name| {
        !registry
            .mapped_variables
            .iter()
            .any(|variable| &variable.name == name)
    }) {
        return Err(invalid(
            "aggregate projection requires a registered child expression",
        ));
    }
    for registration in &registry.paths_per_network {
        let name = format!("paths:{}", registration.network);
        if registry.mapped_variables.iter().any(|v| v.name == name)
            || registry.component_classes.iter().any(|c| c.name == name)
        {
            return Err(invalid("path-count branching generator name conflict"));
        }
    }
    Ok(())
}

/// Selection metadata can name mapped variables, prepared cluster generators,
/// packing assignments, explicit component classes and per-network path counts.
pub(super) fn validate_selection(options: &ModelTreeOptions) -> Result<()> {
    let registry = options.branching_registry();
    if registry.branch_selection.is_empty()
        || (registry.clusters.is_empty()
            && registry.component_classes.is_empty()
            && registry.paths_per_network.is_empty()
            && registry.packing.is_none())
    {
        return model_mapping::validate_selection(
            &registry.mapped_variables,
            &registry.branch_selection,
        );
    }
    let mut variables = registry.mapped_variables.to_vec();
    let mut names: HashSet<_> = variables.iter().map(|v| v.name.clone()).collect();
    let mut add_generator = |name: String, priority| -> Result<()> {
        if !names.insert(name.clone()) {
            if registry.branch_selection.contains_key(&name) {
                return Err(invalid("ambiguous branching selection generator name"));
            }
        } else if registry.branch_selection.contains_key(&name) {
            variables.push(MappedBranchVariable {
                name,
                priority,
                networks: vec![],
                masters: vec![],
            });
        }
        Ok(())
    };
    for cluster in &registry.clusters {
        for name in cluster.generator_names() {
            add_generator(name.to_owned(), cluster.priority)?;
        }
    }
    for class in &registry.component_classes {
        add_generator(class.name.clone(), class.priority)?;
    }
    for path in &registry.paths_per_network {
        add_generator(format!("paths:{}", path.network), path.priority)?;
    }
    if let Some(packing) = &registry.packing {
        for name in packing.generator_names() {
            add_generator(name.to_owned(), packing.priority)?;
        }
    }
    model_mapping::validate_selection(&variables, &registry.branch_selection)
}

/// Validate component registrations against their network graphs before the
/// tree commits the registry. The lookup also works for resumed snapshots.
pub(super) fn validate_components<'a, const D: usize>(
    options: &ModelTreeOptions,
    mut graph: impl FnMut(i32) -> Option<&'a crate::graph::Graph<D>>,
) -> Result<()> {
    let registry = options.branching_registry();
    let mut class_names = HashSet::new();
    for class in &registry.component_classes {
        if !class_names.insert(&class.name) {
            return Err(invalid("duplicate component class name"));
        }
        class.for_graph(graph(class.network).ok_or_else(|| invalid("component class network"))?)?;
    }
    let mut generator_names = HashSet::new();
    for generator in &registry.component_generators {
        if !generator_names.insert(&generator.name) {
            return Err(invalid("duplicate component generator name"));
        }
        generator.validate(
            graph(generator.network).ok_or_else(|| invalid("component generator network"))?,
        )?;
    }
    Ok(())
}

/// Map every registered generator to its admission family and that family's
/// first registration position. The prepared tree keeps this lookup for all
/// priority groups and nodes.
pub(super) fn mapped_families(
    options: &ModelTreeOptions,
) -> std::borrow::Cow<'_, HashMap<String, (BranchFamilyRegistration, usize)>> {
    if let Some(registry) = &options.branching_registry.prepared {
        return std::borrow::Cow::Borrowed(&registry.mapped_families);
    }
    let registry = options.branching_registry();
    let mut first_family = HashMap::new();
    let mut result = HashMap::new();
    for (index, variable) in registry.mapped_variables.iter().enumerate() {
        let family = match registry.branch_selection.get(&variable.name) {
            Some(selection) => {
                let first = *first_family
                    .entry(selection.family.clone())
                    .or_insert(index);
                (
                    BranchFamilyRegistration::MappedFamily(selection.family.clone()),
                    first,
                )
            }
            None => (BranchFamilyRegistration::MappedRegistration(index), index),
        };
        result.insert(variable.name.clone(), family);
    }
    std::borrow::Cow::Owned(result)
}

/// Stable construction positions for generators that accept shared selection
/// metadata, in the order their families were registered.
pub(super) fn selection_order(
    options: &ModelTreeOptions,
) -> std::borrow::Cow<'_, HashMap<String, usize>> {
    if let Some(registry) = &options.branching_registry.prepared {
        return std::borrow::Cow::Borrowed(&registry.selection_order);
    }
    let registry = options.branching_registry();
    let mut order = HashMap::new();
    if registry.branch_selection.is_empty() {
        return std::borrow::Cow::Owned(order);
    }
    let mut seen = std::collections::HashSet::new();
    let mut index = 0;
    let mut add = |name: &str| {
        if seen.insert(name.to_owned()) {
            if registry.branch_selection.contains_key(name) {
                order.insert(name.to_owned(), index);
            }
            index += 1;
        }
    };
    for family in &registry.creation_order {
        match family {
            BranchFamilyRegistration::MappedRegistration(i) => {
                if let Some(variable) = registry.mapped_variables.get(*i) {
                    add(&variable.name);
                }
            }
            BranchFamilyRegistration::Cluster(i) => {
                if let Some(cluster) = registry.clusters.get(*i) {
                    for name in cluster.generator_names() {
                        add(name);
                    }
                }
            }
            BranchFamilyRegistration::ComponentClass(i) => {
                if let Some(class) = registry.component_classes.get(*i) {
                    add(&class.name);
                }
            }
            BranchFamilyRegistration::PathsPerNetwork(i) => {
                if let Some(path) = registry.paths_per_network.get(*i) {
                    add(&format!("paths:{}", path.network));
                }
            }
            BranchFamilyRegistration::PackingAssignment => {
                if let Some(packing) = &registry.packing {
                    for name in packing.generator_names() {
                        add(name);
                    }
                }
            }
            _ => {}
        }
    }
    // Internal construction may populate a family before it is registered.
    // Include those generators deterministically after registered families.
    for variable in &registry.mapped_variables {
        add(&variable.name);
    }
    for cluster in &registry.clusters {
        for name in cluster.generator_names() {
            add(name);
        }
    }
    for class in &registry.component_classes {
        add(&class.name);
    }
    for path in &registry.paths_per_network {
        add(&format!("paths:{}", path.network));
    }
    if let Some(packing) = &registry.packing {
        for name in packing.generator_names() {
            add(name);
        }
    }
    std::borrow::Cow::Owned(order)
}

/// All declared family priorities, including families that produce no
/// candidate at the current node. Empty families still affect admission.
pub(super) fn registered_priorities(options: &ModelTreeOptions) -> impl Iterator<Item = f64> + '_ {
    let registry = options.branching_registry();
    registry
        .mapped_variables
        .iter()
        .map(|v| v.priority)
        .chain(registry.packing.iter().map(|p| p.priority))
        .chain(registry.clusters.iter().map(|c| c.priority))
        .chain(registry.component_classes.iter().map(|c| c.priority))
        .chain(registry.component_generators.iter().map(|g| g.priority))
        .chain(registry.ryan_foster.iter().map(|r| r.priority))
        .chain(registry.paths_per_network.iter().map(|p| p.priority))
}

/// Priority range occupied by explicitly registered families. Presets use
/// this range to place inferred fallback families without overriding callers.
pub(super) fn registered_priority_range(options: &ModelTreeOptions) -> (f64, f64) {
    registered_priorities(options).fold((0., 0.), |(low, high), priority| {
        (low.min(priority), high.max(priority))
    })
}

/// Generate candidates from the explicitly configured branching families.
pub(super) fn registered_proposals<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    deadline: Option<Instant>,
) -> Result<Vec<model_strong::Proposal>> {
    let mut proposals = model_strong::mapped_proposals(output, options)?;
    for (index, class) in options
        .branching_registry()
        .component_classes
        .iter()
        .enumerate()
    {
        if let Some(mut proposal) = class.proposal(output, options.tolerance)? {
            proposal.origin = Some(model_strong::ProducerFamily::ComponentClass(index));
            proposals.push(proposal);
        }
    }
    let quota = options
        .branching_registry()
        .strong_branching
        .as_ref()
        .and_then(|strong| strong.phases.first())
        .map_or(1, |phase| phase.candidates);
    proposals.extend(component::generated_proposals(
        output,
        &options.branching_registry().component_generators,
        quota,
        options.tolerance,
        deadline,
    )?);
    for (index, clusters) in options.branching_registry().clusters.iter().enumerate() {
        let mut generated = clusters.proposals(output, options.tolerance)?;
        for proposal in &mut generated {
            proposal.origin = Some(model_strong::ProducerFamily::Cluster(index));
        }
        proposals.extend(generated);
    }
    if let Some(ryan_foster) = &options.branching_registry().ryan_foster {
        proposals.extend(ryan_foster.proposals(output)?);
    }
    Ok(proposals)
}

/// Produce registered per-network count candidates at their position in the
/// dispatcher, after mapped/component families and before resource families.
pub(super) fn path_proposals<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
) -> Result<Vec<model_strong::Proposal>> {
    path_counts::proposals(
        output,
        &options.branching_registry().paths_per_network,
        options.tolerance,
    )
}

/// Packing candidate separation is enabled only with strong evaluation.
pub(super) fn packing_proposals<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
) -> Vec<model_strong::Proposal> {
    let registry = options.branching_registry();
    if registry.strong_branching.is_none() {
        return Vec::new();
    }
    registry.packing.as_ref().map_or_else(Vec::new, |packing| {
        packing.proposals(output, options.tolerance)
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/family_registration_tests.rs"]
mod tests;
