//! Recovered constructor defaults and explicit behavior support.
//! Unsupported settings remain readable; only their recovered default is accepted.
use crate::{invalid, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionKind {
    Int,
    Double,
    Bool,
}
/// Availability of an option's behavior in this port. Implemented options may
/// require a particular operation or an explicitly configured cut family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionSupport {
    Implemented(&'static str),
    Unsupported(&'static str),
}
#[derive(Clone, Copy, Debug)]
pub struct OptionInfo {
    pub name: &'static str,
    pub kind: OptionKind,
    pub default: f64,
    pub support: OptionSupport,
}
pub const OPTION_TABLE: &[OptionInfo] = &[
    OptionInfo {
        name: "RCSPstopCutGenTimeThresholdInPricing",
        kind: OptionKind::Double,
        default: 10.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs"),
    },
    OptionInfo {
        name: "RCSPhardTimeThresholdInPricing",
        kind: OptionKind::Double,
        default: 20.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; src/pricing/solve.rs; src/master/tree/mod.rs"),
    },
    OptionInfo {
        name: "RCSPredCostFixingTimeThreshold",
        kind: OptionKind::Double,
        default: 100.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs; src/master/root/controller.rs"),
    },
    OptionInfo {
        name: "RCSPnumberOfBucketsPerVertex",
        kind: OptionKind::Int,
        default: 25.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; src/api/options.rs"),
    },
    OptionInfo {
        name: "RCSPdynamicBucketSteps",
        kind: OptionKind::Int,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs; src/pricing/solve.rs"),
    },
    OptionInfo {
        name: "RCSPuseBidirectionalSearch",
        kind: OptionKind::Int,
        default: 2.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; src/api/options.rs"),
    },
    OptionInfo {
        name: "RCSPapplyReducedCostFixing",
        kind: OptionKind::Int,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfColsPerIteration",
        kind: OptionKind::Int,
        default: 30.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; src/pricing/solve.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfColsPerExactIteration",
        kind: OptionKind::Int,
        default: 150.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfLabelsInEnumeration",
        kind: OptionKind::Int,
        default: 1000000.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs; src/pricing/universe.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfEnumeratedSolutions",
        kind: OptionKind::Int,
        default: 1000000.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs; src/pricing/universe.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfEnumSolutionsForMIP",
        kind: OptionKind::Int,
        default: 10000.0,
        support: if cfg!(all(feature = "clp", feature = "highs")) {
            OptionSupport::Implemented("src/master/root/integer.rs; normal route threshold and normalization of the opt-in end-of-node HiGHS MIP threshold")
        } else { OptionSupport::Unsupported("enumerated node integer solving requires the clp and highs features") },
    },
    OptionInfo {
        name: "RCSPmaxNumOfEnumSolsForEndOfNodeMIP",
        kind: OptionKind::Int,
        default: 0.0,
        support: if cfg!(all(feature = "clp", feature = "highs")) {
            OptionSupport::Implemented("src/master/root/integer.rs; route-count threshold for complete enumerated node HiGHS MIP")
        } else { OptionSupport::Unsupported("enumerated node integer solving requires the clp and highs features") },
    },
    OptionInfo {
        name: "RCSPinitNGneighbourhoodSize",
        kind: OptionKind::Int,
        default: 8.0,
        support: OptionSupport::Implemented("src/api/relaxation.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNGneighbourhoodSize",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/api/relaxation.rs; src/master/root/controller.rs"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsMaxNumRows",
        kind: OptionKind::Int,
        default: 5.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs; src/cuts/kpath/mod.rs"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsMemoryType",
        kind: OptionKind::Int,
        default: 2.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsLSnumIterations",
        kind: OptionKind::Int,
        default: 1000.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPallowRoutesWithSameVerticesSet",
        kind: OptionKind::Bool,
        default: 1.0,
        support: OptionSupport::Unsupported("Native route-pool equivalence control is not implemented; node enumeration preserves branch-distinct arc paths"),
    },
    OptionInfo {
        name: "RCSPredCostFixingFalseGap",
        kind: OptionKind::Double,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfLabelsInHeurEnumeration",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs"),
    },
    OptionInfo {
        name: "RCSPuseMetaSolver",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Unsupported("The native MetaSolver is not implemented"),
    },
    OptionInfo {
        name: "RCSPprintLevel",
        kind: OptionKind::Int,
        default: -2.0,
        support: OptionSupport::Unsupported("Native pricing log-level control is not implemented"),
    },
    OptionInfo {
        name: "RCSPcheckDominInOtherBuckets",
        kind: OptionKind::Bool,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "RCSPuseCompletionBoundsInPricing",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Unsupported("Optional completion-bound pruning in ordinary pricing is not implemented"),
    },
    OptionInfo {
        name: "RCSPdomChecksThresholdInPricing",
        kind: OptionKind::Double,
        default: 1000000000000000.0,
        support: OptionSupport::Unsupported("The native dominance-check stopping threshold is not implemented"),
    },
    OptionInfo {
        name: "RCSPheurLabelingStrategy",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "RCSPlabelSplitStrategy",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Unsupported("The native configurable label-split strategy is not implemented"),
    },
    OptionInfo {
        name: "RCSPlabelDominanceStrategy",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; label dominance mode (0 or 1), shared with the C strategy setter"),
    },
    OptionInfo {
        name: "RCSPdynBuckStepAdjustRatioThreshold",
        kind: OptionKind::Double,
        default: 500.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs"),
    },
    OptionInfo {
        name: "RCSPdynBuckStepAdjustNumBuckArcsThreshold",
        kind: OptionKind::Double,
        default: 10000.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs"),
    },
    OptionInfo {
        name: "RCSPuseComplBoundsInRedCostFixing",
        kind: OptionKind::Int,
        default: 2.0,
        support: OptionSupport::Unsupported("Configurable fixing completion-bound modes are not implemented; the current pairing algorithm is fixed"),
    },
    OptionInfo {
        name: "RCSPuseDSSRInMode",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Unsupported("DSSR reduction is not implemented"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfColumnsInDSSR",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Unsupported("DSSR reduction is not implemented"),
    },
    OptionInfo {
        name: "RCSPmaxCycleSizeInDSSR",
        kind: OptionKind::Int,
        default: 5.0,
        support: OptionSupport::Unsupported("DSSR reduction is not implemented"),
    },
    OptionInfo {
        name: "RCSPdynamicNGmode",
        kind: OptionKind::Int,
        default: -1.0,
        support: OptionSupport::Implemented("src/api/relaxation.rs; src/master/root/controller.rs"),
    },
    OptionInfo {
        name: "RCSPprintNGMemories",
        kind: OptionKind::Bool,
        default: 0.0,
        support: OptionSupport::Unsupported("Native NG-memory diagnostic output is not implemented"),
    },
    OptionInfo {
        name: "RCSPmaxNumOfColumnsInDynamicNG",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/api/relaxation.rs; src/master/root/controller.rs"),
    },
    OptionInfo {
        name: "RCSPmaxCycleSizeInDynamicNG",
        kind: OptionKind::Int,
        default: 5.0,
        support: OptionSupport::Implemented("src/api/relaxation.rs; src/master/root/controller.rs"),
    },
    OptionInfo {
        name: "RCSPmaxNGaverNeighbourhoodSize",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/api/relaxation.rs; src/master/root/controller.rs"),
    },
    OptionInfo {
        name: "RCSPuseCapacityCuts",
        kind: OptionKind::Bool,
        default: 1.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; explicit option overrides a configured capacity separator and is inherited by children"),
    },
    OptionInfo {
        name: "RCSPcapacityCutsSeparator",
        kind: OptionKind::Int,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; accepts the available native separator; external CVRPSEP (0) is rejected"),
    },
    OptionInfo {
        name: "RCSPcapCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; explicit per-round override for configured capacity separation, including inherited nodes"),
    },
    OptionInfo {
        name: "RCSPresConsKnapsackCutsMode",
        kind: OptionKind::Int,
        default: -1.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; src/cuts/knapsack/mod.rs"),
    },
    OptionInfo {
        name: "RCSPresConsKnapsackCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; src/cuts/knapsack/mod.rs"),
    },
    OptionInfo {
        name: "RCSPcliqueCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Unsupported("Clique-cut separation is not implemented"),
    },
    OptionInfo {
        name: "RCSPcliqueCutsMaxSepTime",
        kind: OptionKind::Double,
        default: 5.0,
        support: OptionSupport::Unsupported("Clique-cut separation is not implemented"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsArcMemoryType",
        kind: OptionKind::Int,
        default: 2.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsSpDependentMemory",
        kind: OptionKind::Bool,
        default: 0.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsNeighbourhoodSize",
        kind: OptionKind::Int,
        default: 16.0,
        support: OptionSupport::Unsupported("The routing cut adapter has no packing-distance input; native distance-neighbourhood construction is not exposed"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsMaxNumTriplets",
        kind: OptionKind::Int,
        default: 10000.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPrankOneCutsTypeToSeparate",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPuseCovSetsForSepOneRowPackingCuts",
        kind: OptionKind::Bool,
        default: 0.0,
        support: OptionSupport::Implemented("src/cuts/registry.rs"),
    },
    OptionInfo {
        name: "RCSPuseExactComplBoundsInEnumeration",
        kind: OptionKind::Bool,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/universe.rs"),
    },
    OptionInfo {
        name: "RCSPmaxGapToRunEnumeration",
        kind: OptionKind::Double,
        default: 1.0,
        support: OptionSupport::Unsupported("The native enumeration-gap trigger is not implemented; the root driver currently uses its fixed relative-gap rule"),
    },
    OptionInfo {
        name: "RCSPuseVertexIdsInEnumeration",
        kind: OptionKind::Bool,
        default: 0.0,
        support: OptionSupport::Unsupported("Configurable vertex-ID enumeration identity is not implemented"),
    },
    OptionInfo {
        name: "RCSPstopRatioForConcatenationInEnum",
        kind: OptionKind::Double,
        default: 30.0,
        support: OptionSupport::Unsupported("The native enumeration concatenation stop-ratio policy is not implemented"),
    },
];
#[derive(Clone, Debug)]
pub struct Options {
    values: Vec<f64>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            values: OPTION_TABLE.iter().map(|o| o.default).collect(),
        }
    }
}
impl Options {
    pub fn index(name: &str) -> Result<usize> {
        OPTION_TABLE
            .iter()
            .position(|o| o.name == name)
            .ok_or_else(|| invalid(format!("unknown RCSP option: {name}")))
    }
    pub fn get(&self, name: &str) -> Result<f64> {
        Ok(self.values[Self::index(name)?])
    }
    pub fn set(&mut self, name: &str, value: f64) -> Result<()> {
        let i = Self::index(name)?;
        match OPTION_TABLE[i].kind {
            OptionKind::Bool if value != 0. && value != 1. => {
                return Err(invalid("boolean option must be 0 or 1"))
            }
            OptionKind::Int
                if !value.is_finite()
                    || value.fract() != 0.
                    || value < i32::MIN as f64
                    || value > i32::MAX as f64 =>
            {
                return Err(invalid("finite integer option required"))
            }
            OptionKind::Double if value.is_nan() => return Err(invalid("option value is NaN")),
            _ => (),
        }
        if let OptionSupport::Unsupported(reason) = OPTION_TABLE[i].support {
            if value != OPTION_TABLE[i].default {
                return Err(invalid(format!("unsupported RCSP option {name}: {reason}; only the recovered default {} is accepted", OPTION_TABLE[i].default)));
            }
        }
        self.values[i] = value;
        Ok(())
    }
    /// Support and consumer/reason metadata for every recovered option.
    pub fn support(name: &str) -> Result<OptionSupport> {
        Ok(OPTION_TABLE[Self::index(name)?].support)
    }
    pub fn default_of(name: &str) -> Result<f64> {
        Ok(OPTION_TABLE[Self::index(name)?].default)
    }
}
