//! Supported BALDES pricing options, defaults, and compatibility lookup.
use crate::{invalid, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionKind {
    Int,
    Double,
    Bool,
}
/// Availability of an option's behavior. Implemented options may
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
/// Supported canonical BALDES options available in this build.
/// Conditional entries require both LP-tree and integer-solver features.
pub const OPTION_TABLE: &[OptionInfo] = &[
    OptionInfo {
        name: "BALDESstopCutGenTimeThresholdInPricing",
        kind: OptionKind::Double,
        default: 10.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs"),
    },
    OptionInfo {
        name: "BALDEShardTimeThresholdInPricing",
        kind: OptionKind::Double,
        default: 20.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; src/pricing/solve.rs; src/master/tree/mod.rs"),
    },
    OptionInfo {
        name: "BALDESredCostFixingTimeThreshold",
        kind: OptionKind::Double,
        default: 100.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs; src/master/root/controller/enumeration.rs"),
    },
    OptionInfo {
        name: "BALDESnumberOfBucketsPerVertex",
        kind: OptionKind::Int,
        default: 25.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "BALDESdynamicBucketSteps",
        kind: OptionKind::Int,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs; src/pricing/solve.rs"),
    },
    OptionInfo {
        name: "BALDESuseBidirectionalSearch",
        kind: OptionKind::Int,
        default: 2.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "BALDESapplyReducedCostFixing",
        kind: OptionKind::Int,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNumOfColsPerIteration",
        kind: OptionKind::Int,
        default: 30.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; src/pricing/solve.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNumOfColsPerExactIteration",
        kind: OptionKind::Int,
        default: 150.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNumOfLabelsInEnumeration",
        kind: OptionKind::Int,
        default: 1000000.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs; src/pricing/universe.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNumOfEnumeratedSolutions",
        kind: OptionKind::Int,
        default: 1000000.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs; src/pricing/universe.rs"),
    },
    #[cfg(all(feature = "clp", feature = "highs"))]
    OptionInfo {
        name: "BALDESmaxNumOfEnumSolutionsForMIP",
        kind: OptionKind::Int,
        default: 10000.0,
        support: OptionSupport::Implemented("src/master/root/integer.rs; normal route threshold and normalization of the opt-in end-of-node HiGHS MIP threshold"),
    },
    #[cfg(all(feature = "clp", feature = "highs"))]
    OptionInfo {
        name: "BALDESmaxNumOfEnumSolsForEndOfNodeMIP",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/master/root/integer.rs; route-count threshold for complete enumerated node HiGHS MIP"),
    },
    OptionInfo {
        name: "BALDESinitNGneighbourhoodSize",
        kind: OptionKind::Int,
        default: 8.0,
        support: OptionSupport::Implemented("src/graph/ng.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNGneighbourhoodSize",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/master/root/controller/relaxation.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsMaxNumRows",
        kind: OptionKind::Int,
        default: 5.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs; src/cuts/kpath/mod.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsMaxPerVertex",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsMemoryType",
        kind: OptionKind::Int,
        default: 2.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsLSnumIterations",
        kind: OptionKind::Int,
        default: 1000.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESredCostFixingFalseGap",
        kind: OptionKind::Double,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNumOfLabelsInHeurEnumeration",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/fixing_workflow.rs"),
    },
    OptionInfo {
        name: "BALDEScheckDominInOtherBuckets",
        kind: OptionKind::Bool,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "BALDESheurLabelingStrategy",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs"),
    },
    OptionInfo {
        name: "BALDESlabelDominanceStrategy",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; label dominance mode (0 or 1), shared with the C strategy setter"),
    },
    OptionInfo {
        name: "BALDESdynBuckStepAdjustRatioThreshold",
        kind: OptionKind::Double,
        default: 500.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs"),
    },
    OptionInfo {
        name: "BALDESdynBuckStepAdjustNumBuckArcsThreshold",
        kind: OptionKind::Double,
        default: 10000.0,
        support: OptionSupport::Implemented("src/pricing/controller.rs"),
    },
    OptionInfo {
        name: "BALDESdynamicNGmode",
        kind: OptionKind::Int,
        default: -1.0,
        support: OptionSupport::Implemented("src/master/root/controller/relaxation.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNumOfColumnsInDynamicNG",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/master/root/controller/relaxation.rs"),
    },
    OptionInfo {
        name: "BALDESmaxCycleSizeInDynamicNG",
        kind: OptionKind::Int,
        default: 5.0,
        support: OptionSupport::Implemented("src/master/root/controller/relaxation.rs"),
    },
    OptionInfo {
        name: "BALDESselectionTighteningSchedule",
        kind: OptionKind::Int,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/relaxation.rs"),
    },
    OptionInfo {
        name: "BALDESmaxNGaverNeighbourhoodSize",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/master/root/controller/relaxation.rs"),
    },
    OptionInfo {
        name: "BALDESuseCapacityCuts",
        kind: OptionKind::Bool,
        default: 1.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; explicit option overrides a configured capacity separator and is inherited by children"),
    },
    OptionInfo {
        name: "BALDEScapacityCutsSeparator",
        kind: OptionKind::Int,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/mod.rs; accepts the available native separator; external CVRPSEP (0) is rejected"),
    },
    OptionInfo {
        name: "BALDEScapCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; explicit per-round override for configured capacity separation, including inherited nodes"),
    },
    OptionInfo {
        name: "BALDESresConsKnapsackCutsMode",
        kind: OptionKind::Int,
        default: -1.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; src/cuts/knapsack/mod.rs"),
    },
    OptionInfo {
        name: "BALDESresConsKnapsackCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
        support: OptionSupport::Implemented("src/master/root/mod.rs; src/cuts/knapsack/mod.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsArcMemoryType",
        kind: OptionKind::Int,
        default: 2.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsSpDependentMemory",
        kind: OptionKind::Bool,
        default: 0.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsMaxNumTriplets",
        kind: OptionKind::Int,
        default: 10000.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESrankOneCutsTypeToSeparate",
        kind: OptionKind::Int,
        default: 0.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESuseCovSetsForSepOneRowPackingCuts",
        kind: OptionKind::Bool,
        default: 0.0,
        support: OptionSupport::Implemented("src/cuts/registry/rank_one.rs"),
    },
    OptionInfo {
        name: "BALDESuseExactComplBoundsInEnumeration",
        kind: OptionKind::Bool,
        default: 1.0,
        support: OptionSupport::Implemented("src/pricing/universe.rs"),
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
    fn matches(info: &OptionInfo, name: &str) -> bool {
        info.name == name
            || name
                .strip_prefix("RCSP")
                .is_some_and(|suffix| info.name.strip_prefix("BALDES") == Some(suffix))
    }

    fn definition(name: &str) -> Result<&'static OptionInfo> {
        OPTION_TABLE
            .iter()
            .find(|info| Self::matches(info, name))
            .ok_or_else(|| invalid(format!("unknown BALDES option: {name}")))
    }

    /// Canonical spelling for a known option, accepting the legacy input prefix.
    pub fn canonical_name(name: &str) -> Result<&'static str> {
        Ok(Self::definition(name)?.name)
    }

    /// Position in the supported public table.
    pub fn index(name: &str) -> Result<usize> {
        OPTION_TABLE
            .iter()
            .position(|info| Self::matches(info, name))
            .ok_or_else(|| {
                invalid(format!(
                    "BALDES option is not available in this build: {name}"
                ))
            })
    }

    pub fn get(&self, name: &str) -> Result<f64> {
        let info = Self::definition(name)?;
        Ok(self.values[Self::index(info.name)?])
    }

    pub fn set(&mut self, name: &str, value: f64) -> Result<()> {
        let info = Self::definition(name)?;
        match info.kind {
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
        self.values[Self::index(info.name)?] = value;
        Ok(())
    }

    /// Availability for an explicit option query. Listings use OPTION_TABLE.
    pub fn support(name: &str) -> Result<OptionSupport> {
        Ok(Self::definition(name)?.support)
    }

    pub fn default_of(name: &str) -> Result<f64> {
        Ok(Self::definition(name)?.default)
    }
}
