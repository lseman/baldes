//! Recovered constructor defaults. Stored options do not imply a wired consumer.
use crate::{invalid, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionKind {
    Int,
    Double,
    Bool,
}
#[derive(Clone, Copy, Debug)]
pub struct OptionInfo {
    pub name: &'static str,
    pub kind: OptionKind,
    pub default: f64,
}
pub const OPTION_TABLE: &[OptionInfo] = &[
    OptionInfo {
        name: "RCSPstopCutGenTimeThresholdInPricing",
        kind: OptionKind::Double,
        default: 10.0,
    },
    OptionInfo {
        name: "RCSPhardTimeThresholdInPricing",
        kind: OptionKind::Double,
        default: 20.0,
    },
    OptionInfo {
        name: "RCSPredCostFixingTimeThreshold",
        kind: OptionKind::Double,
        default: 100.0,
    },
    OptionInfo {
        name: "RCSPnumberOfBucketsPerVertex",
        kind: OptionKind::Int,
        default: 25.0,
    },
    OptionInfo {
        name: "RCSPdynamicBucketSteps",
        kind: OptionKind::Int,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPuseBidirectionalSearch",
        kind: OptionKind::Int,
        default: 2.0,
    },
    OptionInfo {
        name: "RCSPapplyReducedCostFixing",
        kind: OptionKind::Int,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfColsPerIteration",
        kind: OptionKind::Int,
        default: 30.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfColsPerExactIteration",
        kind: OptionKind::Int,
        default: 150.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfLabelsInEnumeration",
        kind: OptionKind::Int,
        default: 1000000.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfEnumeratedSolutions",
        kind: OptionKind::Int,
        default: 1000000.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfEnumSolutionsForMIP",
        kind: OptionKind::Int,
        default: 10000.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfEnumSolsForEndOfNodeMIP",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPinitNGneighbourhoodSize",
        kind: OptionKind::Int,
        default: 8.0,
    },
    OptionInfo {
        name: "RCSPmaxNGneighbourhoodSize",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsMaxNumRows",
        kind: OptionKind::Int,
        default: 5.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsMemoryType",
        kind: OptionKind::Int,
        default: 2.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsLSnumIterations",
        kind: OptionKind::Int,
        default: 1000.0,
    },
    OptionInfo {
        name: "RCSPallowRoutesWithSameVerticesSet",
        kind: OptionKind::Bool,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPredCostFixingFalseGap",
        kind: OptionKind::Double,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfLabelsInHeurEnumeration",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPuseMetaSolver",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPprintLevel",
        kind: OptionKind::Int,
        default: -2.0,
    },
    OptionInfo {
        name: "RCSPcheckDominInOtherBuckets",
        kind: OptionKind::Bool,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPuseCompletionBoundsInPricing",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPdomChecksThresholdInPricing",
        kind: OptionKind::Double,
        default: 1000000000000000.0,
    },
    OptionInfo {
        name: "RCSPheurLabelingStrategy",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPlabelSplitStrategy",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPlabelDominanceStrategy",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPdynBuckStepAdjustRatioThreshold",
        kind: OptionKind::Double,
        default: 500.0,
    },
    OptionInfo {
        name: "RCSPdynBuckStepAdjustNumBuckArcsThreshold",
        kind: OptionKind::Double,
        default: 10000.0,
    },
    OptionInfo {
        name: "RCSPuseComplBoundsInRedCostFixing",
        kind: OptionKind::Int,
        default: 2.0,
    },
    OptionInfo {
        name: "RCSPuseDSSRInMode",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfColumnsInDSSR",
        kind: OptionKind::Int,
        default: 100.0,
    },
    OptionInfo {
        name: "RCSPmaxCycleSizeInDSSR",
        kind: OptionKind::Int,
        default: 5.0,
    },
    OptionInfo {
        name: "RCSPdynamicNGmode",
        kind: OptionKind::Int,
        default: -1.0,
    },
    OptionInfo {
        name: "RCSPprintNGMemories",
        kind: OptionKind::Bool,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPmaxNumOfColumnsInDynamicNG",
        kind: OptionKind::Int,
        default: 100.0,
    },
    OptionInfo {
        name: "RCSPmaxCycleSizeInDynamicNG",
        kind: OptionKind::Int,
        default: 5.0,
    },
    OptionInfo {
        name: "RCSPmaxNGaverNeighbourhoodSize",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPuseCapacityCuts",
        kind: OptionKind::Bool,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPcapacityCutsSeparator",
        kind: OptionKind::Int,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPcapCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
    },
    OptionInfo {
        name: "RCSPresConsKnapsackCutsMode",
        kind: OptionKind::Int,
        default: -1.0,
    },
    OptionInfo {
        name: "RCSPresConsKnapsackCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 100.0,
    },
    OptionInfo {
        name: "RCSPcliqueCutsMaxNumPerRound",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPcliqueCutsMaxSepTime",
        kind: OptionKind::Double,
        default: 5.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsArcMemoryType",
        kind: OptionKind::Int,
        default: 2.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsSpDependentMemory",
        kind: OptionKind::Bool,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsNeighbourhoodSize",
        kind: OptionKind::Int,
        default: 16.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsMaxNumTriplets",
        kind: OptionKind::Int,
        default: 10000.0,
    },
    OptionInfo {
        name: "RCSPrankOneCutsTypeToSeparate",
        kind: OptionKind::Int,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPuseCovSetsForSepOneRowPackingCuts",
        kind: OptionKind::Bool,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPuseExactComplBoundsInEnumeration",
        kind: OptionKind::Bool,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPmaxGapToRunEnumeration",
        kind: OptionKind::Double,
        default: 1.0,
    },
    OptionInfo {
        name: "RCSPuseVertexIdsInEnumeration",
        kind: OptionKind::Bool,
        default: 0.0,
    },
    OptionInfo {
        name: "RCSPstopRatioForConcatenationInEnum",
        kind: OptionKind::Double,
        default: 30.0,
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
        self.values[i] = value;
        Ok(())
    }
    pub fn default_of(name: &str) -> Result<f64> {
        Ok(OPTION_TABLE[Self::index(name)?].default)
    }
}
