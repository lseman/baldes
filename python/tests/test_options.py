"""Supported option discovery and canonical/legacy setter interoperability."""
import pytest
from pybaldes import Solver


def test_option_discovery_is_supported_canonical_and_feature_specific():
    names = Solver.option_names()
    assert len(names) in (40, 42)
    assert len(names) == len(set(names))
    assert all(name.startswith("BALDES") for name in names)
    assert all(Solver.option_support(name)[0] for name in names)
    mip = "BALDESmaxNumOfEnumSolutionsForMIP"
    if mip in names:
        assert Solver.option_support(mip)[0]
    else:
        with pytest.raises(ValueError, match="unknown BALDES option"):
            Solver.option_support(mip)


def test_legacy_inputs_share_values_and_validation_with_canonical_names():
    solver = Solver(2, 0)
    solver.set_option("RCSPlabelDominanceStrategy", 1)
    assert solver.get_option("BALDESlabelDominanceStrategy") == 1
    with pytest.raises(RuntimeError, match="unsupported label dominance strategy"):
        solver.set_option("RCSPlabelDominanceStrategy", 2)
    assert solver.get_option("BALDESlabelDominanceStrategy") == 1
    solver.set_option("BALDESlabelDominanceStrategy", 0)
    assert solver.get_option("RCSPlabelDominanceStrategy") == 0


def test_unregistered_options_reject_even_zero_defaults():
    solver = Solver(2, 0)
    with pytest.raises(ValueError, match="unknown BALDES option"):
        Solver.option_support("BALDESnotARealOption")
    with pytest.raises(RuntimeError, match="unknown BALDES option"):
        solver.set_option("BALDESnotARealOption", 0)
    with pytest.raises(RuntimeError, match="unknown BALDES option"):
        solver.get_option("BALDESnotARealOption")
