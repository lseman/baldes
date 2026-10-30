"""Native common-start synchronization through the Python interface."""
from dataclasses import replace

import pytest
from pybaldes import (NetworkProblem, PricingArc, PricingNetwork, PricingResource,
                      SynchronizedService, SynchronizedServiceMember, has_clp)

pytestmark = pytest.mark.skipif(not has_clp, reason="requires Clp")


def model(a=(0, 9), b=(5, 10)):
    return NetworkProblem((), tuple(
        PricingNetwork(n, 3, (0, 1), (PricingArc(0, 1, 0), PricingArc(1, 2, 0)),
                       resources=(PricingResource(initial=0, bounds=((0, 0), w, (0, 100))),))
        for n, w in ((7, a), (42, b))))


def service(start=(0, 10)):
    return SynchronizedService(17, 0, 2, 1, start,
                              (SynchronizedServiceMember(7, 1), SynchronizedServiceMember(42, 1)))


@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
@pytest.mark.parametrize("copies", [1, 2])
def test_joint_schedule_and_one_reward(bound_mode, copies):
    problem = model()
    problem = replace(problem, networks=(
        replace(problem.networks[0], multiplicity=(copies, copies)),
        problem.networks[1],
    ))
    synchronized = replace(service(), vehicles_required=copies + 1)
    result = problem.solve(tree=True, bound_mode=bound_mode,
                           synchronized_services=(synchronized,))
    assert result.status == "Optimal"
    assert result.incumbent == result.bound == -1
    assert len(result.routes) == 2
    assert next(r for r in result.routes if r.network == 7).value == copies
    assert all(r.resource_profile[1][0] == 5 for r in result.routes)


def test_incompatible_windows_are_not_an_integer_witness():
    result = model((0, 4), (6, 10)).solve(tree=True, synchronized_services=(service(),))
    assert result.status == "Optimal"
    assert result.incumbent == 0
    assert not result.routes


def test_snapshot_domains_are_immutable_and_cannot_change_definitions():
    root = model().solve(synchronized_services=(service(),))
    child = root.snapshot.with_synchronized_start(17, 8, 9)
    result = child.solve(tree=True)
    assert result.incumbent == -1
    assert all(r.resource_profile[1][0] == 8 for r in result.routes)
    parent = root.snapshot.solve(tree=True)
    assert all(r.resource_profile[1][0] == 5 for r in parent.routes)
    with pytest.raises(ValueError, match="cannot be changed"):
        child.solve(synchronized_services=(service((0, 8)),))


def test_invalid_resource_and_integer_domain_are_rejected():
    with pytest.raises(ValueError, match="integer start domain"):
        model().solve(synchronized_services=(service((9, 8)),))
    with pytest.raises(ValueError):
        model().solve(synchronized_services=(service((0.5, 8)),))
