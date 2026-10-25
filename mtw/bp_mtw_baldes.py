"""Original MTW master/search with BALDES bucket labeling for every pricing call.

Preserves branching, diving, strong branching, dynamic NG growth, limited-memory
rank-1 cuts and schedule validation. Python bucket-arc elimination is explicitly
off: it does not yet have an equivalent binding in the low-level Python solver.
Cuts are separated by the Python master and priced natively with the same arc
memory; pass root_cut_rounds=0 and node_cut_rounds=0 to disable them.
NG growth is capped at the native local memory capacity (15, 31 or 63
elements, selected when pybaldes is compiled).
"""
from baldes_pricing import NG_CAPACITY, GroupPricer, rank_one_cuts
from bp_mtw import BranchCutPrice, Column, Instance, RC_TOL, check_routes


class BaldesBranchPrice(BranchCutPrice):
    def __init__(self, inst, **options):
        options.setdefault('ng_max', min(NG_CAPACITY, 15))
        if options['ng_max'] > NG_CAPACITY:
            raise ValueError(f'BALDES native local NG memory supports ng_max <= {NG_CAPACITY}')
        for name, value in [('arc_elimination', False), ('parallel', False)]:
            if name in options and options[name] != value:
                raise ValueError(f'{name} must be {value!r} with the BALDES pricing adapter')
            options[name] = value
        super().__init__(inst, **options)
        self.baldes_calls = 0
        self.baldes_pricers = [GroupPricer(inst, g) for g in inst.groups]

    def price(self, node, duals, mu, exact):
        out = {}
        cuts = rank_one_cuts(duals[1])
        for group in self.inst.groups:
            gid = group.gid
            pi, resources = self.group_duals(duals, gid)
            result = self.baldes_pricers[gid].price(
                pi, self.S['groups'][gid]['ng'], mu=mu[gid], exact=exact,
                maximum=self.max_cols_per_group, mode=1 if self.bidirectional else 0,
                removed=node.removed_arcs[gid], banned=node.banned_jobs(gid),
                windows=self.node_windows(gid, node), arc_cost=resources.get('arc_cost'),
                cuts=cuts,
            )
            self.baldes_calls += 1
            columns = [Column(gid, path) for cost, path in result.columns if cost < -RC_TOL]
            out[gid] = (columns, -result.minimum if result.minimum is not None else 0.)
        return out


def solve_bcp(jobs, distance_matrix, num_antennas, **options):
    inst = Instance(jobs, distance_matrix, num_antennas,
                    group_antennas=options.pop('group_antennas', True))
    bcp = BaldesBranchPrice(inst, **options)
    result = bcp.solve()
    errors = check_routes(inst, result['routes'])
    if errors:
        raise RuntimeError('invalid solution: ' + '; '.join(errors))
    result['pricing_backend'] = 'pybaldes'
    result['baldes_calls'] = bcp.baldes_calls
    result['baldes_native_calls'] = sum(p.native_calls for p in bcp.baldes_pricers)
    result['baldes_preparations'] = sum(p.preparations for p in bcp.baldes_pricers)
    result['baldes_cached_results'] = sum(p.cached_results for p in bcp.baldes_pricers)
    result['rank_one_cuts'] = bcp.root_cut_rounds > 0 or bcp.node_cut_rounds > 0
    result['python_bucket_arc_elimination'] = False
    return result
