"""Original MTW master/search with BALDES bucket labeling for every pricing call.

Preserves branching, diving, strong branching, dynamic NG growth and schedule
validation. Rank-one cuts and Python bucket-arc elimination are explicitly off:
these do not yet have equivalent bindings in the low-level Python solver.
NG growth is capped at fifteen elements, the native local memory capacity.
"""
from baldes_pricing import GroupPricer
from bp_mtw import BranchCutPrice, Column, Instance, RC_TOL, check_routes


class BaldesBranchPrice(BranchCutPrice):
    def __init__(self, inst, **options):
        options.setdefault('ng_max', 15)
        if options['ng_max'] > 15:
            raise ValueError('BALDES native local NG memory supports ng_max <= 15')
        for name, value in [('root_cut_rounds', 0), ('node_cut_rounds', 0),
                            ('arc_elimination', False), ('parallel', False)]:
            if name in options and options[name] != value:
                raise ValueError(f'{name} must be {value!r} with the BALDES pricing adapter')
            options[name] = value
        super().__init__(inst, **options)
        self.baldes_calls = 0
        self.baldes_pricers = [GroupPricer(inst, g) for g in inst.groups]

    def price(self, node, duals, mu, exact):
        if self.cuts:
            raise RuntimeError('rank-one cuts are not supported by this adapter')
        out = {}
        for group in self.inst.groups:
            gid = group.gid
            pi, resources = self.group_duals(duals, gid)
            if resources['sigma']:
                raise RuntimeError('active cut duals cannot be dropped during BALDES pricing')
            result = self.baldes_pricers[gid].price(
                pi, self.S['groups'][gid]['ng'], mu=mu[gid], exact=exact,
                maximum=self.max_cols_per_group, mode=1 if self.bidirectional else 0,
                removed=node.removed_arcs[gid], banned=node.banned_jobs(gid),
                windows=self.node_windows(gid, node), arc_cost=resources.get('arc_cost'),
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
    result['rank_one_cuts'] = False
    result['python_bucket_arc_elimination'] = False
    return result
