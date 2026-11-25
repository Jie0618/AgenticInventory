"""Original GBS feedback rule on the benchmark's periodic decision epochs.

Stolyar and Wang (2022), section 3.2, eqs. (4)-(5) and ordering step 3.
Use the explicitly permitted practical version without the analytical upper
target truncation (p.9, third remark). Y=x-B, Z=all pending orders,
X*=mean(D)*L; order ceil([X* - gamma*(Y-xstar) - Z]+).
The paper has continuous review, random independent lead times, permanent
backlogging and no caps. Here it is a related-model comparator, not a native
finite-K method and not covered by the paper's performance guarantee.
"""
import numpy as np

from optimization import ParameterSpec

BACKEND = 'ltb_gbs_candidate_x_path_batch'


def definition(case):
    bound = (case.lead_time + 1) * case.n
    return dict(id='gbs', type='baseline', execution='gbs', method_class='SOF',
                provenance='published_policy_related_model_transfer',
                reference='Stolyar and Wang (2022), DOI 10.1287/opre.2021.2129, sec.3.2 pp.8-9',
                formula='u=clip(ceil([mean_D*L-gamma*(x-B-xstar)-pipeline]^+),0,n)',
                parameter_specs=[vars(ParameterSpec('gamma', kind='continuous', lower=.1, upper=5., default=1.)),
                                 vars(ParameterSpec('xstar', kind='continuous', lower=-bound, upper=bound, default=0.))])


def actions_many(states, params_batch, case):
    gamma = np.asarray([p['gamma'] for p in params_batch])[:, None]
    target = np.asarray([p['xstar'] for p in params_batch])[:, None]
    l = case.lead_time
    net_inventory = states[:, :, 0] - states[:, :, 1+l:].sum(axis=2)
    pending = states[:, :, 1:1+l].sum(axis=2)
    mean_pipeline = float(case.demand.values @ case.demand.probs) * l
    gap = mean_pipeline - gamma * (net_inventory - target) - pending
    return np.clip(np.ceil(np.maximum(gap, 0)), 0, case.n).astype(np.int32)
