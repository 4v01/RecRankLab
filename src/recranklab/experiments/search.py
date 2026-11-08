"""Bounded validation-only research memory. No evaluator or dataset access here."""
import math


def propose(config, memory):
    """Return a permitted hypothesis from config + validation records only.

    EASE's positive ridge is the only mutable parameter. Two anchors, then
    multiplicative mutations around the best observed parent. Finite budget,
    deterministic ties, no arbitrary code generation or external memory.
    """
    policy = config.get('search', 'grid')
    if policy == 'grid':
        return dict(config['trials'][len(memory)])
    if policy != 'ridge_evolution':
        raise ValueError(f'Unknown search policy: {policy}')
    if config.get('model') != 'ease':
        raise ValueError('ridge_evolution is restricted to EASE')
    for record in memory:
        if any(k not in ('trial_id', 'overrides', 'metrics', 'fit_seconds', 'loss_history', 'hypothesis') for k in record):
            raise ValueError('Memory must contain validation records only')
        if 'test_metrics' in record['metrics']:
            raise ValueError('Test feedback is forbidden')
    if len(memory) < 2:
        return {'ridge': float(config['anchors'][len(memory)])}
    parent = max(memory, key=lambda r: r['metrics']['ndcg@10'])
    tried = {round(r['overrides']['ridge'], 8) for r in memory}
    # Try shrink then expand around the current elite; fall back to fixed anchors.
    for value in [parent['overrides']['ridge'] / 3, parent['overrides']['ridge'] * 3,
                  10., 30., 300., 3000., 10000.]:
        value = round(value, 8)
        if value not in tried and 1 <= value <= 10000 and math.isfinite(value):
            return {'ridge': value}
    raise ValueError('Permitted search space exhausted')


def hypothesis(config, memory, candidate):
    parent = max(memory, key=lambda r: r['metrics']['ndcg@10'])['trial_id'] if memory else None
    return {'policy': config.get('search', 'grid'), 'parent_trial': parent,
            'mechanism': config.get('hypothesis', 'Predeclared finite model comparison'),
            'candidate': candidate, 'feedback_partition': 'validation',
            'budget': config.get('search_budget', len(config.get('trials', [])))}
