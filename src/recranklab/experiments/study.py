"""Post-study collection of separately frozen primary and follow-up plans."""
from pathlib import Path
import yaml


def load_study():
    paths=[('primary','configs/v3/study.yaml'),
           ('low_decay_followup','configs/v3/followup_study.yaml'),
           ('svdpp_reference_followup','configs/v3/svdpp_study.yaml')]
    result={'seeds':None,'experiments':{},'comparisons':[],'phases':{},'phase_plans':{}}
    for phase,path in paths:
        plan=yaml.safe_load(Path(path).read_text())
        if result['seeds'] is None:result['seeds']=plan['seeds']
        if result['seeds']!=plan['seeds']:raise ValueError('Phase seed mismatch')
        if set(result['experiments'])&set(plan['experiments']):raise ValueError('Duplicate phase experiment')
        result['experiments'].update(plan['experiments']);result['comparisons']+=plan['comparisons']
        result['phases'][phase]=list(plan['experiments']);result['phase_plans'][phase]={'path':path,'plan':plan}
    return result
