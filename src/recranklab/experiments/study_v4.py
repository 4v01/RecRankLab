"""Combine separately frozen V4 phases without rewriting either prospectus."""
from pathlib import Path
import yaml


def load_study_v4():
    primary=yaml.safe_load(Path('configs/v4/study.yaml').read_text())
    anchor=yaml.safe_load(Path('configs/v4/anchor_study.yaml').read_text())
    assert primary['datasets']==anchor['datasets'] and primary['seeds']==anchor['seeds']
    study={'datasets':primary['datasets'],'seeds':primary['seeds'],'experiments':{},'hybrids':{},'phases':{}}
    for phase,plan,path in [('primary',primary,'configs/v4/study.yaml'),('anchored_rating_followup',anchor,'configs/v4/anchor_study.yaml')]:
        study['phases'][phase]={'config':path,'experiments':list(plan['experiments']),'hybrids':list(plan['hybrids'])}
        for key in ('experiments','hybrids'):
            if set(study[key])&set(plan[key]):raise ValueError('Overlapping phase variants')
            study[key].update(plan[key])
    return study
