import json
from pathlib import Path
import pandas as pd
from recranklab.data.prepare import load
from recranklab.evaluation.metrics import RankingProtocol

parts,manifest=load('ml-1m')
nu,ni=manifest['n_users'],manifest['n_items']
stages={}
for stage,fit,target in [('validation',parts['train'],parts['val']),('test',pd.concat([parts['train'],parts['val']]),parts['test'])]:
 positive=fit[fit.rating>=4]
 stages[stage]={'fit_rows':len(fit),'fit_positives':len(positive),'fit_users':fit.user.nunique(),
                'fit_items':fit.item.nunique(),'users_without_positive_fit_support':nu-positive.user.nunique(),
                'items_without_positive_fit_support':ni-positive.item.nunique(),
                **RankingProtocol(fit,target,nu,ni).audit}
result={'dataset':manifest,'catalog_universe':'unique IDs observed in ratings; unrated metadata-only movies are not counted',
        'partition_rating_counts':{k:len(v) for k,v in parts.items()},
        'partition_positive_counts':{k:int((v.rating>=4).sum()) for k,v in parts.items()},'stages':stages}
Path('reports/data_audit.json').write_text(json.dumps(result,indent=2))
