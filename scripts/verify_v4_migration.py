"""Compare numerical definitions across local and hosted execution sources."""
import ast,hashlib,json,subprocess
from pathlib import Path

LOCAL='4e531a0e016f563375794138aec34123244f8b4a'
HOSTED='a2b7dd8c05849a23afd34a0a90c71243cf06789f'

def source(sha,path):return subprocess.check_output(['git','show',sha+':'+path])
def node_hash(sha,path,name):
    tree=ast.parse(source(sha,path).decode());node=next(x for x in tree.body if getattr(x,'name',None)==name)
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def verify():
    checks={}
    for path,names in [('src/recranklab/models/scalable.py',['StreamingMF','SGDMF','MFDCN','SparseEASE','HistoryTower']),('src/recranklab/models/core.py',['BiasMF']),('src/recranklab/experiments/run.py',['run_one']),('src/recranklab/evaluation/metrics.py',['RankingProtocol','rating_metrics','rating_user_errors','stable_top_indices'])]:
        for name in names:
            a,b=node_hash(LOCAL,path,name),node_hash(HOSTED,path,name);assert a==b,(path,name);checks[path+':'+name]=a
    for path in ['src/recranklab/models/sgd_kernel.cpp','src/recranklab/experiments/training.py','src/recranklab/data/prepare.py']:
        a,b=source(LOCAL,path),source(HOSTED,path);assert a==b,path;checks[path]=hashlib.sha256(a).hexdigest()
    return {'scope':'AST-identical numerical definitions and byte-identical kernel/training/data files; storage/provenance changes are separate; not a hardware bitwise guarantee','local_source':LOCAL,'hosted_source':HOSTED,'checks':checks}
if __name__=='__main__':
    result=verify();path=Path('reports/v4/migration_verification.json')
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,indent=2))
    print('PASS: numerical model, fitting, stopping, evaluator and data definitions unchanged across the storage/execution migration.')
