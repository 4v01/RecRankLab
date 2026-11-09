"""A synthetic dtype microprofile, not recommendation accuracy or training time."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from recranklab.utils.repro import seed_all,provenance

p=argparse.ArgumentParser();p.add_argument('--output',default='reports/v4/ease_dtype_microprofile.json');a=p.parse_args()
seed_all(17,2);environment=provenance();rng=np.random.default_rng(17)
x=sparse.random(128,4096,density=.02,format='csr',random_state=17,dtype=np.float64)
b=rng.standard_normal((4096,4096),dtype=np.float32);cached=b.astype(np.float64)
uncached=x@b;reference=x@cached;np.testing.assert_array_equal(uncached,reference)
data_hash=hashlib.sha256()
for value in (x.indptr,x.indices,x.data,b):data_hash.update(value.tobytes())
times={}
for name,matrix in [('implicit_repeated_upcast',b),('cached_float64_view',cached)]:
    start=time.perf_counter()
    for _ in range(5):values=x@matrix
    times[name]=time.perf_counter()-start
r={'scope':'synthetic dtype-path microprofile; not end-to-end recommendation, accuracy or training speed','environment':environment,'seed':17,'shape':[128,4096,4096],'density':.02,'synthetic_data_sha256':data_hash.hexdigest(),'repeats':5,'identical_outputs':True,'seconds':times,'speed_ratio':times['implicit_repeated_upcast']/times['cached_float64_view'],'persistent_memory_tradeoff':'one float64 view per active SparseEASE instance, O(items squared); excluded from checkpoints'}
Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k] for k in ['identical_outputs','seconds','speed_ratio']},indent=2))
