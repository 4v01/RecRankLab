from recranklab.utils.repro import config_load
from recranklab.experiments.run import run_one
cfg=config_load('configs/v4/sparse_autorec.yaml')
cfg.update(experiment='sparse_autorec_engine_smoke_v4',sparse_training=True,dataset='ml-100k',epochs=1,trials=[cfg['trials'][0]],output_root='outputs/v4_engine_smoke')
run_one('sparse_autorec',cfg,17)
