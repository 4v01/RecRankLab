"""Assemble the model registry after core definitions are complete."""
from .core import MODELS, Recommender, create
from .scalable import BiasBaseline, SGDMF, MFDCN, SparseEASE, HistoryTower, SparseAutoRec
MODELS.update(bias_baseline=BiasBaseline, sgdmf=SGDMF, mf_dcn=MFDCN,
              sparse_ease=SparseEASE, history_tower=HistoryTower, sparse_autorec=SparseAutoRec)
