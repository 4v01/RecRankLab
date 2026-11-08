import numpy as np
import pandas as pd
import pytest
import torch

from recranklab.models.core import create, Recommender
from recranklab.experiments.search import propose
from recranklab.utils.repro import seed_all


@pytest.fixture
def rows():
    return pd.DataFrame([(0,0,5),(0,1,2),(1,0,4),(1,2,5),(2,1,5),(2,2,4)],
                        columns=['user','item','rating'])


def test_ease_closed_form_zero_diagonal_and_checkpoint(rows, tmp_path):
    m = create('ease', 3, 4, {'ridge': 10}).fit(rows)
    x = np.array([[1,0,0,0],[1,0,1,0],[0,1,1,0]], dtype=float)
    p = np.linalg.inv(x.T @ x + 10 * np.eye(4))
    b = -p / np.diag(p)[None,:]; np.fill_diagonal(b,0)
    np.testing.assert_allclose(m.predict_scores([0,1,2]), x @ b, atol=1e-7)
    assert torch.all(torch.diag(m.coefficients) == 0)
    # Observed low rating is not a positive; unsupported item has no signal.
    assert m.inputs[0,1] == 0
    assert np.all(m.predict_scores([0,1,2], [3]) == 0)
    m.save(tmp_path/'ease.pt')
    np.testing.assert_array_equal(m.predict_scores([0,1,2]), Recommender.load(tmp_path/'ease.pt').predict_scores([0,1,2]))
    for invalid in [0, -1, float('nan')]:
        with pytest.raises(ValueError, match='ridge'):
            create('ease',3,4,{'ridge': invalid}).fit(rows)


def test_item_autorec_empty_items_denoising_and_reload(rows, tmp_path):
    cfg = dict(seed=17, hidden=3, auto_variant='item', input_dropout=.2,
               auto_batch_size=1, epochs=2, lr=.001, reg=.0001)
    seed_all(17,1)
    m = create('autorec',3,4,cfg).fit(rows)
    assert m.inputs.shape == (4,3) and m.inputs[1,0] == 2
    assert np.isfinite(m.predict_scores([0,1,2])).all()
    m.save(tmp_path/'item.pt')
    np.testing.assert_array_equal(m.predict_scores([0,1,2]), Recommender.load(tmp_path/'item.pt').predict_scores([0,1,2]))
    seed_all(17,1)
    n = create('autorec',3,4,cfg).fit(rows)
    np.testing.assert_array_equal(m.predict_scores([0,1,2]), n.predict_scores([0,1,2]))


def test_search_uses_validation_elite_records_failures_and_no_duplicates():
    cfg = dict(model='ease', search='ridge_evolution', anchors=[100.,1000.])
    memory=[]
    for score in [.2,.1,.3,.05]:
        candidate=propose(cfg,memory)
        assert candidate not in [r['overrides'] for r in memory]
        memory.append(dict(trial_id=len(memory),overrides=candidate,metrics={'ndcg@10':score}))
    assert [r['overrides']['ridge'] for r in memory] == [100.,1000.,33.33333333,11.11111111]
    assert len(memory) == 4  # Failed proposals are retained as research memory.
    memory[0]['test_metrics']={'ndcg@10':1.}
    with pytest.raises(ValueError, match='validation'):
        propose(cfg,memory)
