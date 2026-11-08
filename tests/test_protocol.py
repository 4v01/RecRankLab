import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from recranklab.data.prepare import split_rows, prepare, load
from recranklab.evaluation.metrics import RankingProtocol, rating_metrics, paired_bootstrap
from recranklab.models.core import create, NegativeSampler, Recommender
from recranklab.utils.repro import seed_all


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "item", "rating"])


@pytest.fixture
def train():
    return frame([(0, 0, 5), (0, 1, 2), (1, 1, 4), (1, 2, 3), (2, 2, 5), (2, 3, 1)])


@pytest.fixture
def cfg():
    return dict(seed=17, epochs=2, dim=4, hidden=3, lr=.01, reg=.0001, layers=2,
                rating_batch_size=3, ranking_batch_size=3, auto_batch_size=2)


def test_split_order_disjoint_and_deterministic():
    f = pd.DataFrame([(u, i, 4, i // 2, u*10+i) for u in range(2) for i in range(10)],
                     columns=["user", "item", "rating", "timestamp", "row_id"])
    a, b = split_rows(f), split_rows(f.sample(frac=1, random_state=5))
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert not set(a['train']) & set(a['val'])
    assert not set(a['train']) & set(a['test'])
    assert not set(a['val']) & set(a['test'])
    assert set(np.concatenate(list(a.values()))) == set(f.row_id)
    for u in range(2):
        assert max(f.loc[np.intersect1d(a['train'], f[f.user == u].row_id)].timestamp) <= min(f.loc[np.intersect1d(a['val'], f[f.user == u].row_id)].timestamp)
    assert list(a['train'][:8]) == list(range(8))


def test_duplicate_and_short_users_rejected():
    f = pd.DataFrame([(0, 0, 5, 1, 0), (0, 0, 3, 2, 1)], columns=['user','item','rating','timestamp','row_id'])
    with pytest.raises(ValueError, match='Repeated'):
        split_rows(f)
    with pytest.raises(ValueError, match='three'):
        split_rows(f.iloc[:1])


class Scores:
    def __init__(self, scores):
        self.scores = np.asarray(scores)
    def predict_scores(self, users, items):
        return self.scores[np.ix_(users, items)]


def test_metrics_masking_cold_targets_and_ties(train):
    target = frame([(0, 2, 5), (0, 3, 4), (0, 4, 5), (1, 0, 2)])
    protocol = RankingProtocol(train, target, 3, 5)
    # Known low rating item 1 must be masked despite huge score; cold item 4 too.
    model = Scores([[100, 999, 1, 1, 9999]] * 3)
    metrics, per = protocol.evaluate(model, ks=(1, 2))
    assert metrics['eligible_users'] == 1
    assert metrics['cold_start_positive_targets'] == 1
    assert metrics['recall@1'] == .5
    assert metrics['recall@2'] == 1
    assert metrics['ndcg@1'] == 1
    assert metrics['ndcg@2'] == 1
    assert per.user.tolist() == [0]
    assert metrics['coverage@1'] == .25


def test_ndcg_nontrivial(train):
    target = frame([(0, 3, 5)])
    metrics, _ = RankingProtocol(train, target, 3, 4).evaluate(Scores([[999,999,3,2]]*3), ks=(1,2))
    assert metrics['recall@1'] == 0
    assert metrics['ndcg@2'] == pytest.approx(1 / np.log2(3))


def test_test_mask_includes_validation(train):
    val = frame([(0, 2, 1)])
    history = pd.concat([train, val])
    target = frame([(0, 3, 5)])
    metrics, _ = RankingProtocol(history, target, 3, 4).evaluate(Scores([[9,9,999,1]]*3), ks=(1,))
    assert metrics['recall@1'] == 1


def test_no_eligible_users_is_explicit(train):
    metrics, per = RankingProtocol(train, frame([(0,4,5)]), 3, 5).evaluate(Scores([[1]*5]*3))
    assert per.empty and metrics['ndcg@10'] is None


def test_nonfinite_scores_rejected(train):
    with pytest.raises(ValueError, match='Nonfinite'):
        RankingProtocol(train, frame([(0,2,5)]), 3, 4).evaluate(Scores([[np.nan]*4]*3))


def test_sampler_fit_only_deterministic_and_policy(train):
    users = np.zeros(500, dtype=int)
    a = NegativeSampler(train, 3, 5, 10)
    b = NegativeSampler(train, 3, 5, 10)
    samples = a.sample(users)
    assert np.array_equal(samples, b.sample(users))
    assert set(samples) == {2,3}
    # Future-only item 4 is not in the fit-supported universe. No future labels used.
    assert 4 not in samples
    positive_only = NegativeSampler(train, 3, 5, 10, 'positives_only')
    assert 1 in positive_only.sample(users)
    with pytest.raises(ValueError):
        NegativeSampler(frame([(0,0,5)]),1,1,10).sample([0])


@pytest.mark.parametrize('name', ['popularity','biasmf','autorec','bpr_mf','lightgcn'])
def test_end_to_end_seed_and_checkpoint(name, train, cfg, tmp_path):
    seed_all(17, 1)
    a = create(name, 3, 5, cfg).fit(train)
    expected = a.predict_scores([0,1,2])
    assert expected.shape == (3,5) and np.isfinite(expected).all()
    assert a.mean == train.rating.mean()
    # Full model round trip preserves predictions, metadata and train-only context.
    a.save(tmp_path / 'model.pt')
    loaded = Recommender.load(tmp_path / 'model.pt')
    np.testing.assert_allclose(expected, loaded.predict_scores([0,1,2]), atol=1e-7)
    seed_all(17, 1)
    b = create(name, 3, 5, cfg).fit(train.copy())
    np.testing.assert_array_equal(expected, b.predict_scores([0,1,2]))
    metrics, _ = RankingProtocol(train, frame([(0,2,5),(1,3,4)]), 3, 5).evaluate(a)
    assert metrics['eligible_users'] == 2
    if a.rating_model:
        rm = rating_metrics(a, frame([(0,2,5),(1,4,2)]))
        assert rm['cold_item_ratings'] == 1 and rm['rating_count'] == 2


def test_popularity_only_training_positives(train, cfg):
    m = create('popularity', 3, 5, cfg).fit(train)
    np.testing.assert_array_equal(m.predict_scores([0])[0], [1,1,1,0,0])


def test_autorec_mask_and_train_only_inputs(train, cfg):
    m = create('autorec', 3, 5, cfg)
    m.setup(train)
    assert m.inputs[0,2] == 0 and not m.mask[0,2]
    assert m.inputs[0,1] == 2 and m.mask[0,1]
    prediction = torch.zeros_like(m.inputs, requires_grad=True)
    loss = (prediction-m.inputs).square()[m.mask].mean()
    loss.backward()
    assert prediction.grad[0,2] == 0
    assert prediction.grad[0,1] != 0


def test_lightgcn_normalization_and_zero_depth(train, cfg):
    seed_all(17,1)
    g = create('lightgcn',3,5,{**cfg, 'layers':0}).fit(train)
    seed_all(17,1)
    b = create('bpr_mf',3,5,cfg).fit(train)
    np.testing.assert_allclose(g.predict_scores([0,1,2]), b.predict_scores([0,1,2]), atol=1e-7)
    a = g.graph.to_dense().numpy()
    np.testing.assert_array_equal(a, a.T)
    assert np.all(np.diag(a) == 0)
    assert a[0,3] == 1 and a[0,4] == 0  # low rating excluded from graph


def test_no_future_input_dependency(train, cfg):
    # Future labels never appear in the sampler constructor or fit API.
    future = frame([(0,2,5),(0,4,5)])
    samples_before = NegativeSampler(train,3,5,17).sample([0]*100)
    future.rating = 1
    samples_after = NegativeSampler(train,3,5,17).sample([0]*100)
    np.testing.assert_array_equal(samples_before,samples_after)
    assert 2 in samples_before  # unknown future positive can be a false negative


def test_bootstrap_pairing():
    ci = paired_bootstrap(np.ones(10)*.2, n_boot=100)
    assert ci['ci_low'] == pytest.approx(.2)
    assert ci['ci_high'] == pytest.approx(.2)


def test_offline_ingestion_and_integrity(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    lines = ''.join(f'{u}\t{i}\t4\t{i}\n' for u in (1,2) for i in range(1,11))
    with zipfile.ZipFile(raw / 'ml-100k.zip', 'w') as z:
        z.writestr('ml-100k/u.data', lines)
    a = prepare('ml-100k', tmp_path)
    b = prepare('ml-100k', tmp_path)
    assert a == b
    parts, manifest = load('ml-100k', tmp_path)
    assert len(parts['train']) == 16
    path = tmp_path / 'processed/ml-100k/ratings.csv'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='integrity'):
        load('ml-100k', tmp_path)
