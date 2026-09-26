import numpy as np
import pytest
from scipy.sparse import csr_matrix
from streamer_recommender import HybridStreamerRecommender as Recommender


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("diversity,expected", [(0, [0, 1]), (0.4, [0, 2])])
def test_diversity(sparse, diversity, expected):
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    if sparse:
        embeddings = csr_matrix(embeddings)
    result = Recommender._rerank_for_diversity(
        np.array([0, 1, 2]), np.array([1.0, 0.9, 0.8, 10.0]), embeddings, 2, diversity
    )
    assert result == expected
    assert len(result) == len(set(result))
    assert set(result) <= {0, 1, 2}


@pytest.mark.parametrize("candidates,expected", [([], []), ([2], [2])])
def test_small_candidate_sets(candidates, expected):
    assert (
        Recommender._rerank_for_diversity(
            np.array(candidates, dtype=int), np.ones(3), np.eye(3), 10, 0.2
        )
        == expected
    )
