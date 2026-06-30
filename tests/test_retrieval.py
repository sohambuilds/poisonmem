from __future__ import annotations

from memory_poisoning_sanity.retrieval import reciprocal_rank_fusion


def test_rrf_uses_top_twelve_candidates_and_deterministic_ties():
    rankings = {
        "bm25": list(range(20)),
        "e5": list(range(1, 20)) + [0],
        "bge": list(range(2, 20)) + [0, 1],
    }
    ordered, scores, votes = reciprocal_rank_fusion(rankings, candidate_k=12, constant=60)
    assert 19 not in scores
    assert votes[2] == ["bm25", "e5", "bge"]
    assert ordered[0] == 2
    assert scores[2] == 1 / 63 + 1 / 62 + 1 / 61


def test_rrf_vote_count_supports_agreement_filter():
    rankings = {
        "bm25": [0, 1, 2, 3],
        "e5": [4, 0, 5, 6],
        "bge": [7, 8, 0, 9],
    }
    ordered, _, votes = reciprocal_rank_fusion(rankings, candidate_k=3, constant=60)
    accepted = [index for index in ordered if len(votes[index]) >= 2]
    assert accepted == [0]

