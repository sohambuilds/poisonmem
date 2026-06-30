from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .constants import BASE_RETRIEVERS, RETRIEVERS
from .models import Memory, Query, RetrievalConfig


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _stable_order(scores: np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]], candidate_k: int = 12, constant: int = 60
) -> tuple[list[int], dict[int, float], dict[int, list[str]]]:
    scores: dict[int, float] = {}
    votes: dict[int, list[str]] = {}
    for retriever, ranking in rankings.items():
        for rank, index in enumerate(ranking[:candidate_k], start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (constant + rank)
            votes.setdefault(index, []).append(retriever)
    ordered = sorted(scores, key=lambda index: (-scores[index], index))
    return ordered, scores, votes


@dataclass(frozen=True)
class RetrievalResult:
    method: str
    memories: list[Memory]
    scores: list[float]
    votes: list[list[str]]


class DenseEncoder:
    def __init__(self, model_name: str, cache_dir: Path, e5: bool = False) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.e5 = e5
        self.model = SentenceTransformer(model_name, device="cpu")

    def _format(self, texts: list[str], query: bool) -> list[str]:
        if not self.e5:
            return texts
        prefix = "query: " if query else "passage: "
        return [prefix + text for text in texts]

    def encode(self, texts: list[str], *, query: bool, namespace: str) -> np.ndarray:
        formatted = self._format(texts, query)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"model": self.model_name, "query": query, "texts": formatted},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"{namespace}-{fingerprint[:20]}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        embeddings = self.model.encode(
            formatted,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        return embeddings


class BankRetriever:
    def __init__(
        self,
        memories: list[Memory],
        config: RetrievalConfig,
        e5: DenseEncoder,
        bge: DenseEncoder,
        namespace: str,
    ) -> None:
        self.memories = memories
        self.config = config
        self.e5 = e5
        self.bge = bge
        tokenized = [tokenize(memory.text) for memory in memories]
        self.bm25 = BM25Okapi(tokenized)
        texts = [memory.text for memory in memories]
        self.e5_documents = e5.encode(texts, query=False, namespace=f"{namespace}-e5-docs")
        self.bge_documents = bge.encode(texts, query=False, namespace=f"{namespace}-bge-docs")

    def retrieve(self, query: Query) -> dict[str, RetrievalResult]:
        query_text = query.instruction
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query_text)))
        e5_query = self.e5.encode([query_text], query=True, namespace=f"query-{query.id}")[0]
        bge_query = self.bge.encode([query_text], query=True, namespace=f"query-{query.id}")[0]
        score_vectors = {
            "bm25": bm25_scores,
            "e5": self.e5_documents @ e5_query,
            "bge": self.bge_documents @ bge_query,
        }
        rankings = {name: _stable_order(scores) for name, scores in score_vectors.items()}
        fused, fused_scores, votes = reciprocal_rank_fusion(
            rankings,
            candidate_k=self.config.candidate_k,
            constant=self.config.rrf_constant,
        )

        results: dict[str, RetrievalResult] = {}
        for method in BASE_RETRIEVERS:
            indexes = rankings[method][: self.config.final_k]
            results[method] = RetrievalResult(
                method=method,
                memories=[self.memories[index] for index in indexes],
                scores=[float(score_vectors[method][index]) for index in indexes],
                votes=[votes.get(index, []) for index in indexes],
            )

        fused_indexes = fused[: self.config.final_k]
        results["rrf3"] = RetrievalResult(
            method="rrf3",
            memories=[self.memories[index] for index in fused_indexes],
            scores=[fused_scores[index] for index in fused_indexes],
            votes=[votes[index] for index in fused_indexes],
        )
        agreement_indexes = [index for index in fused if len(votes[index]) >= 2][
            : self.config.final_k
        ]
        results["agreement"] = RetrievalResult(
            method="agreement",
            memories=[self.memories[index] for index in agreement_indexes],
            scores=[fused_scores[index] for index in agreement_indexes],
            votes=[votes[index] for index in agreement_indexes],
        )
        assert set(results) == set(RETRIEVERS)
        assert all(len(result.memories) <= self.config.final_k for result in results.values())
        return results


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def retrieval_input_fingerprint(
    banks: dict[str, list[Memory]], queries: list[Query], config: RetrievalConfig
) -> str:
    payload = {
        "banks": {
            bank: [memory.model_dump() for memory in memories]
            for bank, memories in sorted(banks.items())
        },
        "queries": [query.model_dump() for query in queries],
        "config": config.model_dump(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def result_to_row(
    query: Query,
    bank: str,
    result: RetrievalResult,
    pairwise_jaccard: dict[str, float],
) -> dict[str, Any]:
    poison_ranks = [
        rank for rank, memory in enumerate(result.memories, start=1) if memory.is_poison
    ]
    relevant_clean = sum(
        not memory.is_poison and memory.domain == query.domain for memory in result.memories
    )
    denominator = min(4, 20)
    return {
        "query_id": query.id,
        "query_split": query.split,
        "query_domain": query.domain,
        "bank": bank,
        "retriever": result.method,
        "memory_ids": [memory.id for memory in result.memories],
        "memory_domains": [memory.domain for memory in result.memories],
        "poison_flags": [memory.is_poison for memory in result.memories],
        "scores": result.scores,
        "retriever_votes": result.votes,
        "returned_count": len(result.memories),
        "exposure_at_4": bool(poison_ranks),
        "prp_at_4": len(poison_ranks) / len(result.memories) if result.memories else 0.0,
        "highest_poison_rank": min(poison_ranks) if poison_ranks else None,
        "clean_relevant_recall_at_4": relevant_clean / denominator,
        "pairwise_jaccard": json.dumps(pairwise_jaccard, sort_keys=True),
    }


def run_retrieval_analysis(
    banks: dict[str, list[Memory]],
    queries: list[Query],
    config: RetrievalConfig,
    cache_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    input_fingerprint = retrieval_input_fingerprint(banks, queries, config)
    e5 = DenseEncoder(config.e5_model, cache_dir, e5=True)
    bge = DenseEncoder(config.bge_model, cache_dir, e5=False)
    rows: list[dict[str, Any]] = []
    for bank, memories in banks.items():
        bank_retriever = BankRetriever(memories, config, e5, bge, namespace=bank)
        for query in queries:
            results = bank_retriever.retrieve(query)
            overlaps = {
                f"{left}_{right}": jaccard(
                    {memory.id for memory in results[left].memories},
                    {memory.id for memory in results[right].memories},
                )
                for left, right in combinations(BASE_RETRIEVERS, 2)
            }
            rows.extend(
                result_to_row(query, bank, result, overlaps) for result in results.values()
            )
    frame = pd.DataFrame(rows).sort_values(["bank", "query_id", "retriever"])
    frame["input_fingerprint"] = input_fingerprint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def load_retrieved_memories(
    retrieval_frame: pd.DataFrame,
    banks: dict[str, list[Memory]],
    query_id: str,
    bank: str,
    retriever: str,
) -> list[Memory]:
    match = retrieval_frame[
        (retrieval_frame["query_id"] == query_id)
        & (retrieval_frame["bank"] == bank)
        & (retrieval_frame["retriever"] == retriever)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one retrieval row for {query_id}/{bank}/{retriever}, got {len(match)}"
        )
    ids = list(match.iloc[0]["memory_ids"])
    lookup = {memory.id: memory for memory in banks[bank]}
    return [lookup[memory_id] for memory_id in ids]
