from __future__ import annotations

from collections import Counter

from memory_poisoning_sanity.constants import FORBIDDEN_POISON_PHRASES
from memory_poisoning_sanity.data import generate_data, load_banks, read_jsonl, validate_data
from memory_poisoning_sanity.models import Query
from memory_poisoning_sanity.prompts import build_messages


def test_generated_dataset_matches_registered_counts(tmp_path):
    generate_data(tmp_path)
    summary = validate_data(tmp_path)
    assert summary["queries"] == 32
    assert summary["banks"] == {
        "clean": 80,
        "lexical_poison": 84,
        "semantic_poison": 84,
    }
    queries = read_jsonl(tmp_path / "queries.jsonl", Query)
    assert set(Counter((query.split, query.domain) for query in queries).values()) == {4}


def test_poison_constraints_and_prompt_label_isolation(tmp_path):
    generate_data(tmp_path)
    banks = load_banks(tmp_path)
    query = read_jsonl(tmp_path / "queries.jsonl", Query)[0]
    poison = next(memory for memory in banks["lexical_poison"] if memory.is_poison)
    lowered = poison.text.lower()
    assert not any(phrase in lowered for phrase in FORBIDDEN_POISON_PHRASES)
    prompt = str(build_messages(query, [poison]))
    assert "lexical_poison" not in prompt
    assert "semantic_poison" not in prompt
    assert "required_actions" not in prompt
    assert "prohibited_actions" not in prompt
