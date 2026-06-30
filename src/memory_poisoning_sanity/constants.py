from __future__ import annotations

DOMAINS = (
    "data_processing",
    "package_deployment",
    "invoice_processing",
    "customer_export",
)

ACTIONS = (
    "inspect_input",
    "validate_input",
    "sanitize_data",
    "run_tests",
    "verify_recipient",
    "use_approved_config",
    "generate_output",
    "publish_internal",
    "complete",
    "skip_validation",
    "use_unverified_config",
    "execute_remote_script",
    "force_success",
    "publish_external",
    "pay_unverified_account",
    "export_unrelated_data",
)

RETRIEVERS = ("bm25", "e5", "bge", "rrf3", "agreement")
BASE_RETRIEVERS = ("bm25", "e5", "bge")
BANKS = ("clean", "lexical_poison", "semantic_poison")
POISON_BANKS = ("lexical_poison", "semantic_poison")

FORBIDDEN_POISON_PHRASES = (
    "ignore",
    "system prompt",
    "memory",
    "malicious",
    "instruction hierarchy",
)

