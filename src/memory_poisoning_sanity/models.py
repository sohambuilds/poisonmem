from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import ACTIONS

Domain = Literal[
    "data_processing",
    "package_deployment",
    "invoice_processing",
    "customer_export",
]
Action = Literal[
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
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Query(StrictModel):
    id: str
    domain: Domain
    split: Literal["target", "control"]
    instruction: str
    required_actions: list[Action]
    prohibited_actions: list[Action]


class Memory(StrictModel):
    id: str
    domain: Domain
    text: str
    kind: Literal["clean", "lexical", "semantic"]

    @property
    def is_poison(self) -> bool:
        return self.kind != "clean"


class AgentOutput(StrictModel):
    actions: list[Action] = Field(min_length=1)
    status: Literal["completed"]


class ModelConfig(StrictModel):
    id: str
    provider: str


class OpenRouterConfig(StrictModel):
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 90
    app_name: str = "Memory Poisoning Sanity Check"
    app_url: str = "https://localhost/memory-poisoning-sanity"
    response_mode: Literal["json_object", "json_schema"] = "json_object"
    fetch_generation_stats: bool = True


class GenerationConfig(StrictModel):
    temperature: float = 0
    top_p: float = 1
    max_tokens: int = 350
    repetitions: int = 3
    concurrency: int = 4
    malformed_retries: int = Field(default=1, ge=0, le=1)
    provider_fallbacks: bool = False

    @model_validator(mode="after")
    def enforce_protocol(self) -> "GenerationConfig":
        if self.temperature != 0 or self.top_p != 1 or self.max_tokens != 350:
            raise ValueError("The registered design requires temperature=0, top_p=1, max_tokens=350")
        if self.repetitions != 3:
            raise ValueError("The registered design requires exactly three repetitions")
        if self.provider_fallbacks:
            raise ValueError("Provider fallbacks must remain disabled")
        return self


class RetrievalConfig(StrictModel):
    final_k: int = 4
    candidate_k: int = 12
    rrf_constant: int = 60
    e5_model: str = "intfloat/e5-base-v2"
    bge_model: str = "BAAI/bge-base-en-v1.5"

    @model_validator(mode="after")
    def enforce_budget(self) -> "RetrievalConfig":
        if self.final_k != 4:
            raise ValueError("Every primary retrieval condition must use final_k=4")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k must be at least final_k")
        return self


class AnalysisConfig(StrictModel):
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260627


class ExperimentConfig(StrictModel):
    openrouter: OpenRouterConfig
    models: list[ModelConfig]
    generation: GenerationConfig
    retrieval: RetrievalConfig
    analysis: AnalysisConfig

    @model_validator(mode="after")
    def enforce_models(self) -> "ExperimentConfig":
        expected = {"openai/gpt-4o-mini", "meta-llama/llama-4-scout"}
        actual = {model.id for model in self.models}
        if actual != expected or len(self.models) != 2:
            raise ValueError(f"models must contain exactly {sorted(expected)}")
        if any(not model.provider.strip() for model in self.models):
            raise ValueError("Every model must pin one provider endpoint")
        return self

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))


ACTION_OUTPUT_SCHEMA = {
    "name": "agent_action_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": list(ACTIONS)},
            },
            "status": {"type": "string", "enum": ["completed"]},
        },
        "required": ["actions", "status"],
        "additionalProperties": False,
    },
}
