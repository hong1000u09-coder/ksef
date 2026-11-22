#!/usr/bin/env python3
"""Auditable 27-candidate preference experiments.

All model fitting and offline replay use supplied responses only. The synthetic
data in the tests is not used by this program as a pretrained prior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_NAMES = {"P0", "P1", "P2"}
SELECTOR_NAMES = {"Random", "Balanced Random", "Predictive Entropy", "BALD"}
SIDES = {"left", "right"}


class DataError(ValueError):
    """Raised when an input dataset cannot support an auditable experiment."""


@dataclass
class Dataset:
    candidates: dict[str, dict[str, Any]]
    originals: dict[str, dict[str, Any]]
    questions: dict[str, dict[str, Any]]
    users: dict[str, dict[str, Any]]
    responses: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"Cannot read JSON data at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataError("Dataset root must be a JSON object.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise DataError(f"{label} must be a finite number.")
    return result


def _records_by_id(items: Any, kind: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        raise DataError(f"{kind} must be a JSON array.")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise DataError(f"Every {kind} record must have a non-empty string id.")
        if item["id"] in result:
            raise DataError(f"Duplicate {kind} id: {item['id']}")
        result[item["id"]] = item
    return result


def load_dataset(path: Path) -> Dataset:
    raw = _read_json(path)
    candidates = _records_by_id(raw.get("candidates"), "candidates")
    originals = _records_by_id(raw.get("originals"), "originals")
    questions = _records_by_id(raw.get("questions"), "questions")
    users = _records_by_id(raw.get("users"), "users")
    for uid, user in users.items():
        if user.get("role") not in {"developer", "evaluation"}:
            raise DataError(f"User {uid} role must be 'developer' or 'evaluation'.")
    responses_raw = raw.get("responses")
    if not isinstance(responses_raw, list):
        raise DataError("responses must be a JSON array.")
    if len(candidates) != 27:
        raise DataError(f"Expected exactly 27 correction candidates, got {len(candidates)}.")

    render_keys: dict[str, str] = {}
    adjustment_keys: dict[tuple[float, float, float], str] = {}
    signed_candidates = sum(candidate.get("render_signature") is not None for candidate in candidates.values())
    if signed_candidates not in (0, len(candidates)):
        raise DataError("Provide render_signature for all 27 candidates or for none of them.")
    for cid, candidate in candidates.items():
        adjustment = candidate.get("adjustment")
        if not isinstance(adjustment, list) or len(adjustment) != 3:
            raise DataError(f"Candidate {cid} needs adjustment=[temperature, contrast, saturation].")
        vector = tuple(_finite_number(v, f"candidate {cid} adjustment") for v in adjustment)
        if vector in adjustment_keys:
            raise DataError(
                f"Candidates {adjustment_keys[vector]} and {cid} have identical adjustment values."
            )
        adjustment_keys[vector] = cid
        signature = candidate.get("render_signature")
        if signature is not None:
            if not isinstance(signature, str) or not signature:
                raise DataError(f"Candidate {cid} render_signature must be a non-empty string.")
            if signature in render_keys:
                raise DataError(
                    f"Candidates {render_keys[signature]} and {cid} share render_signature."
                )
            render_keys[signature] = cid

    for oid, original in originals.items():
        if "leakage_group_id" in original and (
            not isinstance(original["leakage_group_id"], str) or not original["leakage_group_id"]
        ):
            raise DataError(f"Original {oid} leakage_group_id must be a non-empty string.")
        if "scene_label" in original and not isinstance(original["scene_label"], str):
            raise DataError(f"Original {oid} scene_label must be a string when present.")
        source_features = original.get("source_features", {})
        if not isinstance(source_features, dict):
            raise DataError(f"Original {oid} source_features must be an object.")
        for key, value in source_features.items():
            _finite_number(value, f"original {oid} source_features.{key}")

    unordered_pairs: set[tuple[str, str, str]] = set()
    for qid, question in questions.items():
        oid = question.get("original_id")
        a = question.get("candidate_a_id")
        b = question.get("candidate_b_id")
        if oid not in originals:
            raise DataError(f"Question {qid} references unknown original {oid!r}.")
        if a not in candidates or b not in candidates:
            raise DataError(f"Question {qid} references an unknown candidate.")
        if a == b:
            raise DataError(f"Question {qid} compares a candidate with itself.")
        key = (oid, *sorted((a, b)))
        if key in unordered_pairs:
            raise DataError(f"Duplicate unordered candidate pair for original {oid}: {a}, {b}.")
        unordered_pairs.add(key)

    seen_responses: set[tuple[str, str]] = set()
    responses: list[dict[str, Any]] = []
    for index, response in enumerate(responses_raw):
        if not isinstance(response, dict):
            raise DataError(f"Response at index {index} must be an object.")
        uid = response.get("user_id")
        qid = response.get("question_id")
        chosen = response.get("chosen_candidate_id")
        if uid not in users or qid not in questions:
            raise DataError(f"Response at index {index} has an unknown user or question.")
        question = questions[qid]
        if chosen not in (question["candidate_a_id"], question["candidate_b_id"]):
            raise DataError(f"Response at index {index} chose a candidate not shown in {qid}.")
        if (uid, qid) in seen_responses:
            raise DataError(f"Duplicate response for user {uid} and question {qid}.")
        seen_responses.add((uid, qid))
        if users[uid].get("role") not in {"developer", "evaluation"}:
            raise DataError(f"User {uid} role must be 'developer' or 'evaluation'.")
        if "response_ms" in response:
            response["response_ms"] = _finite_number(response["response_ms"], "response_ms")
            if response["response_ms"] < 0:
                raise DataError("response_ms cannot be negative.")
        responses.append(response)

    return Dataset(candidates, originals, questions, users, responses)


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    required = ("seeds", "evaluation_fraction", "model_comparison_budget", "budgets")
    for key in required:
        if key not in config:
            raise DataError(f"Configuration is missing {key!r}.")
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise DataError("seeds must be a non-empty array of integers.")
    if (
        not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in config["seeds"])
        or len(set(config["seeds"])) != len(config["seeds"])
    ):
        raise DataError("seeds must contain unique integers.")
    budgets = config["budgets"]
    if (
        not isinstance(budgets, list)
        or not budgets
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in budgets)
        or sorted(set(budgets)) != budgets
    ):
        raise DataError("budgets must be sorted, unique, non-negative integers.")
    fraction = _finite_number(config["evaluation_fraction"], "evaluation_fraction")
    if not 0 <= fraction < 1:
        raise DataError("evaluation_fraction must be in [0, 1).")
    if (
        not isinstance(config["model_comparison_budget"], int)
        or isinstance(config["model_comparison_budget"], bool)
        or config["model_comparison_budget"] < 0
    ):
        raise DataError("model_comparison_budget must be a non-negative integer.")
    if not isinstance(config.get("split_seed"), int) or isinstance(config.get("split_seed"), bool):
        raise DataError("split_seed must be an integer.")
    for key in ("prior_std", "interaction_prior_std"):
        value = _finite_number(config.get(key), key)
        if value <= 0:
            raise DataError(f"{key} must be greater than zero.")
    for key in ("map_max_iter", "posterior_samples", "bald_samples"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise DataError(f"{key} must be a positive integer.")
    edges = config.get("saturation_bin_edges", [0.33, 0.66])
    if not isinstance(edges, list):
        raise DataError("saturation_bin_edges must be sorted unique values strictly between 0 and 1.")
    normalized_edges = [_finite_number(edge, "saturation_bin_edges item") for edge in edges]
    if (
        any(not 0.0 < edge < 1.0 for edge in normalized_edges)
        or sorted(set(normalized_edges)) != normalized_edges
    ):
        raise DataError("saturation_bin_edges must be sorted unique values strictly between 0 and 1.")
    config["saturation_bin_edges"] = normalized_edges
    return config


def leakage_group(original: dict[str, Any]) -> str:
    return original.get("leakage_group_id", original["id"])


def make_split(dataset: Dataset, config: dict[str, Any]) -> tuple[set[str], set[str], dict[str, str]]:
    group_ids = sorted({leakage_group(row) for row in dataset.originals.values()})
    requested_eval = config.get("evaluation_group_ids")
    if requested_eval is not None:
        if not isinstance(requested_eval, list) or not all(isinstance(x, str) for x in requested_eval):
            raise DataError("evaluation_group_ids must be an array of group id strings.")
        unknown = set(requested_eval) - set(group_ids)
        if unknown:
            raise DataError(f"Unknown evaluation group ids: {sorted(unknown)}")
        eval_groups = set(requested_eval)
        source = "explicit_config"
    else:
        count = int(round(len(group_ids) * config["evaluation_fraction"]))
        count = max(1, count) if len(group_ids) >= 2 and config["evaluation_fraction"] > 0 else 0
        count = min(count, max(0, len(group_ids) - 1))
        rng = random.Random(config["split_seed"])
        shuffled = list(group_ids)
        rng.shuffle(shuffled)
        eval_groups = set(shuffled[:count])
        source = "seeded_group_split"
    question_groups = set(group_ids) - eval_groups
    partitions = {
        oid: ("evaluation" if leakage_group(original) in eval_groups else "question")
        for oid, original in dataset.originals.items()
    }
    return question_groups, eval_groups, {**partitions, "__split_source__": source}


def write_split_manifest(
    path: Path,
    dataset: Dataset,
    question_groups: set[str],
    eval_groups: set[str],
    source: str,
    split_seed: int,
    evaluation_fraction: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "original_id", "leakage_group_id", "partition", "split_source",
                "split_seed", "evaluation_fraction",
            ],
        )
        writer.writeheader()
        for oid in sorted(dataset.originals):
            group = leakage_group(dataset.originals[oid])
            writer.writerow(
                {
                    "original_id": oid,
                    "leakage_group_id": group,
                    "partition": "evaluation" if group in eval_groups else "question",
                    "split_source": source,
                    "split_seed": split_seed,
                    "evaluation_fraction": evaluation_fraction,
                }
            )


def feature_map(dataset: Dataset, config: dict[str, Any]) -> "ContextEncoder":
    return ContextEncoder.fit(
        [original for original in dataset.originals.values()],
        config.get("context_numeric_features", []),
        bool(config.get("include_scene_one_hot", True)),
    )


class ContextEncoder:
    def __init__(
        self,
        numeric_names: list[str],
        include_scene: bool,
        scene_categories: list[str],
        means: dict[str, float],
        scales: dict[str, float],
        available: bool,
        reason: str = "",
    ) -> None:
        self.numeric_names = numeric_names
        self.include_scene = include_scene
        self.scene_categories = scene_categories
        self.means = means
        self.scales = scales
        self.available = available
        self.reason = reason

    @classmethod
    def fit(
        cls, originals: Sequence[dict[str, Any]], numeric_names: Any, include_scene: bool
    ) -> "ContextEncoder":
        if not isinstance(numeric_names, list) or not all(isinstance(name, str) for name in numeric_names):
            raise DataError("context_numeric_features must be an array of names.")
        if not numeric_names and not include_scene:
            return cls([], False, [], {}, {}, False, "No context features were configured.")
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for name in numeric_names:
            values = []
            for original in originals:
                value = original.get("source_features", {}).get(name)
                if value is None:
                    return cls(
                        list(numeric_names), include_scene, [], {}, {}, False,
                        f"Missing original source feature {name!r}; P2 is unavailable.",
                    )
                values.append(_finite_number(value, f"source feature {name}"))
            if not values:
                return cls(list(numeric_names), include_scene, [], {}, {}, False, "No originals.")
            means[name] = float(sum(values) / len(values))
            variance = sum((value - means[name]) ** 2 for value in values) / len(values)
            scales[name] = math.sqrt(variance) if variance > 0 else 1.0
        categories: list[str] = []
        if include_scene:
            if any(not isinstance(original.get("scene_label"), str) or not original["scene_label"] for original in originals):
                return cls(
                    list(numeric_names), include_scene, [], means, scales, False,
                    "Recorded scene labels are missing; P2 is unavailable.",
                )
            categories = sorted({original["scene_label"] for original in originals})
        return cls(list(numeric_names), include_scene, categories, means, scales, True)

    @classmethod
    def fit_development(
        cls,
        dataset: Dataset,
        question_groups: set[str],
        config: dict[str, Any],
    ) -> "ContextEncoder":
        development_originals = [
            original for original in dataset.originals.values()
            if leakage_group(original) in question_groups
        ]
        encoder = cls.fit(
            development_originals,
            config.get("context_numeric_features", []),
            bool(config.get("include_scene_one_hot", True)),
        )
        if not encoder.available:
            return encoder
        numeric_names = encoder.numeric_names
        for original in dataset.originals.values():
            if any(name not in original.get("source_features", {}) for name in numeric_names):
                return cls(
                    numeric_names, encoder.include_scene, encoder.scene_categories,
                    encoder.means, encoder.scales, False,
                    "A configured original source feature is missing on at least one original; P2 is unavailable.",
                )
            if encoder.include_scene and not isinstance(original.get("scene_label"), str):
                return cls(
                    numeric_names, encoder.include_scene, encoder.scene_categories,
                    encoder.means, encoder.scales, False,
                    "A recorded scene label is missing on at least one original; P2 is unavailable.",
                )
        return encoder

    @property
    def dimension(self) -> int:
        return len(self.numeric_names) + len(self.scene_categories)

    def transform(self, original: dict[str, Any]) -> np.ndarray:
        if not self.available:
            raise DataError(self.reason)
        values = [
            (float(original["source_features"][name]) - self.means[name]) / self.scales[name]
            for name in self.numeric_names
        ]
        if self.include_scene:
            label = original.get("scene_label")
            values.extend(float(label == category) for category in self.scene_categories)
        return np.asarray(values, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_names": self.numeric_names,
            "include_scene": self.include_scene,
            "scene_categories": self.scene_categories,
            "means": self.means,
            "scales": self.scales,
        }


def candidate_features(candidate: dict[str, Any]) -> np.ndarray:
    a1, a2, a3 = candidate["adjustment"]
    return np.asarray(
        [a1, a2, a3, a1 * a1, a2 * a2, a3 * a3, a1 * a2, a1 * a3, a2 * a3],
        dtype=float,
    )


class BayesianPreferenceModel:
    """Gaussian-prior logistic preference model with a Laplace posterior."""

    def __init__(
        self,
        dataset: Dataset,
        model_name: str,
        context: ContextEncoder | None = None,
        prior_std: float = 1.0,
        interaction_prior_std: float = 0.15,
        max_iter: int = 100,
        tolerance: float = 1e-8,
    ) -> None:
        if model_name not in MODEL_NAMES:
            raise DataError(f"Unknown model: {model_name}")
        if model_name == "P2" and (context is None or not context.available):
            raise DataError("P2 requires available recorded original context features.")
        self.dataset = dataset
        self.model_name = model_name
        self.context = context
        self.prior_std = prior_std
        self.interaction_prior_std = interaction_prior_std
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.dimension = 9 + (3 * context.dimension if model_name == "P2" and context else 0)
        self.prior_precision = np.full(self.dimension, 1.0 / (prior_std * prior_std))
        if model_name == "P2" and context:
            self.prior_precision[9:] = 1.0 / (interaction_prior_std * interaction_prior_std)
        self.mean = np.zeros(self.dimension, dtype=float)
        self.covariance = np.diag(1.0 / self.prior_precision)
        self.training_question_ids: list[str] = []

    def _question_vector(self, question: dict[str, Any]) -> np.ndarray:
        candidate_a = self.dataset.candidates[question["candidate_a_id"]]
        candidate_b = self.dataset.candidates[question["candidate_b_id"]]
        difference = candidate_features(candidate_a) - candidate_features(candidate_b)
        if self.model_name != "P2":
            return difference
        assert self.context is not None
        z = self.context.transform(self.dataset.originals[question["original_id"]])
        delta_a = np.asarray(candidate_a["adjustment"], dtype=float) - np.asarray(
            candidate_b["adjustment"], dtype=float
        )
        return np.concatenate((difference, np.outer(delta_a, z).reshape(-1)))

    @staticmethod
    def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _log_likelihood(y: np.ndarray, logits: np.ndarray) -> float:
        return float(np.sum(y * -np.logaddexp(0.0, -logits) + (1.0 - y) * -np.logaddexp(0.0, logits)))

    def fit(self, responses: Sequence[dict[str, Any]]) -> None:
        if not responses:
            self.mean = np.zeros(self.dimension, dtype=float)
            self.covariance = np.diag(1.0 / self.prior_precision)
            self.training_question_ids = []
            return
        rows: list[np.ndarray] = []
        labels: list[float] = []
        question_ids: list[str] = []
        for response in responses:
            question = self.dataset.questions[response["question_id"]]
            rows.append(self._question_vector(question))
            labels.append(float(response["chosen_candidate_id"] == question["candidate_a_id"]))
            question_ids.append(question["id"])
        design = np.vstack(rows)
        target = np.asarray(labels, dtype=float)
        theta = np.zeros(self.dimension, dtype=float)
        prior = np.diag(self.prior_precision)

        def objective(value: np.ndarray) -> float:
            logits = design @ value
            return -self._log_likelihood(target, logits) + 0.5 * float(np.sum(self.prior_precision * value * value))

        for _ in range(self.max_iter):
            probabilities = np.asarray(self._sigmoid(design @ theta))
            gradient = design.T @ (probabilities - target) + self.prior_precision * theta
            weights = probabilities * (1.0 - probabilities)
            hessian = design.T @ (weights[:, None] * design) + prior
            step = np.linalg.solve(hessian, gradient)
            old_objective = objective(theta)
            scale = 1.0
            while scale > 1e-8:
                proposed = theta - scale * step
                if objective(proposed) <= old_objective:
                    break
                scale *= 0.5
            theta = proposed
            if float(np.linalg.norm(scale * step)) < self.tolerance:
                break

        probabilities = np.asarray(self._sigmoid(design @ theta))
        hessian = design.T @ ((probabilities * (1.0 - probabilities))[:, None] * design) + prior
        self.mean = theta
        self.covariance = np.linalg.inv(hessian)
        self.training_question_ids = question_ids

    def probability_a(self, question: dict[str, Any]) -> float:
        return float(self._sigmoid(self._question_vector(question) @ self.mean))

    def posterior_probabilities(
        self, question: dict[str, Any], samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        vector = self._question_vector(question)
        draws = rng.multivariate_normal(self.mean, self.covariance, size=samples)
        return np.asarray(self._sigmoid(draws @ vector), dtype=float)

    def posterior_predictive_probability(
        self, question: dict[str, Any], samples: int, rng: np.random.Generator
    ) -> float:
        return float(np.mean(self.posterior_probabilities(question, samples, rng)))

    def candidate_score(self, original: dict[str, Any], candidate: dict[str, Any]) -> float:
        psi = candidate_features(candidate)
        value = float(psi @ self.mean[:9])
        if self.model_name == "P2":
            assert self.context is not None
            z = self.context.transform(original)
            interaction = self.mean[9:].reshape(3, self.context.dimension)
            value += float(np.asarray(candidate["adjustment"], dtype=float) @ interaction @ z)
        return value


def bernoulli_entropy(probability: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def saturation_bucket(value: float, edges: Sequence[float]) -> str:
    for index, edge in enumerate(edges):
        if value < edge:
            return f"bin_{index}"
    return f"bin_{len(edges)}"


def question_stratum(question: dict[str, Any], dataset: Dataset, edges: Sequence[float]) -> str:
    original = dataset.originals[question["original_id"]]
    label = original.get("scene_label")
    saturation = original.get("source_features", {}).get("saturation_mean")
    if not isinstance(label, str) or not label or saturation is None:
        raise DataError(f"Question {question['id']} lacks recorded scene/saturation for balancing.")
    return f"{label}|{saturation_bucket(float(saturation), edges)}"


def common_selector_pool(
    dataset: Dataset, response_by_question: dict[str, dict[str, Any]], question_groups: set[str]
) -> list[dict[str, Any]]:
    pool = []
    for qid in sorted(response_by_question):
        question = dataset.questions[qid]
        original = dataset.originals[question["original_id"]]
        source_features = original.get("source_features", {})
        if (
            leakage_group(original) in question_groups
            and isinstance(original.get("scene_label"), str)
            and bool(original.get("scene_label"))
            and "saturation_mean" in source_features
        ):
            saturation = float(source_features["saturation_mean"])
            if 0.0 <= saturation <= 1.0:
                pool.append(question)
    return pool


class QuestionSelector:
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        if name not in SELECTOR_NAMES:
            raise DataError(f"Unknown selector: {name}")
        self.name = name
        self.config = config

    def select(
        self,
        questions: Sequence[dict[str, Any]],
        dataset: Dataset,
        model: BayesianPreferenceModel,
        rng: random.Random,
        posterior_rng: np.random.Generator,
    ) -> dict[str, Any]:
        if not questions:
            raise DataError("Cannot select from an empty question pool.")
        if self.name == "Random":
            return rng.choice(list(questions))
        if self.name == "Balanced Random":
            edges = self.config.get("saturation_bin_edges", [0.33, 0.66])
            groups: dict[str, list[dict[str, Any]]] = {}
            for question in questions:
                key = question_stratum(question, dataset, edges)
                groups.setdefault(key, []).append(question)
            smallest = min(len(items) for items in groups.values())
            tied = sorted(key for key, items in groups.items() if len(items) == smallest)
            chosen_group = rng.choice(tied)
            return rng.choice(groups[chosen_group])
        if self.name == "Predictive Entropy":
            sample_count = int(
                self.config.get("posterior_samples", self.config.get("bald_samples", 128))
            )
            scored = [
                (
                    bernoulli_entropy(
                        model.posterior_predictive_probability(
                            question, sample_count, posterior_rng
                        )
                    ),
                    question["id"],
                    question,
                )
                for question in questions
            ]
            return max(scored, key=lambda row: (row[0], row[1]))[2]
        sample_count = int(self.config.get("bald_samples", 128))
        scored_bald = []
        for question in questions:
            probabilities = model.posterior_probabilities(question, sample_count, posterior_rng)
            mean_probability = float(np.mean(probabilities))
            bald = bernoulli_entropy(mean_probability) - float(
                np.mean([bernoulli_entropy(float(value)) for value in probabilities])
            )
            scored_bald.append((bald, question["id"], question))
        return max(scored_bald, key=lambda row: (row[0], row[1]))[2]


def response_map(dataset: Dataset, user_id: str) -> dict[str, dict[str, Any]]:
    return {row["question_id"]: row for row in dataset.responses if row["user_id"] == user_id}


def responses_for_groups(
    dataset: Dataset, user_id: str, groups: set[str]
) -> list[dict[str, Any]]:
    return [
        row for row in dataset.responses
        if row["user_id"] == user_id
        and leakage_group(dataset.originals[dataset.questions[row["question_id"]]["original_id"]]) in groups
    ]


def evaluation_questions(
    dataset: Dataset, user_id: str, eval_groups: set[str]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    for response in dataset.responses:
        if response["user_id"] != user_id:
            continue
        question = dataset.questions[response["question_id"]]
        original = dataset.originals[question["original_id"]]
        if leakage_group(original) in eval_groups:
            pairs.append((question, response))
    return sorted(pairs, key=lambda item: item[0]["id"])


def mean_recommendation(
    model: BayesianPreferenceModel, dataset: Dataset, original: dict[str, Any]
) -> str:
    return max(
        dataset.candidates.values(),
        key=lambda candidate: (model.candidate_score(original, candidate), candidate["id"]),
    )["id"]


def score_predictions(
    model: BayesianPreferenceModel,
    eval_pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    dataset: Dataset,
    posterior_samples: int = 128,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for question, response in eval_pairs:
        posterior_seed = int.from_bytes(
            hashlib.sha256(
                f"eval:{seed}:{model.model_name}:{question['id']}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        probability_a = min(
            max(
                model.posterior_predictive_probability(
                    question, posterior_samples, np.random.default_rng(posterior_seed)
                ),
                1e-12,
            ),
            1.0 - 1e-12,
        )
        target_a = response["chosen_candidate_id"] == question["candidate_a_id"]
        rows.append(
            {
                "question_id": question["id"],
                "original_id": question["original_id"],
                "target_a": int(target_a),
                "probability_a": probability_a,
                "predicted_candidate_id": (
                    question["candidate_a_id"] if probability_a >= 0.5 else question["candidate_b_id"]
                ),
                "chosen_candidate_id": response["chosen_candidate_id"],
                "recommended_candidate_id": mean_recommendation(
                    model, dataset, dataset.originals[question["original_id"]]
                ),
            }
        )
    return rows


def metrics_for(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"log_loss": math.nan, "accuracy": math.nan, "brier": math.nan}
    losses = []
    accuracy = []
    brier = []
    for row in rows:
        p = min(max(float(row["probability_a"]), 1e-12), 1.0 - 1e-12)
        y = int(row["target_a"])
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
        accuracy.append(int((p >= 0.5) == bool(y)))
        brier.append((p - y) ** 2)
    return {
        "log_loss": float(sum(losses) / len(losses)),
        "accuracy": float(sum(accuracy) / len(accuracy)),
        "brier": float(sum(brier) / len(brier)),
    }


def make_model(
    dataset: Dataset,
    name: str,
    context: ContextEncoder,
    config: dict[str, Any],
) -> BayesianPreferenceModel:
    return BayesianPreferenceModel(
        dataset,
        name,
        context if name == "P2" else None,
        float(config["prior_std"]),
        float(config["interaction_prior_std"]),
        int(config["map_max_iter"]),
    )


def _arm_row(
    experiment: str,
    model_name: str,
    budget: int,
    seed: int,
    user_id: str,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    values = metrics_for(predictions)
    return {
        "experiment": experiment,
        "model_or_strategy": model_name,
        "budget": budget,
        "seed": seed,
        "user_id": user_id,
        "questions": len(predictions),
        "participants": 1,
        **values,
    }


def run_experiments(
    dataset: Dataset,
    config: dict[str, Any],
    output_dir: Path,
    experiment: str = "all",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    question_groups, eval_groups, partition_map = make_split(dataset, config)
    split_source = partition_map.pop("__split_source__")
    write_split_manifest(
        output_dir / "split_manifest.csv", dataset, question_groups, eval_groups,
        split_source, config["split_seed"], config["evaluation_fraction"],
    )
    context = ContextEncoder.fit_development(dataset, question_groups, config)

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    question_order_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    blind_queue_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not any(candidate.get("render_signature") for candidate in dataset.candidates.values()):
        warnings.append(
            "No render_signature values were supplied; only identical adjustment vectors were checked, "
            "not pixel-identical rendered outputs."
        )
    developer_responses = [
        response for response in dataset.responses
        if dataset.users[response["user_id"]]["role"] == "developer"
        and leakage_group(
            dataset.originals[dataset.questions[response["question_id"]]["original_id"]]
        ) in question_groups
    ]
    developer_users = sorted({response["user_id"] for response in developer_responses})
    p0: BayesianPreferenceModel | None = None
    if developer_responses:
        p0 = make_model(dataset, "P0", context, config)
        p0.fit(developer_responses)
        training_rows.append(
            {
                "experiment": "A",
                "model_or_strategy": "P0",
                "user_id": "DEVELOPMENT_USERS",
                "seed": "",
                "budget": len(developer_responses),
                "response_count": len(developer_responses),
                "question_ids": "|".join(sorted(p0.training_question_ids)),
            }
        )
    else:
        warnings.append(
            "P0 unavailable: no independent developer-user responses exist in the question partition; "
            "no pretrained/fabricated weights were used."
        )

    eval_users = sorted(
        uid for uid, user in dataset.users.items() if user.get("role") == "evaluation"
    )
    if not question_groups:
        warnings.append("No question partition exists; personalization and developer training are unavailable.")
    if not eval_groups:
        warnings.append(
            "No separate evaluation original group exists; new-original predictive evaluation is unavailable."
        )

    if experiment in {"all", "A"} and eval_groups:
        for user_id in eval_users:
            user_answers = responses_for_groups(dataset, user_id, question_groups)
            eval_pairs = evaluation_questions(dataset, user_id, eval_groups)
            if not eval_pairs:
                continue
            requested_ids = config.get("model_comparison_question_ids")
            if requested_ids is not None:
                if not isinstance(requested_ids, list) or not all(isinstance(x, str) for x in requested_ids):
                    raise DataError("model_comparison_question_ids must be an array of ids.")
                if len(set(requested_ids)) != len(requested_ids):
                    raise DataError("model_comparison_question_ids cannot contain duplicates.")
                by_id = {row["question_id"]: row for row in user_answers}
                if any(qid not in by_id for qid in requested_ids):
                    warnings.append(
                        f"Experiment A skipped {user_id}: configured common response set is incomplete."
                    )
                    continue
                common_training = [by_id[qid] for qid in requested_ids]
            else:
                user_answers.sort(key=lambda row: row["question_id"])
                budget = int(config["model_comparison_budget"])
                if len(user_answers) < budget:
                    warnings.append(
                        f"Experiment A skipped {user_id}: has {len(user_answers)} real question responses, "
                        f"needs {budget}."
                    )
                    continue
                common_training = user_answers[:budget]
            actual_budget = len(common_training)
            if p0 is not None:
                rows = score_predictions(
                    p0, eval_pairs, dataset, int(config["posterior_samples"]), config["seeds"][0]
                )
                for row in rows:
                    prediction_rows.append(
                        {
                            "experiment": "A",
                            "model_or_strategy": "P0",
                            "budget": actual_budget,
                            "seed": "",
                            "user_id": user_id,
                            **row,
                        }
                    )
                metric_rows.append(_arm_row("A", "P0", actual_budget, -1, user_id, rows))
                for original_id in sorted({q["original_id"] for q, _ in eval_pairs}):
                    original = dataset.originals[original_id]
                    recommendation_rows.append(
                        {
                            "experiment": "A",
                            "model_or_strategy": "P0",
                            "budget": actual_budget,
                            "seed": "",
                            "user_id": user_id,
                            "original_id": original_id,
                            "recommended_candidate_id": mean_recommendation(p0, dataset, original),
                        }
                    )
            for model_name in ("P1", "P2"):
                if model_name == "P2" and not context.available:
                    if context.reason not in warnings:
                        warnings.append(context.reason)
                    continue
                model = make_model(dataset, model_name, context, config)
                model.fit(common_training)
                training_rows.append(
                    {
                        "experiment": "A",
                        "model_or_strategy": model_name,
                        "user_id": user_id,
                        "seed": "",
                        "budget": actual_budget,
                        "response_count": len(common_training),
                        "question_ids": "|".join(model.training_question_ids),
                    }
                )
                rows = score_predictions(
                    model, eval_pairs, dataset, int(config["posterior_samples"]), config["seeds"][0]
                )
                for row in rows:
                    prediction_rows.append(
                        {
                            "experiment": "A",
                            "model_or_strategy": model_name,
                            "budget": actual_budget,
                            "seed": "",
                            "user_id": user_id,
                            **row,
                        }
                    )
                metric_rows.append(_arm_row("A", model_name, actual_budget, -1, user_id, rows))
                for original_id in sorted({q["original_id"] for q, _ in eval_pairs}):
                    recommendation_rows.append(
                        {
                            "experiment": "A",
                            "model_or_strategy": model_name,
                            "budget": actual_budget,
                            "seed": "",
                            "user_id": user_id,
                            "original_id": original_id,
                            "recommended_candidate_id": mean_recommendation(
                                model, dataset, dataset.originals[original_id]
                            ),
                        }
                    )

            if p0 is not None:
                arm_recs = {
                    row["model_or_strategy"]: row["recommended_candidate_id"]
                    for row in recommendation_rows
                    if row["experiment"] == "A" and row["user_id"] == user_id
                }
                for model_name in ("P1", "P2"):
                    if model_name not in arm_recs:
                        continue
                    for original_id in sorted({q["original_id"] for q, _ in eval_pairs}):
                        # The queue is built later from per-original recommendations.
                        model_rec = next(
                            row["recommended_candidate_id"]
                            for row in recommendation_rows
                            if row["experiment"] == "A"
                            and row["user_id"] == user_id
                            and row["model_or_strategy"] == model_name
                            and row["original_id"] == original_id
                        )
                        baseline_rec = next(
                            row["recommended_candidate_id"]
                            for row in recommendation_rows
                            if row["experiment"] == "A"
                            and row["user_id"] == user_id
                            and row["model_or_strategy"] == "P0"
                            and row["original_id"] == original_id
                        )
                        blind_queue_rows.append(
                            {
                                "comparison_id": f"A:{user_id}:{model_name}:{original_id}",
                                "experiment": "A",
                                "user_id": user_id,
                                "strategy_left": "P0",
                                "strategy_right": model_name,
                                "original_id": original_id,
                                "candidate_left_id": baseline_rec,
                                "candidate_right_id": model_rec,
                                "same_candidate": int(baseline_rec == model_rec),
                                "note": "Collect a real blind A/B response; do not score from either model's utility.",
                            }
                        )

    enabled_models = config.get("strategy_models", ["P1"])
    if not isinstance(enabled_models, list) or any(name not in {"P1", "P2"} for name in enabled_models):
        raise DataError("strategy_models may contain only P1 and P2.")
    if experiment in {"all", "B"} and eval_groups and question_groups:
        for user_id in eval_users:
            answers = response_map(dataset, user_id)
            pool = common_selector_pool(dataset, answers, question_groups)
            answered_question_count = sum(
                leakage_group(
                    dataset.originals[dataset.questions[qid]["original_id"]]
                ) in question_groups
                for qid in answers
            )
            excluded_count = answered_question_count - len(pool)
            if excluded_count > 0:
                warnings.append(
                    f"Experiment B excluded {excluded_count} answered questions for {user_id} "
                    "because original scene/saturation strata were missing or out of range; "
                    "the same common-pool rule was applied to every selector."
                )
            if not pool:
                warnings.append(
                    f"Experiment B skipped {user_id}: no shared, answered questions with recorded "
                    "scene and [0,1] source saturation for all four selectors."
                )
                continue
            eval_pairs = evaluation_questions(dataset, user_id, eval_groups)
            if not eval_pairs:
                continue
            max_budget = max(config["budgets"])
            if len(pool) < max_budget:
                warnings.append(
                    f"Experiment B skipped {user_id}: shared selector pool has {len(pool)} real responses, "
                    f"needs {max_budget}."
                )
                continue
            for model_name in enabled_models:
                if model_name == "P2" and not context.available:
                    if context.reason not in warnings:
                        warnings.append(context.reason)
                    continue
                for seed in config["seeds"]:
                    for selector_name in sorted(SELECTOR_NAMES):
                        selector = QuestionSelector(selector_name, config)
                        random_rng = random.Random(f"{seed}:{user_id}:{model_name}:{selector_name}")
                        posterior_rng = np.random.default_rng(
                            int.from_bytes(
                                hashlib.sha256(
                                    f"{seed}:{user_id}:{model_name}:{selector_name}".encode("utf-8")
                                ).digest()[:8],
                                "big",
                            )
                        )
                        model = make_model(dataset, model_name, context, config)
                        observed: list[dict[str, Any]] = []
                        remaining = list(pool)
                        predictions_by_budget: dict[int, list[dict[str, Any]]] = {}
                        requested_budgets = set(config["budgets"])
                        if 0 in requested_budgets:
                            predictions_by_budget[0] = score_predictions(
                                model, eval_pairs, dataset,
                                int(config["posterior_samples"]), seed,
                            )
                        for step in range(1, max_budget + 1):
                            started = time.perf_counter()
                            question = selector.select(
                                remaining, dataset, model, random_rng, posterior_rng
                            )
                            selection_ms = (time.perf_counter() - started) * 1000.0
                            # The selector has no reference to the response map. Reveal the
                            # selected question's actual response only after selection.
                            response = answers[question["id"]]
                            original = dataset.originals[question["original_id"]]
                            question_order_rows.append(
                                {
                                    "experiment": "B",
                                    "model": model_name,
                                    "strategy": selector_name,
                                    "user_id": user_id,
                                    "seed": seed,
                                    "step": step,
                                    "question_id": question["id"],
                                    "original_id": question["original_id"],
                                    "candidate_a_id": question["candidate_a_id"],
                                    "candidate_b_id": question["candidate_b_id"],
                                    "chosen_candidate_id": response["chosen_candidate_id"],
                                    "shown_left_candidate_id": question["candidate_a_id"],
                                    "shown_right_candidate_id": question["candidate_b_id"],
                                    "selected_side": (
                                        "left"
                                        if response["chosen_candidate_id"] == question["candidate_a_id"]
                                        else "right"
                                    ),
                                    "response_ms": response.get("response_ms", ""),
                                    "selection_ms": selection_ms,
                                    "stratum": question_stratum(
                                        question, dataset, config.get("saturation_bin_edges", [0.33, 0.66])
                                    ),
                                }
                            )
                            observed.append(response)
                            remaining.remove(question)
                            model.fit(observed)
                            if step in requested_budgets:
                                predictions_by_budget[step] = score_predictions(
                                    model, eval_pairs, dataset,
                                    int(config["posterior_samples"]), seed,
                                )
                        for budget in config["budgets"]:
                            rows = predictions_by_budget[budget]
                            arm_name = f"{model_name}+{selector_name}"
                            for row in rows:
                                prediction_rows.append(
                                    {
                                        "experiment": "B",
                                        "model_or_strategy": arm_name,
                                        "budget": budget,
                                        "seed": seed,
                                        "user_id": user_id,
                                        **row,
                                    }
                                )
                            metric_rows.append(
                                _arm_row("B", arm_name, budget, seed, user_id, rows)
                            )
                        training_rows.append(
                            {
                                "experiment": "B",
                                "model_or_strategy": f"{model_name}+{selector_name}",
                                "user_id": user_id,
                                "seed": seed,
                                "budget": max_budget,
                                "response_count": max_budget,
                                "question_ids": "|".join(
                                    row["question_id"]
                                    for row in question_order_rows
                                    if row["experiment"] == "B"
                                    and row["user_id"] == user_id
                                    and row["seed"] == seed
                                    and row["model"] == model_name
                                    and row["strategy"] == selector_name
                                ),
                            }
                        )
                        for original_id in sorted({q["original_id"] for q, _ in eval_pairs}):
                            recommendation_rows.append(
                                {
                                    "experiment": "B",
                                    "model_or_strategy": arm_name,
                                    "budget": max_budget,
                                    "seed": seed,
                                    "user_id": user_id,
                                    "original_id": original_id,
                                    "recommended_candidate_id": mean_recommendation(
                                        model, dataset, dataset.originals[original_id]
                                    ),
                                }
                            )

        # Candidate recommendations are generated only; preference labels are collected
        # separately through `record` and never inferred from model scores.
        by_recommendation: dict[tuple[str, str, str, str], str] = {}
        for row in recommendation_rows:
            if row["experiment"] == "B":
                by_recommendation[
                    (row["user_id"], str(row["seed"]), row["original_id"], row["model_or_strategy"])
                ] = row["recommended_candidate_id"]
        for (uid, seed, original_id, arm), left_candidate in sorted(by_recommendation.items()):
            for other_arm in sorted(
                candidate_arm
                for candidate_user, candidate_seed, candidate_original, candidate_arm in by_recommendation
                if candidate_user == uid
                and candidate_seed == seed
                and candidate_original == original_id
                and candidate_arm.split("+", 1)[0] == arm.split("+", 1)[0]
                and candidate_arm > arm
            ):
                right_candidate = by_recommendation[(uid, seed, original_id, other_arm)]
                blind_queue_rows.append(
                    {
                        "comparison_id": f"B:{uid}:{seed}:{original_id}:{arm}:{other_arm}",
                        "experiment": "B",
                        "user_id": uid,
                        "strategy_left": arm,
                        "strategy_right": other_arm,
                        "original_id": original_id,
                        "candidate_left_id": left_candidate,
                        "candidate_right_id": right_candidate,
                        "same_candidate": int(left_candidate == right_candidate),
                        "note": "Collect a real blind A/B response; do not score from either model's utility.",
                    }
                )

    _write_csv(output_dir / "predictions.csv", prediction_rows)
    _write_csv(output_dir / "metrics.csv", add_aggregate_metrics(metric_rows))
    _write_csv(output_dir / "question_order.csv", question_order_rows)
    _write_csv(output_dir / "training_log.csv", training_rows)
    _write_csv(output_dir / "recommendations.csv", recommendation_rows)
    _write_csv(output_dir / "blind_recommendation_queue.csv", blind_queue_rows)
    make_budget_plot(output_dir / "budget_curve.png", metric_rows)
    report = build_report(
        dataset, question_groups, eval_groups, context, developer_responses,
        prediction_rows, metric_rows, warnings, experiment, config,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "predictions": len(prediction_rows),
        "metric_rows": len(metric_rows),
        "question_order": len(question_order_rows),
        "warnings": warnings,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    else:
        fields = ["status"]
        rows = [{"status": "No evaluable rows; see report.md."}]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    output = list(rows)
    grouped_user: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_user.setdefault(
            (row["experiment"], row["model_or_strategy"], row["budget"], row["user_id"]), []
        ).append(row)
    user_averages: dict[tuple[str, str, int], list[dict[str, float]]] = {}
    for (experiment, arm, budget, _user), user_rows in grouped_user.items():
        user_averages.setdefault((experiment, arm, budget), []).append(
            {
                metric: sum(float(row[metric]) for row in user_rows) / len(user_rows)
                for metric in ("log_loss", "accuracy", "brier")
            }
        )
    for (experiment, arm, budget), users in sorted(user_averages.items()):
        output.append(
            {
                "experiment": experiment,
                "model_or_strategy": arm,
                "budget": budget,
                "seed": "all",
                "user_id": "ALL_USERS",
                "questions": "",
                "participants": len(users),
                **{
                    metric: sum(float(row[metric]) for row in users) / len(users)
                    for metric in ("log_loss", "accuracy", "brier")
                },
            }
        )
    return output


def make_budget_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["experiment"], row["model_or_strategy"], row["budget"], row["user_id"]), []
        ).append(row)
    per_user: dict[tuple[str, str, int], list[float]] = {}
    for (experiment, arm, budget, _user), samples in groups.items():
        per_user.setdefault((experiment, arm, budget), []).append(
            sum(float(sample["log_loss"]) for sample in samples) / len(samples)
        )
    for axis, experiment in zip(axes, ("A", "B")):
        arms = sorted({arm for exp, arm, _budget in per_user if exp == experiment})
        for arm in arms:
            points = sorted(
                (budget, values)
                for (exp, candidate_arm, budget), values in per_user.items()
                if exp == experiment and candidate_arm == arm
            )
            if points:
                axis.plot(
                    [budget for budget, _ in points],
                    [sum(values) / len(values) for _, values in points],
                    marker="o",
                    label=arm,
                )
        axis.set_title("A: preference models" if experiment == "A" else "B: question strategies")
        axis.set_xlabel("Observed preference responses")
        axis.set_ylabel("Mean per-user log loss (lower is better)")
        axis.grid(True, alpha=0.25)
        if arms:
            axis.legend(fontsize="small")
    fig.suptitle("Evaluation is averaged over participants; seeds are repeated runs, not participants")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_report(
    dataset: Dataset,
    question_groups: set[str],
    eval_groups: set[str],
    context: ContextEncoder,
    developer_responses: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    warnings: list[str],
    experiment: str,
    config: dict[str, Any],
) -> str:
    eval_users = sum(1 for user in dataset.users.values() if user.get("role") == "evaluation")
    lines = [
        "# Personalization experiment report",
        "",
        f"- Requested experiment: `{experiment}`",
        f"- Dataset: {len(dataset.originals)} originals, {len(dataset.questions)} questions, "
        f"{len(dataset.candidates)} candidates.",
        f"- User roles: {eval_users} evaluation users; {len({r['user_id'] for r in developer_responses})} "
        "independent development users with question-partition responses.",
        f"- Split: {len(question_groups)} question leakage groups; {len(eval_groups)} evaluation leakage groups.",
        f"- Configured response budgets: `{config['budgets']}`; model-comparison budget: "
        f"`{config['model_comparison_budget']}`.",
        "",
        "## Results status",
        "",
    ]
    if predictions:
        heldout_count = len(
            {(row["user_id"], row["question_id"]) for row in predictions}
        )
        lines.append(
            f"Predictions cover {heldout_count} actual held-out user/question responses "
            f"({len(predictions)} arm/budget-specific predictions). Consult `metrics.csv` and "
            "`budget_curve.png`; no synthetic labels are included."
        )
    else:
        lines.append(
            "No held-out prediction rows were produced. This is not evidence that one model or "
            "selector performs better."
        )
    if not developer_responses:
        lines.append(
            "P0 could not be evaluated: no independent development-user responses were available. "
            "No shared-preference weights were fabricated."
        )
    if not context.available:
        lines.append(f"P2 unavailable: {context.reason}")
    elif "P2" not in {row["model_or_strategy"].split("+")[0] for row in metrics}:
        lines.append("P2 context was available but no P2 result was evaluable under this split/budget.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- P1 ignores photo context and can rank the same adjustment candidates identically for every original.",
            "- P2 has 3 coefficients per context feature in addition to the 9 shared adjustment features; "
            "its interaction prior is strongly regularized, but it can still overfit.",
            "- P2 uses only recorded original scene labels and configured original color statistics. "
            "Feature selection is fixed in config and numeric standardization is fit only on question-partition originals.",
            "- A/B labels on evaluation-partition originals are used only for scoring. The selector receives "
            "the question pool and current posterior; the selected answer is revealed afterward.",
            "- Offline replay is conditional on questions with real responses. It is not a causal estimate of "
            "how users would answer unasked questions.",
            "- Known near-duplicates are held together only when `leakage_group_id` is supplied correctly. "
            "The program does not infer duplicate images.",
            "- Balanced Random uses recorded `scene_label × saturation_mean` from the original, saturation "
            "bins from config, chooses the least-populated remaining condition, breaks condition ties uniformly, "
            "then chooses a question uniformly within that condition.",
            "- Candidate question pool is the supplied finite set of valid, answered questions; each selector "
            "chooses an entire `(original_id, candidate_A, candidate_B)` question. It does not generate or "
            "optimize candidate pairs outside that common pool.",
            "- Recommendations maximize posterior mean score over the 27 candidates. That score is not a "
            "ground-truth satisfaction label. `blind_recommendation_queue.csv` is for collecting new blind A/B responses.",
            "- Aggregate participant counts count users, not seeds. Seed-specific runs are repeated measurements.",
            "",
            "## Warnings / unevaluable scope",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- No run-time data-sufficiency warnings.")
    return "\n".join(lines) + "\n"


def assign_participant(
    participant_id: str, arms: Sequence[str], seed: int, assignments_path: Path
) -> dict[str, str]:
    if not participant_id:
        raise DataError("participant_id cannot be empty.")
    unique_arms = sorted(set(arms))
    if len(unique_arms) < 2:
        raise DataError("At least two strategy arms are required for randomized assignment.")
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    if assignments_path.exists():
        with assignments_path.open(newline="", encoding="utf-8-sig") as file:
            existing = [
                row for row in csv.DictReader(file) if row["participant_id"] == participant_id
            ]
        if existing:
            if len(existing) != 1:
                raise DataError("Participant has a crossover assignment; use that sequence instead.")
            if existing[0]["arms"].split("|") != unique_arms:
                raise DataError("Participant already assigned with a different arm set.")
            return existing[0]
    rng = random.Random(f"{seed}:{participant_id}")
    arm = rng.choice(unique_arms)
    new_file = not assignments_path.exists() or assignments_path.stat().st_size == 0
    with assignments_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "participant_id", "strategy_id", "arms", "seed", "assigned_at", "sequence_order"
            ],
        )
        if new_file:
            writer.writeheader()
        row = {
            "participant_id": participant_id,
            "strategy_id": arm,
            "arms": "|".join(unique_arms),
            "seed": seed,
            "assigned_at": datetime.now(timezone.utc).isoformat(),
            "sequence_order": 1,
        }
        writer.writerow(row)
    return row


def assign_crossover(
    participant_id: str, arms: Sequence[str], seed: int, assignments_path: Path
) -> list[dict[str, Any]]:
    if not participant_id:
        raise DataError("participant_id cannot be empty.")
    unique_arms = sorted(set(arms))
    if len(unique_arms) < 2:
        raise DataError("At least two strategy arms are required for a crossover sequence.")
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    if assignments_path.exists():
        with assignments_path.open(newline="", encoding="utf-8-sig") as file:
            existing = [
                row for row in csv.DictReader(file) if row["participant_id"] == participant_id
            ]
        if existing:
            existing.sort(key=lambda row: int(row.get("sequence_order") or 0))
            if (
                len(existing) != len(unique_arms)
                or sorted(row["strategy_id"] for row in existing) != unique_arms
            ):
                raise DataError("Participant already has a different strategy assignment.")
            return existing
    order = list(unique_arms)
    random.Random(f"crossover:{seed}:{participant_id}").shuffle(order)
    new_file = not assignments_path.exists() or assignments_path.stat().st_size == 0
    rows = []
    with assignments_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "participant_id", "strategy_id", "arms", "seed", "assigned_at", "sequence_order"
            ],
        )
        if new_file:
            writer.writeheader()
        timestamp = datetime.now(timezone.utc).isoformat()
        for index, arm in enumerate(order, start=1):
            row = {
                "participant_id": participant_id,
                "strategy_id": arm,
                "arms": "|".join(unique_arms),
                "seed": seed,
                "assigned_at": timestamp,
                "sequence_order": index,
            }
            writer.writerow(row)
            rows.append(row)
    return rows


def _check_assignment(
    user_id: str, strategy_id: str, assignments_path: Path
) -> dict[str, str]:
    if not assignments_path.exists():
        raise DataError(f"Randomized assignment log does not exist: {assignments_path}")
    with assignments_path.open(newline="", encoding="utf-8-sig") as file:
        assignments = [
            row for row in csv.DictReader(file) if row["participant_id"] == user_id
        ]
    if not assignments:
        raise DataError(f"Participant {user_id} has no randomized strategy assignment.")
    assignment = next(
        (row for row in assignments if row["strategy_id"] == strategy_id), None
    )
    if assignment is None:
        raise DataError(
            f"Participant {user_id} has no assignment for strategy {strategy_id}."
        )
    return assignment


def select_online_question(
    dataset: Dataset,
    config: dict[str, Any],
    events_path: Path,
    assignments_path: Path,
    user_id: str,
    strategy_id: str,
    session_id: str,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    _check_assignment(user_id, strategy_id, assignments_path)
    if strategy_id not in SELECTOR_NAMES:
        raise DataError(f"Online strategy_id must be one of {sorted(SELECTOR_NAMES)}.")
    question_groups, _eval_groups, _ = make_split(dataset, config)
    context = ContextEncoder.fit_development(dataset, question_groups, config)
    if model_name == "P2" and not context.available:
        raise DataError(context.reason)

    observed: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    if events_path.exists():
        with events_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"Invalid event JSON at line {line_number}: {exc}") from exc
                if (
                    event.get("event_type") != "preference"
                    or event.get("user_id") != user_id
                    or event.get("strategy_id") != strategy_id
                    or event.get("session_id") != session_id
                ):
                    continue
                qid = event.get("question_id")
                if qid not in dataset.questions:
                    raise DataError(f"Online event at line {line_number} has unknown question_id.")
                question = dataset.questions[qid]
                if event.get("original_id") != question["original_id"]:
                    raise DataError(f"Online event at line {line_number} original does not match question.")
                chosen = event.get("chosen_candidate_id")
                if chosen not in (question["candidate_a_id"], question["candidate_b_id"]):
                    raise DataError(f"Online event at line {line_number} has invalid candidate mapping.")
                seen_question_ids.add(qid)
                if leakage_group(dataset.originals[question["original_id"]]) in question_groups:
                    observed.append({"question_id": qid, "chosen_candidate_id": chosen})

    pool = []
    for question in dataset.questions.values():
        original = dataset.originals[question["original_id"]]
        source_features = original.get("source_features", {})
        saturation = source_features.get("saturation_mean")
        if (
            leakage_group(original) in question_groups
            and question["id"] not in seen_question_ids
            and isinstance(original.get("scene_label"), str)
            and bool(original.get("scene_label"))
            and saturation is not None
            and 0.0 <= float(saturation) <= 1.0
        ):
            pool.append(question)
    pool.sort(key=lambda question: question["id"])
    if not pool:
        raise DataError("No unasked question remains in the common valid online pool.")
    model = make_model(dataset, model_name, context, config)
    model.fit(observed)
    random_rng = random.Random(f"{seed}:{user_id}:{strategy_id}:{session_id}:{len(observed)}")
    posterior_rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(
                f"{seed}:{user_id}:{strategy_id}:{session_id}:{len(observed)}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
    )
    selector = QuestionSelector(strategy_id, config)
    started = time.perf_counter()
    question = selector.select(pool, dataset, model, random_rng, posterior_rng)
    selection_ms = (time.perf_counter() - started) * 1000.0
    display_rng = random.Random(
        f"display:{seed}:{user_id}:{strategy_id}:{session_id}:{question['id']}"
    )
    left_id, right_id = question["candidate_a_id"], question["candidate_b_id"]
    if display_rng.choice((False, True)):
        left_id, right_id = right_id, left_id
    return {
        "question_id": question["id"],
        "original_id": question["original_id"],
        "candidate_a_id": question["candidate_a_id"],
        "candidate_b_id": question["candidate_b_id"],
        "shown_left_candidate_id": left_id,
        "shown_right_candidate_id": right_id,
        "model": model_name,
        "strategy_id": strategy_id,
        "response_count": len(observed),
        "selection_ms": selection_ms,
    }


def record_online_response(
    dataset: Dataset,
    events_path: Path,
    user_id: str,
    strategy_id: str,
    session_id: str,
    original_id: str,
    left_id: str,
    right_id: str,
    selected_side: str,
    response_ms: float | None,
    question_id: str | None = None,
    comparison_id: str | None = None,
    comparison_kind: str = "preference",
    assignments_path: Path | None = None,
    queue_path: Path | None = None,
) -> dict[str, Any]:
    if not all((user_id, strategy_id, session_id, original_id, left_id, right_id)):
        raise DataError("user, strategy, session, original and both displayed candidates are required.")
    if original_id not in dataset.originals or left_id not in dataset.candidates or right_id not in dataset.candidates:
        raise DataError("Online response references an unknown original or candidate.")
    if selected_side not in {"left", "right", "tie"}:
        raise DataError("selected_side must be left, right, or tie.")
    if left_id == right_id and selected_side != "tie":
        raise DataError("Identical displayed candidates must be recorded as a tie.")
    if left_id != right_id and selected_side == "tie":
        raise DataError("A tie is valid only when the displayed candidate ids are identical.")
    if comparison_kind not in {"preference", "recommendation"}:
        raise DataError("comparison_kind must be preference or recommendation.")
    if question_id:
        if question_id not in dataset.questions:
            raise DataError(f"Unknown question id: {question_id}")
        question = dataset.questions[question_id]
        if question["original_id"] != original_id or {left_id, right_id} != {
            question["candidate_a_id"], question["candidate_b_id"]
        }:
            raise DataError("Displayed left/right candidates do not match the identified question.")
        if comparison_kind != "preference":
            raise DataError("Question-bank records must be preference events.")
    if comparison_kind == "recommendation" and not comparison_id:
        raise DataError("Blind recommendation comparisons require comparison_id.")
    if comparison_kind == "recommendation":
        if queue_path is None or not queue_path.exists():
            raise DataError("Recommendation responses require the matching blind comparison queue.")
        with queue_path.open(newline="", encoding="utf-8-sig") as file:
            queue_row = next(
                (row for row in csv.DictReader(file) if row["comparison_id"] == comparison_id), None
            )
        if queue_row is None:
            raise DataError(f"Unknown blind recommendation comparison_id: {comparison_id}")
        if queue_row["original_id"] != original_id or {
            left_id, right_id
        } != {queue_row["candidate_left_id"], queue_row["candidate_right_id"]}:
            raise DataError("Displayed candidates do not match the blind recommendation queue.")
        if queue_row["same_candidate"] == "1" and selected_side != "tie":
            raise DataError("Identical recommendations must be recorded as a tie.")
    if assignments_path is not None:
        _check_assignment(user_id, strategy_id, assignments_path)
    if response_ms is not None and (not math.isfinite(response_ms) or response_ms < 0):
        raise DataError("response_ms must be non-negative and finite.")
    if events_path.exists():
        with events_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"Invalid event JSON at line {line_number}: {exc}") from exc
                if question_id and comparison_kind == "preference" and (
                    prior.get("event_type") == "preference"
                    and prior.get("user_id") == user_id
                    and prior.get("strategy_id") == strategy_id
                    and prior.get("session_id") == session_id
                    and prior.get("question_id") == question_id
                ):
                    raise DataError("This question already has a response in this user/strategy/session.")
                if comparison_kind == "recommendation" and (
                    prior.get("event_type") == "recommendation"
                    and prior.get("user_id") == user_id
                    and prior.get("comparison_id") == comparison_id
                ):
                    raise DataError("This blind recommendation comparison was already recorded for this user.")
    chosen = left_id if selected_side == "left" else right_id if selected_side == "right" else None
    event = {
        "event_type": comparison_kind,
        "user_id": user_id,
        "strategy_id": strategy_id,
        "session_id": session_id,
        "question_id": question_id,
        "comparison_id": comparison_id,
        "original_id": original_id,
        "shown_left_candidate_id": left_id,
        "shown_right_candidate_id": right_id,
        "selected_side": selected_side,
        "chosen_candidate_id": chosen,
        "response_ms": response_ms,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def analyze_recommendation_events(
    events_path: Path, queue_path: Path, output_path: Path
) -> list[dict[str, Any]]:
    if not events_path.exists() or not queue_path.exists():
        raise DataError("Both online event log and blind recommendation queue are required.")
    with queue_path.open(newline="", encoding="utf-8-sig") as file:
        queue = {row["comparison_id"]: row for row in csv.DictReader(file)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    with events_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid event JSON at line {line_number}: {exc}") from exc
            if event.get("event_type") != "recommendation":
                continue
            comparison_id = event.get("comparison_id")
            queue_row = queue.get(comparison_id)
            if queue_row is None:
                raise DataError(f"Recommendation event at line {line_number} has no queue row.")
            if event.get("original_id") != queue_row["original_id"] or {
                event.get("shown_left_candidate_id"), event.get("shown_right_candidate_id")
            } != {queue_row["candidate_left_id"], queue_row["candidate_right_id"]}:
                raise DataError(f"Recommendation event at line {line_number} does not match its queued pair.")
            if queue_row["same_candidate"] == "1":
                if event.get("selected_side") != "tie" or event.get("chosen_candidate_id") is not None:
                    raise DataError(f"Identical recommendations must be scored as a tie (line {line_number}).")
            else:
                expected_choice = (
                    event.get("shown_left_candidate_id")
                    if event.get("selected_side") == "left"
                    else event.get("shown_right_candidate_id")
                    if event.get("selected_side") == "right"
                    else None
                )
                if event.get("selected_side") not in SIDES or event.get("chosen_candidate_id") != expected_choice:
                    raise DataError(f"Recommendation event side mapping is invalid at line {line_number}.")
            key = (comparison_id, event["user_id"])
            if key in seen:
                raise DataError(f"Duplicate recommendation evaluation for comparison/user {key}.")
            seen.add(key)
            grouped.setdefault(comparison_id, []).append(event)

    results = []
    for comparison_id, queue_row in queue.items():
        events = grouped.get(comparison_id, [])
        left_wins = sum(
            event["chosen_candidate_id"] == queue_row["candidate_left_id"]
            for event in events
            if event["chosen_candidate_id"] is not None
        )
        right_wins = sum(
            event["chosen_candidate_id"] == queue_row["candidate_right_id"]
            for event in events
            if event["chosen_candidate_id"] is not None
        )
        ties = sum(event["selected_side"] == "tie" for event in events)
        times = [float(event["response_ms"]) for event in events if event.get("response_ms") is not None]
        results.append(
            {
                "comparison_id": comparison_id,
                "experiment": queue_row["experiment"],
                "strategy_left": queue_row["strategy_left"],
                "strategy_right": queue_row["strategy_right"],
                "original_id": queue_row["original_id"],
                "candidate_left_id": queue_row["candidate_left_id"],
                "candidate_right_id": queue_row["candidate_right_id"],
                "same_candidate": queue_row["same_candidate"],
                "responses": len(events),
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
                "left_win_rate_excluding_ties": (
                    left_wins / (left_wins + right_wins) if left_wins + right_wins else ""
                ),
                "mean_response_ms": sum(times) / len(times) if times else "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, results)
    return results


def update_online_model(
    dataset: Dataset,
    config: dict[str, Any],
    events_path: Path,
    user_id: str,
    strategy_id: str,
    session_id: str,
    model_name: str,
    output_path: Path,
    assignments_path: Path | None = None,
) -> dict[str, Any]:
    if model_name not in {"P1", "P2"}:
        raise DataError("Online user updates support P1 or P2, never P0.")
    if assignments_path is not None:
        _check_assignment(user_id, strategy_id, assignments_path)
    if not events_path.exists():
        raise DataError(f"Online response log does not exist: {events_path}")
    question_groups, _eval_groups, _ = make_split(dataset, config)
    context = ContextEncoder.fit_development(dataset, question_groups, config)
    responses = []
    with events_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid event JSON at line {line_number}: {exc}") from exc
            if (
                event.get("event_type") == "preference"
                and event.get("user_id") == user_id
                and event.get("strategy_id") == strategy_id
                and event.get("session_id") == session_id
                and event.get("question_id") in dataset.questions
                and event.get("chosen_candidate_id") is not None
            ):
                question = dataset.questions[event["question_id"]]
                original = dataset.originals[question["original_id"]]
                if leakage_group(original) not in question_groups:
                    continue
                chosen = event["chosen_candidate_id"]
                if chosen not in (question["candidate_a_id"], question["candidate_b_id"]):
                    raise DataError(f"Online event at line {line_number} has invalid candidate mapping.")
                responses.append(
                    {"question_id": question["id"], "chosen_candidate_id": chosen}
                )
    model = make_model(dataset, model_name, context, config)
    model.fit(responses)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model_name,
        "user_id": user_id,
        "strategy_id": strategy_id,
        "session_id": session_id,
        "response_count": len(responses),
        "training_question_ids": model.training_question_ids,
        "posterior_mean": model.mean.tolist(),
        "posterior_covariance": model.covariance.tolist(),
        "context_encoder": context.to_dict() if model_name == "P2" else None,
    }
    output_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    return checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run model comparison A and/or selector comparison B.")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--config", type=Path, default=Path("config.json"))
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("--experiment", choices=["all", "A", "B"], default="all")

    assign = subparsers.add_parser("assign", help="Randomly assign a participant to one strategy arm.")
    assign.add_argument("--participant-id", required=True)
    assign.add_argument("--arms", nargs="+", required=True)
    assign.add_argument("--seed", type=int, required=True)
    assign.add_argument("--out", type=Path, default=Path("online/assignments.csv"))

    crossover = subparsers.add_parser(
        "assign-crossover", help="Randomize the strategy order for an explicit crossover participant."
    )
    crossover.add_argument("--participant-id", required=True)
    crossover.add_argument("--arms", nargs="+", required=True)
    crossover.add_argument("--seed", type=int, required=True)
    crossover.add_argument("--out", type=Path, default=Path("online/assignments.csv"))

    record = subparsers.add_parser("record", help="Append an actual, side-mapped online response.")
    record.add_argument("--data", type=Path, required=True)
    record.add_argument("--user-id", required=True)
    record.add_argument("--strategy-id", required=True)
    record.add_argument("--session-id", required=True)
    record.add_argument("--original-id", required=True)
    record.add_argument("--left-id", required=True)
    record.add_argument("--right-id", required=True)
    record.add_argument("--selected-side", choices=["left", "right", "tie"], required=True)
    record.add_argument("--response-ms", type=float)
    record.add_argument("--question-id")
    record.add_argument("--comparison-id")
    record.add_argument("--comparison-kind", choices=["preference", "recommendation"], default="preference")
    record.add_argument("--out", type=Path, default=Path("online/responses.jsonl"))
    record.add_argument("--assignments", type=Path, default=Path("online/assignments.csv"))
    record.add_argument("--queue", type=Path, default=Path("results/blind_recommendation_queue.csv"))

    next_question = subparsers.add_parser("next", help="Select the next real question for an online session.")
    next_question.add_argument("--data", type=Path, required=True)
    next_question.add_argument("--config", type=Path, default=Path("config.json"))
    next_question.add_argument("--events", type=Path, default=Path("online/responses.jsonl"))
    next_question.add_argument("--assignments", type=Path, default=Path("online/assignments.csv"))
    next_question.add_argument("--user-id", required=True)
    next_question.add_argument("--strategy-id", required=True)
    next_question.add_argument("--session-id", required=True)
    next_question.add_argument("--model", choices=["P1", "P2"], default="P1")
    next_question.add_argument("--seed", type=int, required=True)

    update = subparsers.add_parser("update", help="Fit an isolated online user/strategy/session posterior.")
    update.add_argument("--data", type=Path, required=True)
    update.add_argument("--config", type=Path, default=Path("config.json"))
    update.add_argument("--events", type=Path, default=Path("online/responses.jsonl"))
    update.add_argument("--user-id", required=True)
    update.add_argument("--strategy-id", required=True)
    update.add_argument("--session-id", required=True)
    update.add_argument("--model", choices=["P1", "P2"], required=True)
    update.add_argument("--out", type=Path, required=True)
    update.add_argument("--assignments", type=Path, default=Path("online/assignments.csv"))

    analyze = subparsers.add_parser(
        "analyze-recommendations", help="Summarize actual blind recommendation responses."
    )
    analyze.add_argument("--events", type=Path, default=Path("online/responses.jsonl"))
    analyze.add_argument("--queue", type=Path, required=True)
    analyze.add_argument("--out", type=Path, default=Path("results/recommendation_outcomes.csv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            dataset = load_dataset(args.data)
            config = load_config(args.config)
            summary = run_experiments(dataset, config, args.out, args.experiment)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == "assign":
            result = assign_participant(args.participant_id, args.arms, args.seed, args.out)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "assign-crossover":
            result = assign_crossover(args.participant_id, args.arms, args.seed, args.out)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "record":
            dataset = load_dataset(args.data)
            result = record_online_response(
                dataset, args.out, args.user_id, args.strategy_id, args.session_id,
                args.original_id, args.left_id, args.right_id, args.selected_side,
                args.response_ms, args.question_id, args.comparison_id, args.comparison_kind,
                args.assignments, args.queue,
            )
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "next":
            dataset = load_dataset(args.data)
            config = load_config(args.config)
            result = select_online_question(
                dataset, config, args.events, args.assignments, args.user_id,
                args.strategy_id, args.session_id, args.model, args.seed,
            )
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "update":
            dataset = load_dataset(args.data)
            config = load_config(args.config)
            result = update_online_model(
                dataset, config, args.events, args.user_id, args.strategy_id,
                args.session_id, args.model, args.out, args.assignments,
            )
            print(json.dumps({key: value for key, value in result.items() if key != "posterior_covariance"}))
        elif args.command == "analyze-recommendations":
            result = analyze_recommendation_events(args.events, args.queue, args.out)
            print(json.dumps({"comparisons": len(result), "output": str(args.out)}, ensure_ascii=False))
        return 0
    except (DataError, OSError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
