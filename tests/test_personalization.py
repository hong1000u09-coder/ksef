import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

import personalization as app


def synthetic_fixture():
    candidates = []
    for index, temperature in enumerate((-1, 0, 1)):
        for contrast in (-1, 0, 1):
            for saturation in (-1, 0, 1):
                candidates.append(
                    {
                        "id": f"c{len(candidates):02d}",
                        "adjustment": [temperature, contrast, saturation],
                    }
                )
    originals = []
    questions = []
    responses = []
    for index in range(6):
        originals.append(
            {
                "id": f"img{index}",
                "leakage_group_id": f"group{index}",
                "scene_label": "indoor" if index % 2 == 0 else "outdoor",
                "source_features": {
                    "brightness_mean": 0.1 + index * 0.1,
                    "saturation_mean": 0.1 + index * 0.1,
                },
            }
        )
        questions.append(
            {
                "id": f"q{index}",
                "original_id": f"img{index}",
                "candidate_a_id": "c00",
                "candidate_b_id": "c01",
            }
        )
        if index < 4:
            responses.append(
                {
                    "user_id": "dev",
                    "question_id": f"q{index}",
                    "chosen_candidate_id": "c00" if index % 2 == 0 else "c01",
                }
            )
        responses.append(
            {
                "user_id": "eval",
                "question_id": f"q{index}",
                "chosen_candidate_id": "c00" if index % 2 == 0 else "c01",
                "response_ms": 1000 + index,
            }
        )
    return {
        "candidates": candidates,
        "originals": originals,
        "questions": questions,
        "users": [
            {"id": "dev", "role": "developer"},
            {"id": "eval", "role": "evaluation"},
        ],
        "responses": responses,
    }


def test_config():
    return {
        "split_seed": 3,
        "evaluation_fraction": 0.3,
        "evaluation_group_ids": ["group4", "group5"],
        "model_comparison_budget": 1,
        "model_comparison_question_ids": None,
        "budgets": [0, 1],
        "seeds": [11],
        "strategy_models": ["P1", "P2"],
        "context_numeric_features": ["brightness_mean", "saturation_mean"],
        "include_scene_one_hot": True,
        "saturation_bin_edges": [0.33, 0.66],
        "prior_std": 1.0,
        "interaction_prior_std": 0.1,
        "map_max_iter": 100,
        "posterior_samples": 64,
        "bald_samples": 64,
    }


class FakePosterior:
    def probability_a(self, question):
        return 0.5

    def posterior_probabilities(self, question, samples, rng):
        if question["id"] == "a-low-variance":
            return np.asarray([0.5] * samples)
        return np.resize(np.asarray([0.01, 0.39]), samples)

    def posterior_predictive_probability(self, question, samples, rng):
        return float(np.mean(self.posterior_probabilities(question, samples, rng)))


class PersonalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_path = self.root / "data.json"
        self.data = synthetic_fixture()
        self.data_path.write_text(json.dumps(self.data), encoding="utf-8")
        self.dataset = app.load_dataset(self.data_path)
        self.config = test_config()

    def tearDown(self):
        self.temp.cleanup()

    def test_feature_definition_and_posterior_are_reproducible(self):
        candidate = self.dataset.candidates["c00"]
        np.testing.assert_array_equal(
            app.candidate_features(candidate),
            [-1, -1, -1, 1, 1, 1, 1, 1, 1],
        )
        context = app.ContextEncoder.fit_development(
            self.dataset, {"group0", "group1", "group2", "group3"}, self.config
        )
        responses = app.responses_for_groups(self.dataset, "eval", {"group0", "group1", "group2", "group3"})
        first = app.BayesianPreferenceModel(self.dataset, "P1")
        second = app.BayesianPreferenceModel(self.dataset, "P1")
        first.fit(responses)
        second.fit(responses)
        np.testing.assert_allclose(first.mean, second.mean)
        np.testing.assert_allclose(first.covariance, second.covariance)

    def test_entropy_and_bald_use_different_objectives(self):
        questions = [{"id": "a-low-variance"}, {"id": "b-high-variance"}]
        entropy_choice = app.QuestionSelector("Predictive Entropy", {}).select(
            questions, self.dataset, FakePosterior(), random.Random(1), np.random.default_rng(1)
        )
        bald_choice = app.QuestionSelector("BALD", {"bald_samples": 100}).select(
            questions, self.dataset, FakePosterior(), random.Random(1), np.random.default_rng(1)
        )
        self.assertEqual(entropy_choice["id"], "a-low-variance")
        self.assertEqual(bald_choice["id"], "b-high-variance")

    def test_experiments_use_same_data_budget_and_do_not_leak_eval_answers(self):
        first_out = self.root / "first"
        summary = app.run_experiments(self.dataset, self.config, first_out)
        self.assertGreater(summary["predictions"], 0)
        with (first_out / "training_log.csv").open(newline="", encoding="utf-8-sig") as file:
            training = list(csv.DictReader(file))
        p1 = next(row for row in training if row["experiment"] == "A" and row["model_or_strategy"] == "P1")
        p2 = next(row for row in training if row["experiment"] == "A" and row["model_or_strategy"] == "P2")
        self.assertEqual(p1["question_ids"], p2["question_ids"])
        self.assertNotIn("q4", p1["question_ids"])
        self.assertNotIn("q5", p1["question_ids"])

        with (first_out / "predictions.csv").open(newline="", encoding="utf-8-sig") as file:
            predictions = list(csv.DictReader(file))
        model_eval_questions = {
            arm: {
                row["question_id"]
                for row in predictions
                if row["experiment"] == "A" and row["model_or_strategy"] == arm
            }
            for arm in ("P0", "P1", "P2")
        }
        self.assertEqual(model_eval_questions["P0"], {"q4", "q5"})
        self.assertEqual(model_eval_questions["P0"], model_eval_questions["P1"])
        self.assertEqual(model_eval_questions["P1"], model_eval_questions["P2"])

        with (first_out / "question_order.csv").open(newline="", encoding="utf-8-sig") as file:
            original_order_rows = list(csv.DictReader(file))
        original_order = [
            (row["user_id"], row["model"], row["strategy"], row["seed"], row["step"], row["question_id"])
            for row in original_order_rows
        ]
        question_counts = {}
        for row in original_order_rows:
            key = (row["model"], row["strategy"], row["user_id"], row["seed"])
            question_counts[key] = question_counts.get(key, 0) + 1
            self.assertIn(row["question_id"], {"q0", "q1", "q2", "q3"})
        self.assertTrue(question_counts)
        self.assertEqual(set(question_counts.values()), {1})
        with (first_out / "metrics.csv").open(newline="", encoding="utf-8-sig") as file:
            metrics = list(csv.DictReader(file))
        for arm in ("P1+Random", "P1+Balanced Random", "P1+Predictive Entropy", "P1+BALD"):
            self.assertEqual(
                {
                    int(row["budget"])
                    for row in metrics
                    if row["experiment"] == "B"
                    and row["model_or_strategy"] == arm
                    and row["user_id"] == "eval"
                },
                {0, 1},
            )
        changed = json.loads(json.dumps(self.data))
        for response in changed["responses"]:
            if response["user_id"] == "eval" and response["question_id"] in {"q4", "q5"}:
                response["chosen_candidate_id"] = (
                    "c01" if response["chosen_candidate_id"] == "c00" else "c00"
                )
        changed_path = self.root / "changed.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        second_out = self.root / "second"
        app.run_experiments(app.load_dataset(changed_path), self.config, second_out)
        with (second_out / "question_order.csv").open(newline="", encoding="utf-8-sig") as file:
            changed_order = [
                (row["user_id"], row["model"], row["strategy"], row["seed"], row["step"], row["question_id"])
                for row in csv.DictReader(file)
            ]
        self.assertEqual(original_order, changed_order)

    def test_identical_seed_replays_same_question_order(self):
        paths = []
        for name in ("repeat1", "repeat2"):
            output = self.root / name
            app.run_experiments(self.dataset, self.config, output)
            with (output / "question_order.csv").open(newline="", encoding="utf-8-sig") as file:
                paths.append([row["question_id"] for row in csv.DictReader(file)])
        self.assertEqual(paths[0], paths[1])

    def test_online_side_mapping_assignment_and_session_isolation(self):
        assignment_path = self.root / "assignments.csv"
        assignment = app.assign_participant("eval", ["Random", "BALD"], 7, assignment_path)
        repeated = app.assign_participant("eval", ["Random", "BALD"], 999, assignment_path)
        self.assertEqual(assignment["strategy_id"], repeated["strategy_id"])
        event_path = self.root / "events.jsonl"
        result = app.record_online_response(
            self.dataset,
            event_path,
            "eval",
            assignment["strategy_id"],
            "session-1",
            "img0",
            "c01",
            "c00",
            "left",
            1200,
            question_id="q0",
            assignments_path=assignment_path,
        )
        self.assertEqual(result["chosen_candidate_id"], "c01")
        checkpoint = app.update_online_model(
            self.dataset, self.config, event_path, "eval", assignment["strategy_id"],
            "session-1", "P1", self.root / "checkpoint.json", assignment_path,
        )
        self.assertEqual(checkpoint["training_question_ids"], ["q0"])
        with self.assertRaises(app.DataError):
            app.record_online_response(
                self.dataset, event_path, "eval", "other-arm", "session-1", "img0",
                "c01", "c00", "left", 1200, question_id="q0",
                assignments_path=assignment_path,
            )

    def test_online_next_uses_only_assigned_arm_and_avoids_seen_questions(self):
        assignment_path = self.root / "assignments.csv"
        app.assign_participant("eval", ["Random", "BALD"], 7, assignment_path)
        event_path = self.root / "events.jsonl"
        first = app.select_online_question(
            self.dataset, self.config, event_path, assignment_path, "eval", "Random",
            "session-1", "P1", 7,
        )
        self.assertIn(first["question_id"], {"q0", "q1", "q2", "q3"})
        self.assertEqual(
            {first["shown_left_candidate_id"], first["shown_right_candidate_id"]},
            {first["candidate_a_id"], first["candidate_b_id"]},
        )
        app.record_online_response(
            self.dataset, event_path, "eval", "Random", "session-1",
            first["original_id"], first["shown_left_candidate_id"],
            first["shown_right_candidate_id"], "left", 1000,
            question_id=first["question_id"], assignments_path=assignment_path,
        )
        second = app.select_online_question(
            self.dataset, self.config, event_path, assignment_path, "eval", "Random",
            "session-1", "P1", 7,
        )
        self.assertNotEqual(first["question_id"], second["question_id"])

    def test_crossover_assignment_records_order_and_keeps_arm_histories_separate(self):
        assignment_path = self.root / "assignments.csv"
        sequence = app.assign_crossover("eval", ["Random", "BALD"], 19, assignment_path)
        self.assertEqual(sorted(int(row["sequence_order"]) for row in sequence), [1, 2])
        first_arm = sequence[0]["strategy_id"]
        second_arm = sequence[1]["strategy_id"]
        events_path = self.root / "events.jsonl"
        first_question = app.select_online_question(
            self.dataset, self.config, events_path, assignment_path, "eval", first_arm,
            "session-first", "P1", 19,
        )
        second_before = app.select_online_question(
            self.dataset, self.config, events_path, assignment_path, "eval", second_arm,
            "session-second", "P1", 19,
        )
        app.record_online_response(
            self.dataset, events_path, "eval", first_arm, "session-first",
            first_question["original_id"], first_question["shown_left_candidate_id"],
            first_question["shown_right_candidate_id"], "left", 1100,
            question_id=first_question["question_id"], assignments_path=assignment_path,
        )
        second_after = app.select_online_question(
            self.dataset, self.config, events_path, assignment_path, "eval", second_arm,
            "session-second", "P1", 19,
        )
        self.assertEqual(second_before["response_count"], 0)
        self.assertEqual(second_after["response_count"], 0)
        self.assertEqual(second_before["question_id"], second_after["question_id"])

    def test_blind_recommendation_outcomes_use_actual_displayed_candidate(self):
        assignment_path = self.root / "assignments.csv"
        assignment = app.assign_participant("eval", ["Random", "BALD"], 7, assignment_path)
        queue_path = self.root / "queue.csv"
        with queue_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "comparison_id", "experiment", "user_id", "strategy_left",
                    "strategy_right", "original_id", "candidate_left_id",
                    "candidate_right_id", "same_candidate",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "comparison_id": "rec-1", "experiment": "A", "user_id": "eval",
                    "strategy_left": "P0", "strategy_right": "P1", "original_id": "img4",
                    "candidate_left_id": "c00", "candidate_right_id": "c01", "same_candidate": 0,
                }
            )
        events_path = self.root / "events.jsonl"
        app.record_online_response(
            self.dataset, events_path, "eval", assignment["strategy_id"], "recommendation-session",
            "img4", "c01", "c00", "left", 900, comparison_id="rec-1",
            comparison_kind="recommendation", assignments_path=assignment_path,
            queue_path=queue_path,
        )
        rows = app.analyze_recommendation_events(
            events_path, queue_path, self.root / "recommendation-outcomes.csv"
        )
        self.assertEqual(rows[0]["left_wins"], 0)
        self.assertEqual(rows[0]["right_wins"], 1)

    def test_rejects_reversed_duplicate_question(self):
        duplicate = json.loads(json.dumps(self.data))
        duplicate["questions"].append(
            {
                "id": "duplicate",
                "original_id": "img0",
                "candidate_a_id": "c01",
                "candidate_b_id": "c00",
            }
        )
        path = self.root / "duplicate.json"
        path.write_text(json.dumps(duplicate), encoding="utf-8")
        with self.assertRaisesRegex(app.DataError, "Duplicate unordered candidate pair"):
            app.load_dataset(path)

    def test_rejects_pixel_identical_render_signatures(self):
        rendered = json.loads(json.dumps(self.data))
        for index, candidate in enumerate(rendered["candidates"]):
            candidate["render_signature"] = f"render-{index}"
        rendered["candidates"][1]["render_signature"] = rendered["candidates"][0]["render_signature"]
        path = self.root / "same-render.json"
        path.write_text(json.dumps(rendered), encoding="utf-8")
        with self.assertRaisesRegex(app.DataError, "share render_signature"):
            app.load_dataset(path)

    def test_p2_does_not_invent_missing_original_features(self):
        incomplete = json.loads(json.dumps(self.data))
        del incomplete["originals"][0]["source_features"]["brightness_mean"]
        path = self.root / "missing.json"
        path.write_text(json.dumps(incomplete), encoding="utf-8")
        dataset = app.load_dataset(path)
        encoder = app.ContextEncoder.fit_development(
            dataset, {"group0", "group1", "group2", "group3"}, self.config
        )
        self.assertFalse(encoder.available)
        self.assertIn("Missing original source feature", encoder.reason)


if __name__ == "__main__":
    unittest.main()
