#!/usr/bin/env python3
"""
Test script for verifier functionality in Python
"""

import asyncio
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from parameterized import parameterized

from open_instruct import ground_truth_utils
from open_instruct.ground_truth_utils import (
    DeepSeekMathV2Verifier,
    DeepSeekMathV2VerifierConfig,
    F1Verifier,
    GSM8KVerifier,
    LMJudgeVerifier,
    LMJudgeVerifierConfig,
    PuzzleMatcherVerifier,
    RubricVerifier,
    RubricVerifierConfig,
    cleanup_all_llm_judge_clients,
)


class TestPuzzleMatcherVerifier(unittest.TestCase):
    """Test suite for PuzzleMatcherVerifier"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.verifier = PuzzleMatcherVerifier()

    @parameterized.expand(
        [
            ("simple_match", "The answer is 42", "answer is 42", 1.0),
            ("with_thinking_tags", "<think>Let me solve this</think>Paris", "paris", 1.0),
            ("with_answer_tags", "<answer>New York City!</answer>", "new york city", 1.0),
            ("should_fail", "Wrong answer", "correct answer", 0.0),
            (
                "complex_example",
                "<think>This is about geography</think><answer>The capital of France is Paris.</answer>",
                "capital of france is paris",
                1.0,
            ),
        ]
    )
    def test_basic_scenarios(self, name, prediction, label, expected_score):
        """Test basic puzzle matcher scenarios from quick_test"""
        result = self.verifier([], prediction, label)
        self.assertEqual(result.score, expected_score)

    @parameterized.expand(
        [
            # Basic matching tests
            ("exact_match_numbers", "42", "42", 1.0),
            ("exact_match_text", "hello world", "hello world", 1.0),
            ("case_insensitive", "Hello World", "hello world", 1.0),
            # Tests with thinking tags
            ("thinking_tags_match", "<think>Let me think about this...</think>42", "42", 1.0),
            ("thinking_tags_text_match", "<think>This is complex</think>hello world", "hello world", 1.0),
            ("thinking_tags_no_match", "<think>Analysis...</think>Wrong Answer", "42", 0.0),
            # Tests with answer tags
            ("answer_tags_match", "<answer>42</answer>", "42", 1.0),
            ("answer_tags_text_match", "<answer>hello world</answer>", "hello world", 1.0),
            ("answer_tags_no_match", "<answer>wrong</answer>", "42", 0.0),
            # Combined tags tests
            ("both_tags_match", "<think>Thinking...</think><answer>42</answer>", "42", 1.0),
            (
                "both_tags_text_match",
                "<think>Let me solve this step by step</think><answer>hello world</answer>",
                "hello world",
                1.0,
            ),
            # Punctuation and articles tests
            ("remove_articles_punctuation", "The answer is 42!", "answer is 42", 1.0),
            ("remove_article_a", "A simple test.", "simple test", 1.0),
            ("remove_punctuation", "Hello, world!", "hello world", 1.0),
            # Whitespace tests
            ("normalize_whitespace", "  hello   world  ", "hello world", 1.0),
            ("replace_tabs_newlines", "hello\tworld\n", "hello world", 1.0),
            # Non-matching tests
            ("numbers_no_match", "42", "43", 0.0),
            ("text_no_match", "hello", "world", 0.0),
            ("empty_vs_nonempty", "", "42", 0.0),
            # English examples
            ("capital_city", "<answer>London</answer>", "london", 1.0),
            ("animal_with_article", "<think>Animal question</think>The elephant", "elephant", 1.0),
            ("scientist_name", "<answer>Albert Einstein</answer>", "albert einstein", 1.0),
            ("literature_reference", "Romeo and Juliet by Shakespeare", "romeo and juliet by shakespeare", 1.0),
            ("country_name", "<answer>United States of America</answer>", "united states of america", 1.0),
        ]
    )
    def test_puzzle_matcher_scenarios(self, name, prediction, label, expected_score):
        """Test various puzzle matcher scenarios"""
        result = self.verifier([], prediction, label)
        self.assertEqual(
            result.score, expected_score, f"Failed for {name}: prediction='{prediction}', label='{label}'"
        )


class TestF1Verifier(unittest.TestCase):
    """Test suite for F1Verifier"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.verifier = F1Verifier()

    @parameterized.expand(
        [
            # Basic F1 tests with single string label
            ("exact_match", "hello world", "hello world", 1.0),
            ("partial_match", "hello world", "hello", 2 / 3),  # precision=0.5, recall=1.0, f1=2/3
            ("no_match", "hello world", "goodbye", 0.0),
            # Thinking section removal
            ("with_thinking", "<think>Let me think...</think>hello world", "hello world", 1.0),
            ("with_thinking_partial", "<think>Analysis</think>hello world", "hello", 2 / 3),
            # Answer tag removal
            ("with_answer_tags", "<answer>hello world</answer>", "hello world", 1.0),
            # Combined tags
            ("both_tags", "<think>Thinking...</think><answer>hello world</answer>", "hello world", 1.0),
        ]
    )
    def test_single_label(self, name, prediction, label, expected_score):
        """Test F1 verifier with single string label"""
        result = self.verifier([], prediction, label)
        self.assertAlmostEqual(
            result.score,
            expected_score,
            places=5,
            msg=f"Failed for {name}: prediction='{prediction}', label='{label}'",
        )

    @parameterized.expand(
        [
            # List of labels - should return max F1
            ("first_matches_best", "hello world", ["hello world", "goodbye"], 1.0),
            ("second_matches_best", "hello world", ["goodbye", "hello world"], 1.0),
            ("partial_matches", "hello world", ["hello", "world"], 2 / 3),  # both have same F1
            ("none_match_well", "hello world", ["foo", "bar", "baz"], 0.0),
            # Single element list should behave same as string
            ("single_element_list", "hello world", ["hello world"], 1.0),
            # With thinking section
            ("list_with_thinking", "<think>hmm</think>hello world", ["goodbye", "hello world"], 1.0),
        ]
    )
    def test_list_labels(self, name, prediction, labels, expected_score):
        """Test F1 verifier with list of labels (should return max)"""
        result = self.verifier([], prediction, labels)
        self.assertAlmostEqual(
            result.score,
            expected_score,
            places=5,
            msg=f"Failed for {name}: prediction='{prediction}', labels={labels}",
        )


class TestGSM8KVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = GSM8KVerifier()

    @parameterized.expand(
        [
            ("negative_integer", "Therefore the answer is -3", "-3", 1.0),
            ("positive_integer", "Therefore the answer is +7", "+7", 1.0),
            ("negative_decimal", "Final answer: -3.5", "-3.5", 1.0),
            ("boxed_negative_integer", r"The result is \\boxed{-3}", "-3", 1.0),
            ("wrong_sign", "Therefore the answer is 3", "-3", 0.0),
        ]
    )
    def test_signed_number_extraction(self, _name, prediction, label, expected_score):
        result = self.verifier([], prediction, label)
        self.assertEqual(result.score, expected_score)


def _make_openai_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class TestLMJudgeVerifier(unittest.TestCase):
    def setUp(self):
        LMJudgeVerifier._clients.clear()
        self.verifier = LMJudgeVerifier(
            "quality",
            LMJudgeVerifierConfig(
                llm_judge_model="azure/gpt-4o-mini-standard",
                llm_judge_base_url="https://example.test/v1",
                llm_judge_api_key="test-key",
                llm_judge_max_tokens=256,
                llm_judge_max_context_length=4096,
                llm_judge_temperature=0.0,
                llm_judge_timeout=30,
                seed=17,
            ),
        )

    def test_async_call_uses_openai_client_and_preserves_retry_and_cost(self):
        response = _make_openai_response('{"REASONING":"clear","SCORE":7}', prompt_tokens=10, completion_tokens=5)
        create_mock = AsyncMock(side_effect=[RuntimeError("temporary"), response])
        close_mock = AsyncMock()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)), close=close_mock)
        sleep_mock = AsyncMock()

        with (
            patch("open_instruct.ground_truth_utils.AsyncOpenAI", return_value=client) as client_cls,
            patch(
                "open_instruct.ground_truth_utils.context_window_checker.check_context_window_limit", return_value=True
            ),
            patch("open_instruct.ground_truth_utils.asyncio.sleep", sleep_mock),
        ):
            result = asyncio.run(
                self.verifier.async_call(
                    tokenized_prediction=[],
                    prediction="<answer>final answer</answer>",
                    label="reference",
                    query="What is the answer?",
                )
            )

        self.assertAlmostEqual(result.score, 0.7)
        self.assertEqual(result.reasoning, "clear")
        self.assertAlmostEqual(result.cost, 0.0000045)
        self.assertEqual(create_mock.await_count, 2)
        self.assertEqual(sleep_mock.await_count, 1)
        client_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/v1",
            timeout=30,
        )
        self.assertEqual(create_mock.await_args_list[-1].kwargs["model"], "azure/gpt-4o-mini-standard")
        self.assertEqual(create_mock.await_args_list[-1].kwargs["max_completion_tokens"], 256)
        self.assertNotIn("num_retries", create_mock.await_args_list[-1].kwargs)
        self.assertNotIn("fallbacks", create_mock.await_args_list[-1].kwargs)

    def test_custom_prompt_template_uses_full_response_placeholders(self):
        config = LMJudgeVerifierConfig(
            llm_judge_model="judge-model",
            llm_judge_base_url="https://example.test/v1",
            llm_judge_api_key="test-key",
            llm_judge_prompt_template="Question: {input}\nOutput: {output}\nRef: {label}",
            llm_judge_use_full_response=True,
            llm_judge_max_tokens=256,
            llm_judge_max_context_length=4096,
            llm_judge_temperature=0.0,
            llm_judge_timeout=30,
            seed=17,
        )
        verifier = LMJudgeVerifier(
            "quality_rubric",
            config,
            name="llm_judge",
            prompt_template=config.llm_judge_prompt_template,
        )

        prompt = verifier.format_prompt(
            prediction="<think>proof idea</think>\n<answer>final</answer>",
            label="reference proof",
            query="prove this",
        )

        self.assertIn("Question: prove this", prompt)
        self.assertIn("<think>proof idea</think>", prompt)
        self.assertIn("Ref: reference proof", prompt)

    def test_cleanup_helpers_are_safe_noops(self):
        self.assertIsNone(asyncio.run(LMJudgeVerifier.cleanup_all_clients()))
        self.assertIsNone(asyncio.run(cleanup_all_llm_judge_clients()))


class TestRubricVerifier(unittest.TestCase):
    def setUp(self):
        LMJudgeVerifier._clients.clear()
        self.verifier = RubricVerifier(
            RubricVerifierConfig(
                rubric_judge_model="rubric-model",
                rubric_judge_base_url="https://example.test/v1",
                rubric_judge_api_key="test-key",
                rubric_judge_max_tokens=128,
                rubric_judge_temperature=0.0,
                rubric_judge_timeout=30,
            )
        )

    def test_async_call_uses_openai_client(self):
        response = _make_openai_response('{"score": 2}')
        create_mock = AsyncMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)),
            close=AsyncMock(),
        )
        label = {"query": "prove this", "rubrics": [{"description": "correct proof", "weight": 1.0}]}

        with patch("open_instruct.ground_truth_utils.AsyncOpenAI", return_value=client) as client_cls:
            result = asyncio.run(
                self.verifier.async_call(
                    tokenized_prediction=[],
                    prediction="Here is the proof.",
                    label=label,
                    query="prove this",
                )
            )

        self.assertEqual(result.score, 1.0)
        client_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.test/v1",
            timeout=30,
        )
        self.assertEqual(create_mock.await_args.kwargs["model"], "rubric-model")
        self.assertEqual(create_mock.await_args.kwargs["max_completion_tokens"], 128)


class TestDeepSeekMathV2Verifier(unittest.TestCase):
    def setUp(self):
        LMJudgeVerifier._clients.clear()
        self.config = DeepSeekMathV2VerifierConfig(
            llm_judge_model="judge-model",
            llm_judge_base_url="https://example.test/v1",
            llm_judge_api_key="test-key",
            deepseekmath_v2_max_tokens=128,
            deepseekmath_v2_max_context_length=4096,
            deepseekmath_v2_temperature=0.0,
            deepseekmath_v2_timeout=30,
        )
        self.verifier = DeepSeekMathV2Verifier(self.config)

    @staticmethod
    def formatted_prediction(self_score: str = "0.5") -> str:
        return f"""## Solution
This is a proof attempt.

## Self Evaluation

Here is my evaluation of the solution:
The proof has one minor gap.

Based on my evaluation, the final overall score should be:
\\boxed{{{self_score}}}
"""

    def test_async_call_computes_weighted_proof_and_meta_reward(self):
        create_mock = AsyncMock(
            side_effect=[
                _make_openai_response('{"reasoning":"proof ok","score":1}'),
                _make_openai_response('{"reasoning":"self eval partly ok","score":0.5}'),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)),
            close=AsyncMock(),
        )

        with (
            patch("open_instruct.ground_truth_utils.AsyncOpenAI", return_value=client) as client_cls,
            patch(
                "open_instruct.ground_truth_utils.context_window_checker.check_context_window_limit",
                return_value=True,
            ),
        ):
            result = asyncio.run(
                self.verifier.async_call(
                    tokenized_prediction=[],
                    prediction=self.formatted_prediction("0.5"),
                    label={"problem": "prove this"},
                    query=None,
                )
            )

        self.assertAlmostEqual(result.score, 0.82)
        payload = ground_truth_utils.json.loads(result.reasoning)
        self.assertEqual(payload["proof_score"], 1.0)
        self.assertEqual(payload["self_score"], 0.5)
        self.assertEqual(payload["self_eval_score"], 0.5)
        self.assertEqual(payload["score_alignment"], 0.5)
        self.assertEqual(create_mock.await_count, 2)
        client_cls.assert_called_once_with(api_key="test-key", base_url="https://example.test/v1", timeout=30)
        self.assertEqual(create_mock.await_args_list[0].kwargs["model"], "judge-model")

    def test_async_call_returns_zero_without_required_format(self):
        create_mock = AsyncMock()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)),
            close=AsyncMock(),
        )

        with patch("open_instruct.ground_truth_utils.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                self.verifier.async_call(
                    tokenized_prediction=[],
                    prediction="This proof has no required headings.",
                    label={"problem": "prove this"},
                    query=None,
                )
            )

        self.assertEqual(result.score, 0.0)
        self.assertEqual(create_mock.await_count, 0)
        payload = ground_truth_utils.json.loads(result.reasoning)
        self.assertFalse(payload["format_ok"])
        self.assertIn("missing_solution_heading", payload["format_errors"])

    def test_async_call_can_run_proof_score_only(self):
        config = dataclasses.replace(self.config, deepseekmath_v2_enable_meta_verification=False)
        verifier = DeepSeekMathV2Verifier(config)
        create_mock = AsyncMock(return_value=_make_openai_response('{"reasoning":"minor gap","score":0.5}'))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)),
            close=AsyncMock(),
        )

        with (
            patch("open_instruct.ground_truth_utils.AsyncOpenAI", return_value=client),
            patch(
                "open_instruct.ground_truth_utils.context_window_checker.check_context_window_limit",
                return_value=True,
            ),
        ):
            result = asyncio.run(
                verifier.async_call(
                    tokenized_prediction=[],
                    prediction=self.formatted_prediction("0.5"),
                    label={"problem": "prove this"},
                    query=None,
                )
            )

        self.assertEqual(result.score, 0.5)
        self.assertEqual(create_mock.await_count, 1)

    def test_default_prompts_follow_deepseekmath_v2_appendix_format(self):
        self.assertIn("Here is my evaluation of the solution:", ground_truth_utils.DEEPSEEKMATH_V2_PROOF_JUDGE_PROMPT)
        self.assertIn(
            "Based on my evaluation, the final overall score should be:",
            ground_truth_utils.DEEPSEEKMATH_V2_PROOF_JUDGE_PROMPT,
        )
        self.assertIn(r"\\boxed{{...}}", ground_truth_utils.DEEPSEEKMATH_V2_PROOF_JUDGE_PROMPT)
        self.assertIn(
            'Here is my analysis of the "solution evaluation":',
            ground_truth_utils.DEEPSEEKMATH_V2_META_JUDGE_PROMPT,
        )
        self.assertIn(
            'Based on my analysis, I will rate the "solution evaluation" as:',
            ground_truth_utils.DEEPSEEKMATH_V2_META_JUDGE_PROMPT,
        )
        self.assertIn("{proof analysis}", ground_truth_utils.DEEPSEEKMATH_V2_META_JUDGE_PROMPT)


class TestIFEvalVerifierEmptyInstructions(unittest.TestCase):
    """Regression test for PR #1655: IFEvalVerifier crashed with
    ZeroDivisionError when the constraint's instruction_id list was empty."""

    def test_empty_instruction_list_returns_zero_score(self):
        verifier = ground_truth_utils.IFEvalVerifier()
        label = str([{"instruction_id": [], "kwargs": []}])
        result = verifier(tokenized_prediction=[1, 2, 3], prediction="some non-empty response", label=label)
        self.assertEqual(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()
