"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import ast
import asyncio
import copy
import dataclasses
import json
import logging
import os
import re
import string
import time
import weakref
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from openai import AsyncOpenAI
import requests

from open_instruct import context_window_checker, logger_utils
from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.IFEvalG import instructions_registry
from open_instruct.judge_utils import EXTRACTOR_MAP, JUDGE_PROMPT_MAP, PRICE_PER_MILLION_TOKENS, build_messages
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from open_instruct.rubrics.prompts import RUBRIC_SCORING_PROMPT
from open_instruct.rubrics.run_utils import extract_json_from_response
from open_instruct.utils import extract_final_answer

logger = logger_utils.setup_logger(__name__)

logging.getLogger("cost_calculator").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclasses.dataclass
class VerifierConfig:
    """For now this config exists to support LMJudgeVerifer, can be expanded to support other verifers"""

    @classmethod
    def from_args(cls, *arg_sources) -> "VerifierConfig":
        """
        Create a VerifierConfig from multiple argument sources by automatically matching field names.
        Only fields that exist in both the sources and VerifierConfig will be passed through.
        Later sources override earlier ones if they have the same field.
        """
        verifier_fields = {f.name for f in dataclasses.fields(cls)}

        matching_kwargs = {}
        for source in arg_sources:
            if source is None:
                continue
            for field_name in verifier_fields:
                if hasattr(source, field_name):
                    matching_kwargs[field_name] = getattr(source, field_name)

        return cls(**matching_kwargs)


@dataclasses.dataclass
class LMJudgeVerifierConfig(VerifierConfig):
    # judge args
    llm_judge_model: str
    llm_judge_max_tokens: int
    llm_judge_max_context_length: int
    llm_judge_temperature: float
    llm_judge_timeout: int
    seed: int
    llm_judge_base_url: str | None = None
    llm_judge_api_key_env: str = "OPENAI_API_KEY"
    llm_judge_api_key: str | None = None
    llm_judge_prompt_template: str | None = None
    llm_judge_prompt_template_file: str | None = None
    llm_judge_use_full_response: bool = False

    def resolved_base_url(self) -> str | None:
        return self.llm_judge_base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    def resolved_api_key(self) -> str:
        if self.llm_judge_api_key:
            return self.llm_judge_api_key
        env_name = self.llm_judge_api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise ValueError(
                f"Missing OpenAI-compatible judge API key. Set {env_name} or pass --llm_judge_api_key."
            )
        return api_key


@dataclasses.dataclass
class CodeVerifierConfig(VerifierConfig):
    code_api_url: str
    code_max_execution_time: float
    code_pass_rate_reward_threshold: float
    code_apply_perf_penalty: bool


@dataclasses.dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: str | None = None


@dataclasses.dataclass
class MaxLengthVerifierConfig(VerifierConfig):
    max_length_verifier_max_length: int = 32768


class VerifierFunction(ABC):
    """
    Base class for all verifier functions that evaluate model predictions against ground truth.

    Each verifier function takes a prediction and compares it to a ground truth label,
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config: VerifierConfig | None = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return VerifierConfig

    @abstractmethod
    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query
            rollout_state (Optional[dict]): Rollout state dict (rewards, step_count, done) for env verifiers.

        Returns:
            VerificationResult
        """

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Asynchronous version of __call__. By default, it runs the synchronous __call__ in a thread pool.
        Subclasses can override this method for truly asynchronous implementation.

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query.
            rollout_state (Optional[dict]): Rollout state dict for env verifiers.

        Returns:
            VerificationResult
        """
        # Run the synchronous __call__ in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.__call__(tokenized_prediction, prediction, label, query, rollout_state)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


# small helper to optionally remove thinking section + answer output.
# assumes a certain format, so might not always be useful.
# we don't always need this -- for example, math evaluations just extract a final
# number, so we don't need to remove the thinking section.
def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


class GSM8KVerifier(VerifierFunction):
    """
    Verifier for GSM8K tasks that extracts the last number from the prediction
    and compares it (case-insensitively) to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("gsm8k", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        # Preserve explicit signs on both decimals and integers when extracting the final answer.
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response)
        extracted = numbers[-1] if numbers else response
        score = float(str(extracted).lower() == str(label).lower())
        return VerificationResult(score=score)


class MathVerifier(VerifierFunction):
    """
    Verifier for math problems.

    Attempts several extraction methods (boxed answers, Minerva format,
    last LaTeX answer) and compares the extracted answers to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        raw_answer = prediction
        all_answers = []

        # Attempt extraction from \boxed{}.
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        if boxed_answer is not None:
            all_answers.append(boxed_answer)

        # Attempt extraction via Minerva format.
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)

        # Attempt extraction from the last LaTeX-formatted answer.
        if not all_answers:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)

        # Fallback to the full output.
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
            # also provide original string in case normalization fails
            all_answers.append(prediction)

        # Compare each candidate answer to the ground truth.
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)


class StrictMathVerifier(VerifierFunction):
    """
    Strict verifier for math problems using only the Minerva format extraction.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("strict_math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        raw_answer = prediction
        all_answers = []
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)


class IFEvalVerifier(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint(s) are a list of constraint ids.
    This list is found under the key "instruction_id" in the ground_truth dict.

    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("ifeval", weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | dict,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        instruction_dict = instructions_registry.INSTRUCTION_DICT
        constraint_dict = ast.literal_eval(label)
        constraint_dict = constraint_dict[0]
        if isinstance(constraint_dict, str):
            constraint_dict = json.loads(constraint_dict)
        answer = remove_thinking_section(prediction)
        instruction_keys = constraint_dict["instruction_id"]
        args_list = constraint_dict["kwargs"]
        rewards = []
        if len(prediction) == 0 or len(answer) == 0:
            logger.warning("Empty prediction received for IFEvalVerifier.")
            return VerificationResult(score=0.0)
        for instruction_key, args in zip(instruction_keys, args_list):
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}
            instruction_cls = instruction_dict[instruction_key]
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)
            if prediction.strip() and instruction_instance.check_following(answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return VerificationResult(score=sum(rewards) / max(len(rewards), 1))


class IFEvalVerifierOld(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint may be a JSON string or a dictionary containing a key
    'func_name' used to lookup the evaluation function.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("ifeval_old", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | dict,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        constraint = label
        answer = remove_thinking_section(prediction)
        if isinstance(constraint, str):
            constraint = json.loads(constraint)
        if "func_name" not in constraint:
            logger.warning("Constraint missing 'func_name': %s", constraint)
            return VerificationResult(score=0.0)
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        if not constraint:
            return VerificationResult(score=float(func(answer)))
        return VerificationResult(score=float(func(answer, **non_none_args)))


def normalize_answer(s: str) -> str:
    """
    Normalize the answer by lowercasing, removing punctuation, articles,
    and extra whitespace.

    Based on:
    https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0, "precision": 0, "recall": 0}
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}


class FlanVerifier(VerifierFunction):
    """
    Verifier for Flan tasks that extracts the answer after "The answer is:"
    and compares it to the ground truth after normalization.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("flan", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        answer_string = prediction.split("The answer is: ")[-1].strip()
        score = float(normalize_answer(answer_string) == normalize_answer(label))
        return VerificationResult(score=score)


class StringMatcherVerifier(VerifierFunction):
    """
    Verifier for tasks that require string matching.

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("string_matcher", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        if "<answer>" not in prediction or "</answer>" not in prediction:
            return VerificationResult(score=0.0)
        # extract out of answer tag
        answer_string = prediction.split("<answer>")[-1].split("</answer>")[0]
        # normalize
        score = float(normalize_answer(answer_string) == normalize_answer(label))
        return VerificationResult(score=score)


class F1Verifier(VerifierFunction):
    """
    Verifier that computes the string F1 score between the prediction and the label.

    The label can be a single string or a list of strings. If a list is provided,
    the maximum F1 score across all labels is returned.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("string_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | list[str],
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        prediction = remove_thinking_section(prediction)
        labels: list[str] = label if isinstance(label, list) else [label]
        score = max(f1_score(prediction, str(lab))["f1"] for lab in labels)
        return VerificationResult(score=score)


class PuzzleMatcherVerifier(VerifierFunction):
    """
    Verifier for Puzzle tasks that require string matching (exact matching).

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("puzzle", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        # remove answer tags from the prediction
        prediction = remove_thinking_section(prediction)
        score = float(normalize_answer(prediction) == normalize_answer(label))
        return VerificationResult(score=score)


class ReSearchVerifierF1(VerifierFunction):
    """
    Verifier from ReSearch paper (https://arxiv.org/abs/2503.19470)
    Uses F1 score + format. If format is achieved but f1 is 0, returns 0.1. Otherwise returns F1.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        self.answer_start_tag = "<finish>"
        self.answer_end_tag = "</finish>"
        super().__init__("re_search_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            label = label.strip()
        # extract answer
        if self.answer_start_tag not in prediction and self.answer_end_tag not in prediction:
            return VerificationResult(score=0.0)
        answer_string = prediction.split(self.answer_start_tag)[-1].split(self.answer_end_tag)[0]
        # check answer non-empty
        if not answer_string:
            return VerificationResult(score=0.0)
        # if label is list, max over labels
        if isinstance(label, list):
            f1 = max(f1_score(answer_string, str(lab))["f1"] for lab in label)
        else:
            label = str(label)  # safety.
            f1 = f1_score(answer_string, label)["f1"]
        # if f1 is 0, but format is correct, return 0.1
        if f1 == 0:
            return VerificationResult(score=0.1)
        # otherwise return f1
        return VerificationResult(score=f1)


class R1SearchVerifier(VerifierFunction):
    """
    Verifier based on the Search-R1 paper (https://github.com/PeterGriffinJin/Search-R1).
    Uses normalized exact match: returns 1.0 if answer matches any label, else 0.0.
    Answer extraction is done via a case-insensitive regex on <finish>...</finish> tags.
    """

    # Precompile a case-insensitive regex to extract answer text
    TAG_PATTERN = re.compile(r"<finish>(.*?)</finish>", re.IGNORECASE | re.DOTALL)

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__(name="re_search", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | list[str],
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        # 1. Parse JSON label safely
        parsed_labels: list | str
        try:
            parsed = json.loads(label)
            parsed_labels = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat label as raw string or list-of-strings
            parsed_labels = label if isinstance(label, list) else [str(label).strip()]

        # 2. Extract answer between tags
        match = self.TAG_PATTERN.search(prediction)
        if not match:
            logging.debug("No <finish> tags found in prediction")
            return VerificationResult(score=0.0)

        answer_text = match.group(len(match.groups())).strip()
        if not answer_text:
            logging.debug("Extracted answer is empty after stripping whitespace")
            return VerificationResult(score=0.0)

        # 3. Normalize once
        norm_answer = normalize_answer(answer_text)

        # 4. Compare against each label
        for lbl in parsed_labels:
            try:
                lbl_str = normalize_answer(str(lbl))
                if norm_answer == lbl_str:
                    return VerificationResult(score=1.0)
            except Exception as e:
                logging.warning(f"Error normalizing label '{lbl}': {e}")

        # 5. No match found
        return VerificationResult(score=0.0)


class MaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        desired_length = float(label)
        # return absolute difference between the length of the prediction and the max length
        # make sure to disallow negative rewards
        length_diff = abs(len(tokenized_prediction) - desired_length)
        score = 1 - (length_diff / self.verifier_config.max_length_verifier_max_length)
        return VerificationResult(score=score)

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.
        Returns:
            type: The VerifierConfig class or its subclass
        """
        return MaxLengthVerifierConfig


class UpToMaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("up_to_max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        desired_length = float(label)
        length_diff = len(tokenized_prediction) - desired_length
        # if we were too short, its fine! return 1.0
        if length_diff < 0:
            return VerificationResult(score=1.0)
        # if we were too long, return the difference
        # make sure to disallow negative rewards
        score = 1 - (length_diff / self.verifier_config.max_length_verifier_max_length)
        return VerificationResult(score=score)

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.
        Returns:
            type: The VerifierConfig class or its subclass
        """
        return MaxLengthVerifierConfig


class LMJudgeVerifier(VerifierFunction):
    """
    Verifier that uses a language model's judgement to score a response.
    """

    _clients: dict[tuple[str | None, str], AsyncOpenAI] = {}

    def __init__(
        self,
        judge_type: str,
        verifier_config: LMJudgeVerifierConfig,
        *,
        name: str | None = None,
        prompt_template: str | None = None,
        extractor=None,
    ) -> None:
        super().__init__(name or f"general-{judge_type}", verifier_config=verifier_config, weight=1.0)
        self.prompt_template = prompt_template or JUDGE_PROMPT_MAP[judge_type]
        self.extractor = extractor or EXTRACTOR_MAP[judge_type]
        self._custom_prompt = prompt_template is not None

    @classmethod
    def _client_for_values(cls, base_url: str | None, api_key: str, timeout: int) -> AsyncOpenAI:
        key = (base_url, api_key)
        client = cls._clients.get(key)
        if client is None:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            cls._clients[key] = client
        return client

    @classmethod
    def _client_for_config(cls, verifier_config: LMJudgeVerifierConfig) -> AsyncOpenAI:
        return cls._client_for_values(
            verifier_config.resolved_base_url(),
            verifier_config.resolved_api_key(),
            verifier_config.llm_judge_timeout,
        )

    @staticmethod
    def _read_prompt_template(path: str) -> str:
        with open(path, encoding="utf-8") as file_obj:
            return file_obj.read()

    @classmethod
    def custom_prompt_template(cls, verifier_config: LMJudgeVerifierConfig) -> str | None:
        if verifier_config.llm_judge_prompt_template_file:
            return cls._read_prompt_template(verifier_config.llm_judge_prompt_template_file)
        return verifier_config.llm_judge_prompt_template

    def format_prompt(self, prediction: str, label: str, query: str | None) -> str:
        final_answer = extract_final_answer(prediction)
        judge_output = prediction if self.verifier_config.llm_judge_use_full_response else final_answer
        values = {
            "input": query or "",
            "output": judge_output,
            "prediction": prediction,
            "final_answer": final_answer,
            "label": label,
        }
        if self._custom_prompt:
            rendered = self.prompt_template
            for key, value in values.items():
                rendered = rendered.replace("{" + key + "}", str(value))
            return rendered
        return self.prompt_template.format(**values)

    def parse_completion(self, completion):
        """
        Extract reasoning and score from an OpenAI API completion response.

        Args:
            completion: The OpenAI API completion response object

        Returns:
            tuple: (reasoning, score) extracted from the response
        """
        reasoning = ""
        score = 0.0

        if not completion:
            print("No completion received from the model.")
            return reasoning, score

        try:
            # remove anything between <think> and </think> including the tags using regex
            pattern = r"<think>\s*.*?\s*</think>\s*"
            content = re.sub(pattern, "", completion.choices[0].message.content, flags=re.DOTALL)
            content = content.replace("<answer>", "").replace("</answer>", "")
            reasoning, score = self.extractor(content)

        except Exception as e:
            print(f"Error processing model response: {str(e)}")
            if hasattr(completion, "choices") and completion.choices is not None and len(completion.choices) > 0:
                print(f"Response content: {getattr(completion.choices[0].message, 'content', 'No content available')}")

        return reasoning, score

    def get_cost(self, response, model: str):
        """
        Get the cost of the response.
        """
        model_name = model.split("/")[-1]
        model_name = model_name.replace("-standard", "")  # azure OAI models have -standard in the name
        if getattr(response, "usage", None) is None:
            return 0.0
        return (
            PRICE_PER_MILLION_TOKENS.get(model_name, {}).get("input", 0) * response.usage.prompt_tokens
            + PRICE_PER_MILLION_TOKENS.get(model_name, {}).get("output", 0) * response.usage.completion_tokens
        ) / 1_000_000

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Asynchronous version of __call__ that properly handles the async OpenAI client.
        """
        prompt = self.format_prompt(prediction=prediction, label=label, query=query)

        max_retries = 3  # for rate limits
        retry_delay = 1.0

        for attempt in range(max_retries):
            # judges the quality of a response
            try:
                messages = build_messages(prompt)

                # Check if the request would exceed context window
                if not context_window_checker.check_context_window_limit(
                    messages=messages,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    model_name=self.verifier_config.llm_judge_model,
                    max_context_length=self.verifier_config.llm_judge_max_context_length,  # Adjust based on your model
                    safety_margin=150,
                ):
                    # Try to truncate messages to fit
                    messages = context_window_checker.truncate_messages_to_fit_context(
                        messages=messages,
                        max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                        model_name=self.verifier_config.llm_judge_model,
                        max_context_length=self.verifier_config.llm_judge_max_context_length,
                        safety_margin=200,
                    )

                    # Check again after truncation
                    if not context_window_checker.check_context_window_limit(
                        messages=messages,
                        max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                        model_name=self.verifier_config.llm_judge_model,
                        max_context_length=self.verifier_config.llm_judge_max_context_length,
                        safety_margin=150,
                    ):
                        logger.error("Cannot fit request within context window even after truncation.")
                        return VerificationResult(score=0.0, cost=0.0, reasoning="Error: Context window exceeded")
                # end of Faeze's context window check
                client = self._client_for_config(self.verifier_config)
                response = await client.chat.completions.create(
                    model=self.verifier_config.llm_judge_model,
                    messages=messages,
                    temperature=self.verifier_config.llm_judge_temperature,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    timeout=self.verifier_config.llm_judge_timeout,
                )
                reasoning, score = self.parse_completion(response)
                cost = self.get_cost(response, self.verifier_config.llm_judge_model)
                # normalize score to be between 0 and 1
                return VerificationResult(score=score, cost=cost, reasoning=reasoning)

            except Exception as e:
                logger.warning(f"LLM judge attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM judge failed after {max_retries} attempts. Returning default score of 0.0")
                    return VerificationResult(score=0.0, cost=0.0, reasoning=f"Error: {str(e)}")
                else:
                    await asyncio.sleep(retry_delay * (2**attempt))  # Exponential backoff
        return VerificationResult(score=0.0, cost=0.0, reasoning="Unknown error after all retries.")

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Evaluates the prediction based on an LLM's judgement.

        Args:
            tokenized_prediction (List[int]): Tokenized representation of the prediction (unused).
            prediction (str): The model output string that was judged.
            label (str): An optional reference for the judge. Can be a reference answer or a rubric.
        Returns:
            float: The calculated reward (parsed_rating)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
        except RuntimeError:
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))

    @classmethod
    async def cleanup_all_clients(cls):
        """
        Close OpenAI-compatible judge clients.
        """
        for client in cls._clients.values():
            try:
                await client.close()
            except Exception:
                logger.exception("Failed to close OpenAI judge client")
        cls._clients.clear()
        return None

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return LMJudgeVerifierConfig


DEEPSEEKMATH_V2_PROOF_JUDGE_PROMPT = r"""
## Instruction

Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0
- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria

---

Here is your task input:

## Problem
{question}

## Solution
{proof}
""".strip()


DEEPSEEKMATH_V2_META_JUDGE_PROMPT = r"""
You are given a "problem", "solution", and "solution evaluation", and you need to assess the whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":
1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.

2. You need to analyze the content of the "solution evaluation" from three aspects:

Step Restatement: In the "solution evaluation", certain behaviors of the "solution" may be restated. You need to return to the original text of the "solution" and check whether the "solution" actually has these behaviors mentioned in the "solution evaluation".

Defect Analysis: "solution evaluation" may point out errors or defects in the "solution". You need to carefully analyze whether the mentioned errors and defects are indeed valid.

Expression Analysis: Whether the "solution evaluation"'s expressions are accurate.

Score Analysis: Whether the final score given by the "solution evaluation" matches the defects it found. You need to analyze according to the scoring rules given above.

3. The most important part is **defect analysis**: In this part, your core task is to check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable. In other words, any positive components about the "solution" in the "solution evaluation", regardless of whether they are reasonable, are not within your evaluation scope.

- For example: If the "solution evaluation" says that a certain conclusion in the "solution" is correct, but actually this conclusion is incorrect, then you do not need to care about this point. All parts that the "solution evaluation" considers correct do not belong to your evaluation scope.

- Specifically: If the "solution evaluation" believes that the "solution" is completely accurate and has not found any errors or defects, then regardless of whether the "solution" itself is actually accurate, even if there are obvious errors, you should still consider its analysis of errors to be reasonable.
**Importantly**, for defects found by the "solution evaluation", you need to analyze two points simultaneously:

- whether this defect actually exists
- whether the "solution evaluation"'s analysis of this defect is accurate

These two aspects constitute the analysis of defects.

4. About **expression analysis**, if there are certain expression errors in the "solution evaluation", even minor errors in details, you need to identify them. However, please note that identifying incorrect steps in the "solution" as correct steps does not constitute an **expression error**.

In practice, expression errors include but are not limited to:

- If the "solution evaluation" identifies some reasoning step(s) in the "solution" as incorrect, then it cannot further indicate that subsequent conclusion(s) depending on those reasoning step(s) are wrong, but can only indicate that subsequent conclusion(s) are "not rigorously demonstrated."
- Typos and calculation errors made by "solution evaluation"
- Inaccurate restatement of content from "solution"

5. Finally, you need to present your analysis of the "solution evaluation" in your output and also rate its quality based on the rules below:

First, if there is at least one unreasonable defect among the defects found by the "solution evaluation", then you only need to do **defect analysis**:

- If all defects found by the "solution evaluation" are unreasonable, then you should rate it with \(0\)
- If some defects found by the "solution evaluation" are reasonable and some are unreasonable, then your rating should be \(0.5\)

Next, if the "solution evaluation" points out no errors or defects, or all defects found by the evaluation are reasonable, then you should do the following things:

- Analyze whether "expression errors" exist in the "solution evaluation" (**expression analysis**) or whether "solution evaluation" gives a wrong score according to the rules for "solution evaluation" (**score analysis**). If yes, you should rate the "solution evaluation" with \(0.5\); if no, your rating should be \(1\)

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I will rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating of the "solution evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.

---

Here is your task input:

## Problem
{question}

## Solution
{proof}

## Solution Evaluation
{proof analysis}
""".strip()


@dataclasses.dataclass
class DeepSeekMathV2VerifierConfig(VerifierConfig):
    llm_judge_model: str = "gpt-5.5"
    llm_judge_base_url: str | None = None
    llm_judge_api_key_env: str = "OPENAI_API_KEY"
    llm_judge_api_key: str | None = None
    deepseekmath_v2_proof_judge_model: str | None = None
    deepseekmath_v2_meta_judge_model: str | None = None
    deepseekmath_v2_base_url: str | None = None
    deepseekmath_v2_api_key_env: str = "OPENAI_API_KEY"
    deepseekmath_v2_api_key: str | None = None
    deepseekmath_v2_proof_prompt_template: str | None = None
    deepseekmath_v2_proof_prompt_template_file: str | None = None
    deepseekmath_v2_meta_prompt_template: str | None = None
    deepseekmath_v2_meta_prompt_template_file: str | None = None
    deepseekmath_v2_judge_backend: Literal["api", "local_vllm"] = "api"
    deepseekmath_v2_max_tokens: int = 92160
    deepseekmath_v2_max_context_length: int = 102400
    deepseekmath_v2_context_margin_tokens: int = 256
    deepseekmath_v2_min_completion_tokens: int = 2048
    deepseekmath_v2_temperature: float = 0.7
    deepseekmath_v2_top_p: float = 0.95
    deepseekmath_v2_timeout: int = 60
    deepseekmath_v2_extra_body_json: str | None = None
    deepseekmath_v2_proof_weight: float = 0.76
    deepseekmath_v2_self_eval_weight: float = 0.24
    deepseekmath_v2_enable_meta_verification: bool = True
    deepseekmath_v2_require_format: bool = True
    seed: int = 42

    def proof_model(self) -> str:
        return self.deepseekmath_v2_proof_judge_model or self.llm_judge_model

    def meta_model(self) -> str:
        return self.deepseekmath_v2_meta_judge_model or self.deepseekmath_v2_proof_judge_model or self.llm_judge_model

    def resolved_base_url(self) -> str | None:
        return (
            self.deepseekmath_v2_base_url
            or self.llm_judge_base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )

    def resolved_api_key(self) -> str:
        if self.deepseekmath_v2_api_key:
            return self.deepseekmath_v2_api_key
        if self.llm_judge_api_key:
            return self.llm_judge_api_key
        env_name = self.deepseekmath_v2_api_key_env or self.llm_judge_api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise ValueError(
                f"Missing OpenAI-compatible DeepSeekMath-V2 judge API key. Set {env_name} or pass "
                "--deepseekmath_v2_api_key."
            )
        return api_key

    def extra_body(self, model: str) -> dict[str, Any] | None:
        if self.deepseekmath_v2_extra_body_json:
            parsed = json.loads(self.deepseekmath_v2_extra_body_json)
            if not isinstance(parsed, dict):
                raise ValueError("--deepseekmath_v2_extra_body_json must parse to a JSON object.")
            return parsed
        base_url = self.resolved_base_url() or ""
        if "openrouter.ai" in base_url and model.startswith("deepseek/"):
            return {
                "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
                "provider": {"only": ["deepseek"], "allow_fallbacks": False},
            }
        return None


@dataclasses.dataclass
class DeepSeekMathV2ParsedResponse:
    solution: str
    self_evaluation: str
    self_score: float | None
    format_ok: bool
    format_errors: list[str]


class DeepSeekMathV2Verifier(VerifierFunction):
    """DeepSeekMath-V2-style proof reward with API or local-vLLM proof/meta judges."""

    SCORE_VALUES = (0.0, 0.5, 1.0)

    def __init__(self, verifier_config: DeepSeekMathV2VerifierConfig) -> None:
        super().__init__("deepseekmath_v2", verifier_config=verifier_config, weight=1.0)
        self.config = verifier_config
        self.proof_prompt_template = self._load_template(
            verifier_config.deepseekmath_v2_proof_prompt_template,
            verifier_config.deepseekmath_v2_proof_prompt_template_file,
            DEEPSEEKMATH_V2_PROOF_JUDGE_PROMPT,
        )
        self.meta_prompt_template = self._load_template(
            verifier_config.deepseekmath_v2_meta_prompt_template,
            verifier_config.deepseekmath_v2_meta_prompt_template_file,
            DEEPSEEKMATH_V2_META_JUDGE_PROMPT,
        )

    @staticmethod
    def _load_template(inline_template: str | None, template_file: str | None, default_template: str) -> str:
        if template_file:
            with open(template_file, encoding="utf-8") as file_obj:
                return file_obj.read()
        return inline_template or default_template

    @staticmethod
    def _render_template(template: str, values: dict[str, Any]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
        return rendered

    @classmethod
    def _normalize_score(cls, value: Any) -> float | None:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        nearest = min(cls.SCORE_VALUES, key=lambda allowed: abs(allowed - score))
        if abs(nearest - score) <= 1e-6:
            return nearest
        return None

    @classmethod
    def _extract_score(cls, text: str | None) -> float | None:
        if not text:
            return None
        boxed_matches = re.findall(r"\\boxed\s*\{+\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*\}+", text)
        for value in reversed(boxed_matches):
            parsed = cls._normalize_score(value)
            if parsed is not None:
                return parsed

        if "{" in text and "}" in text:
            obj = extract_json_from_response(text)
            if isinstance(obj, dict):
                for key in ("score", "SCORE", "rating", "RATING"):
                    if key in obj:
                        parsed = cls._normalize_score(obj[key])
                        if parsed is not None:
                            return parsed

        score_matches = re.findall(r"(?i)\b(?:score|rating)\b[^0-9+-]*([-+]?(?:\d+(?:\.\d+)?|\.\d+))", text)
        for value in reversed(score_matches):
            parsed = cls._normalize_score(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _extract_sections(prediction: str) -> tuple[str, str, list[str]]:
        errors: list[str] = []
        solution_match = re.search(r"(?im)^[ \t]*##[ \t]+Solution[ \t]*$", prediction)
        self_eval_match = re.search(r"(?im)^[ \t]*##[ \t]+Self[ \t]+Evaluation[ \t]*$", prediction)

        if not solution_match:
            errors.append("missing_solution_heading")
        if not self_eval_match:
            errors.append("missing_self_evaluation_heading")
        if solution_match and self_eval_match and self_eval_match.start() <= solution_match.start():
            errors.append("self_evaluation_before_solution")

        if solution_match and self_eval_match and self_eval_match.start() > solution_match.start():
            solution = prediction[solution_match.end() : self_eval_match.start()].strip()
            self_evaluation = prediction[self_eval_match.end() :].strip()
        else:
            solution = prediction.strip()
            self_evaluation = ""

        if not solution:
            errors.append("empty_solution")
        if not self_evaluation:
            errors.append("empty_self_evaluation")
        return solution, self_evaluation, errors

    @classmethod
    def parse_prediction(cls, prediction: str) -> DeepSeekMathV2ParsedResponse:
        solution, self_evaluation, errors = cls._extract_sections(prediction)
        if self_evaluation and "Here is my evaluation of the solution:" not in self_evaluation:
            errors.append("missing_evaluation_phrase")
        if self_evaluation and "Based on my evaluation, the final overall score should be:" not in self_evaluation:
            errors.append("missing_score_phrase")
        self_score = cls._extract_score(self_evaluation)
        if self_score is None:
            errors.append("missing_or_invalid_boxed_self_score")
        return DeepSeekMathV2ParsedResponse(
            solution=solution,
            self_evaluation=self_evaluation,
            self_score=self_score,
            format_ok=not errors,
            format_errors=errors,
        )

    @staticmethod
    def _resolve_problem(label: Any, query: str | None) -> str:
        def problem_from_label(value: Any) -> str:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return ""
            if isinstance(value, dict):
                for key in ("query", "question", "problem", "Question", "Problem"):
                    label_value = value.get(key)
                    if label_value:
                        return str(label_value)
            return ""

        label_problem = problem_from_label(label)
        if label_problem:
            return label_problem

        if query:
            problem_match = re.search(r"(?ims)^##[ \t]+Problem[ \t]*\n(?P<problem>.*)$", query)
            if problem_match is not None:
                return problem_match.group("problem").strip()
            return query
        if isinstance(label, str):
            try:
                parsed = json.loads(label)
            except json.JSONDecodeError:
                return label
            label = parsed
        if isinstance(label, dict):
            for key in ("query", "question", "problem", "Question", "Problem"):
                value = label.get(key)
                if value:
                    return str(value)
        return "" if label is None else str(label)

    @staticmethod
    def _response_cost(response: Any, model: str) -> float:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0
        model_name = model.split("/")[-1].replace("-standard", "")
        return (
            PRICE_PER_MILLION_TOKENS.get(model_name, {}).get("input", 0) * usage.prompt_tokens
            + PRICE_PER_MILLION_TOKENS.get(model_name, {}).get("output", 0) * usage.completion_tokens
        ) / 1_000_000

    @staticmethod
    def _estimate_messages_tokens(messages: list[dict[str, str]], model: str) -> tuple[int, str]:
        try:
            encoding = context_window_checker.get_encoding_for_model(model)
            total_tokens = 0
            for message in messages:
                role_tokens = 4 if message.get("role") == "system" else 3
                total_tokens += len(encoding.encode(message.get("content", "") or "")) + role_tokens
            return total_tokens, encoding.name
        except Exception as exc:
            total_chars = sum(len(message.get("content", "") or "") for message in messages)
            logger.warning("DeepSeekMath-V2 judge token estimate failed for %s: %s; using chars/4 fallback.", model, exc)
            return max(1, total_chars // 4), "chars_per_4"

    def _effective_judge_max_tokens(
        self,
        messages: list[dict[str, str]],
        model: str,
        stage: str,
        *,
        max_context_length_override: int | None = None,
    ) -> tuple[list[dict[str, str]], int]:
        prompt_tokens, tokenizer_name = self._estimate_messages_tokens(messages, model)
        configured_context_length = self.config.deepseekmath_v2_max_context_length
        context_length = configured_context_length
        if max_context_length_override is not None:
            context_length = min(configured_context_length, max(1, max_context_length_override))
        margin = max(0, self.config.deepseekmath_v2_context_margin_tokens)
        configured_max_tokens = max(1, self.config.deepseekmath_v2_max_tokens)
        reserved_completion_tokens = min(
            configured_max_tokens,
            max(1, self.config.deepseekmath_v2_min_completion_tokens),
            max(1, context_length - margin - 1),
        )
        effective_max_tokens = min(configured_max_tokens, max(1, context_length - prompt_tokens - margin))
        truncated = False

        if prompt_tokens + margin + reserved_completion_tokens > context_length:
            messages = context_window_checker.truncate_messages_to_fit_context(
                messages=messages,
                max_completion_tokens=reserved_completion_tokens,
                model_name=model,
                max_context_length=context_length,
                safety_margin=margin,
            )
            truncated = True
            prompt_tokens, tokenizer_name = self._estimate_messages_tokens(messages, model)
            effective_max_tokens = min(configured_max_tokens, max(1, context_length - prompt_tokens - margin))

        logger.info(
            "DeepSeekMath-V2 judge budget stage=%s model=%s prompt_tokens=%d tokenizer=%s "
            "configured_max_tokens=%d reserved_completion_tokens=%d effective_max_tokens=%d context_length=%d "
            "configured_context_length=%d local_context_length=%s margin=%d prompt_chars=%d truncated=%s",
            stage,
            model,
            prompt_tokens,
            tokenizer_name,
            configured_max_tokens,
            reserved_completion_tokens,
            effective_max_tokens,
            context_length,
            configured_context_length,
            max_context_length_override,
            margin,
            sum(len(message.get("content", "") or "") for message in messages),
            truncated,
        )
        return messages, effective_max_tokens

    async def _call_judge(
        self,
        prompt: str,
        model: str,
        stage: str,
        rollout_state: dict | None = None,
    ) -> tuple[str, float]:
        local_judge = None
        if self.config.deepseekmath_v2_judge_backend == "local_vllm":
            local_judge = (rollout_state or {}).get("_local_judge_actor")
            if local_judge is None:
                raise ValueError(
                    "DeepSeekMath-V2 local_vllm judge backend requires a local vLLM actor. "
                    "This backend is only supported inside grpo_fast rollout actors."
                )
            model = getattr(local_judge, "model_name", None) or model

        messages = build_messages(prompt)
        local_context_length = None
        if local_judge is not None:
            llm_engine = getattr(local_judge, "llm_engine", None)
            model_config = getattr(llm_engine, "model_config", None)
            local_context_length = getattr(model_config, "max_model_len", None)
        messages, max_completion_tokens = self._effective_judge_max_tokens(
            messages,
            model,
            stage,
            max_context_length_override=local_context_length,
        )
        start_time = time.monotonic()
        if self.config.deepseekmath_v2_judge_backend == "local_vllm":
            client = getattr(local_judge, "client", None)
            if client is None:
                raise ValueError("DeepSeekMath-V2 local_vllm judge backend could not find actor.client.")
            request_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": self.config.deepseekmath_v2_temperature,
                "top_p": self.config.deepseekmath_v2_top_p,
                "max_tokens": max_completion_tokens,
                "timeout": self.config.deepseekmath_v2_timeout,
            }
            response = await client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = getattr(response, "usage", None)
            logger.info(
                "DeepSeekMath-V2 judge response stage=%s backend=local_vllm seconds=%.2f "
                "finish_reason=%s response_chars=%d usage_prompt_tokens=%s usage_completion_tokens=%s",
                stage,
                time.monotonic() - start_time,
                getattr(choice, "finish_reason", None),
                len(content),
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
            )
            return content, 0.0

        client = LMJudgeVerifier._client_for_values(
            self.config.resolved_base_url(),
            self.config.resolved_api_key(),
            self.config.deepseekmath_v2_timeout,
        )
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.config.deepseekmath_v2_temperature,
            "top_p": self.config.deepseekmath_v2_top_p,
            "max_completion_tokens": max_completion_tokens,
            "timeout": self.config.deepseekmath_v2_timeout,
        }
        extra_body = self.config.extra_body(model)
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body
        response = await client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        logger.info(
            "DeepSeekMath-V2 judge response stage=%s backend=api seconds=%.2f "
            "finish_reason=%s response_chars=%d usage_prompt_tokens=%s usage_completion_tokens=%s",
            stage,
            time.monotonic() - start_time,
            getattr(choice, "finish_reason", None),
            len(content),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return content, self._response_cost(response, model)

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        parsed = self.parse_prediction(prediction)
        if self.config.deepseekmath_v2_require_format and not parsed.format_ok:
            return VerificationResult(
                score=0.0,
                reasoning=json.dumps(
                    {
                        "format_ok": False,
                        "format_errors": parsed.format_errors,
                        "proof_score": None,
                        "self_score": parsed.self_score,
                        "self_eval_score": None,
                    },
                    ensure_ascii=False,
                ),
            )

        question = self._resolve_problem(label, query)
        values = {
            "question": question,
            "proof": parsed.solution,
            "solution": parsed.solution,
            "proof_analysis": parsed.self_evaluation,
            "proof analysis": parsed.self_evaluation,
            "self_evaluation": parsed.self_evaluation,
            "prediction": prediction,
            "label": label,
        }

        total_cost = 0.0
        try:
            proof_content, proof_cost = await self._call_judge(
                self._render_template(self.proof_prompt_template, values),
                self.config.proof_model(),
                "proof",
                rollout_state=rollout_state,
            )
            total_cost += proof_cost
            proof_score = self._extract_score(proof_content)
            if proof_score is None:
                return VerificationResult(
                    score=0.0,
                    cost=total_cost,
                    reasoning=json.dumps(
                        {
                            "format_ok": parsed.format_ok,
                            "format_errors": parsed.format_errors,
                            "error": "missing_or_invalid_proof_judge_score",
                            "proof_judge_response": proof_content[:1000],
                        },
                        ensure_ascii=False,
                    ),
                )

            if not self.config.deepseekmath_v2_enable_meta_verification:
                reward = proof_score if parsed.format_ok or not self.config.deepseekmath_v2_require_format else 0.0
                return VerificationResult(
                    score=reward,
                    cost=total_cost,
                    reasoning=json.dumps(
                        {
                            "format_ok": parsed.format_ok,
                            "format_errors": parsed.format_errors,
                            "proof_score": proof_score,
                            "self_score": parsed.self_score,
                            "self_eval_score": None,
                            "score_alignment": None,
                            "meta_verification_enabled": False,
                        },
                        ensure_ascii=False,
                    ),
                )

            meta_content, meta_cost = await self._call_judge(
                self._render_template(self.meta_prompt_template, values),
                self.config.meta_model(),
                "meta",
                rollout_state=rollout_state,
            )
            total_cost += meta_cost
            self_eval_score = self._extract_score(meta_content)
            if self_eval_score is None:
                self_eval_score = 0.0
            self_score = parsed.self_score if parsed.self_score is not None else 0.0
            score_alignment = max(0.0, 1.0 - abs(self_score - proof_score))
            format_multiplier = 1.0 if parsed.format_ok or not self.config.deepseekmath_v2_require_format else 0.0
            reward = format_multiplier * (
                self.config.deepseekmath_v2_proof_weight * proof_score
                + self.config.deepseekmath_v2_self_eval_weight * score_alignment * self_eval_score
            )
            reward = max(0.0, min(1.0, reward))

            return VerificationResult(
                score=reward,
                cost=total_cost,
                reasoning=json.dumps(
                    {
                        "format_ok": parsed.format_ok,
                        "format_errors": parsed.format_errors,
                        "proof_score": proof_score,
                        "self_score": parsed.self_score,
                        "self_eval_score": self_eval_score,
                        "score_alignment": score_alignment,
                        "reward": reward,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:
            logger.warning("DeepSeekMath-V2 verifier failed: %s", exc)
            return VerificationResult(score=0.0, cost=total_cost, reasoning=f"Error: {exc}")

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query, rollout_state))
        raise RuntimeError("Cannot call synchronous method from async context. Use async_call instead.")

    @classmethod
    def get_config_class(cls) -> type:
        return DeepSeekMathV2VerifierConfig


class CodeVerifier(VerifierFunction):
    """
    Verifier that executes Python code against test cases using an external API.

    The label should be a list of test cases or a JSON string representation of a list.
    The API URL should be provided during initialization.
    """

    # Class-level session cache to reuse connections
    _session_cache = weakref.WeakKeyDictionary()

    def __init__(self, verifier_config: CodeVerifierConfig) -> None:
        super().__init__("code", verifier_config=verifier_config, weight=1.0)
        self.pass_rate_reward_threshold = verifier_config.code_pass_rate_reward_threshold
        self.apply_perf_penalty = verifier_config.code_apply_perf_penalty

    def extract_python_code(self, model_output: str) -> str:
        """Extract the last code block between ``` markers from the model output."""
        # Find content between ``` markers
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, model_output, re.DOTALL)

        if not matches:
            return model_output

        # Return the last match, stripped of whitespace
        return matches[-1].strip()

    # Create a session pool for better performance
    _session_pool = None

    @classmethod
    def _get_session(cls):
        if cls._session_pool is None:
            cls._session_pool = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=100,
                pool_maxsize=100,
                max_retries=requests.adapters.Retry(
                    total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
                ),
            )
            cls._session_pool.mount("http://", adapter)
            cls._session_pool.mount("https://", adapter)
        return cls._session_pool

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Asynchronously verify code execution against test cases.

        Args:
            tokenized_prediction: Unused tokenized representation
            prediction: The model output containing Python code
            label: List of test cases or JSON string representation of a list
            query: Unused original query

        Returns:
            VerificationResult with score as the pass rate of test cases
        """
        # Extract Python code from the model output
        python_code = self.extract_python_code(prediction)

        # Test data
        payload = {
            "program": python_code,
            "tests": label,
            "max_execution_time": self.verifier_config.code_max_execution_time,
        }

        try:
            # Use connection pooling session
            session = self._get_session()

            # Calculate timeout
            http_timeout = max(30, min(300, self.verifier_config.code_max_execution_time * 10))

            # Make request in thread pool to keep it async
            def make_request():
                response = session.post(
                    self.verifier_config.code_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=http_timeout,
                )
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(make_request)
            passes = result["results"]
            pass_rate = sum(passes) / len(passes) if passes else 0.0
            score = 0.0 if pass_rate < self.pass_rate_reward_threshold else pass_rate
            if self.apply_perf_penalty and score > 0.0:
                runtimes = result["runtimes"]
                # for each runtime, multiply the percent of the timeout that was used
                multipliers = [
                    (self.verifier_config.code_max_execution_time - runtime)
                    / self.verifier_config.code_max_execution_time
                    for runtime in runtimes
                ]
                penalized_passes = [passes[i] * multipliers[i] for i in range(len(passes))]
                score = sum(penalized_passes) / len(penalized_passes)
            return VerificationResult(score=score)
        except Exception as e:
            logger.warning(f"Error verifying code sample: {e}")
            return VerificationResult(score=0.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Synchronously verify code execution against test cases.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
        except RuntimeError:
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return CodeVerifierConfig


class PassthroughVerifier(VerifierFunction):
    """Passthrough verifier for environment-only tasks.

    Returns 0.0 score — contributes nothing from the verifier side.
    Per-turn rewards from the rollout are handled by the RewardAggregator.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("passthrough", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        return VerificationResult(score=0.0)


class RewardAggregator(ABC):
    """Combines per-turn rewards into a final scalar."""

    @abstractmethod
    def __call__(self, rewards: list[float]) -> float: ...


class LastRewardAggregator(RewardAggregator):
    """Return the last reward (sparse reward envs)."""

    def __call__(self, rewards: list[float]) -> float:
        return rewards[-1] if rewards else 0.0


class SumRewardAggregator(RewardAggregator):
    """Sum all rewards (dense reward envs)."""

    def __call__(self, rewards: list[float]) -> float:
        return sum(rewards)


@dataclasses.dataclass
class RubricVerifierConfig(VerifierConfig):
    """Configuration for rubric verifier."""

    rubric_judge_model: str = "gpt-4.1"
    rubric_judge_base_url: str | None = None
    rubric_judge_api_key_env: str = "OPENAI_API_KEY"
    rubric_judge_api_key: str | None = None
    rubric_judge_max_tokens: int = 2048
    rubric_judge_temperature: float = 0.0
    rubric_judge_timeout: int = 60
    seed: int = 42

    def resolved_base_url(self) -> str | None:
        return self.rubric_judge_base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    def resolved_api_key(self) -> str:
        if self.rubric_judge_api_key:
            return self.rubric_judge_api_key
        env_name = self.rubric_judge_api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
        if not api_key:
            raise ValueError(
                f"Missing OpenAI-compatible rubric judge API key. Set {env_name} or pass --rubric_judge_api_key."
            )
        return api_key


class RubricVerifier(VerifierFunction):
    """
    Verifier that scores responses against rubrics defined in the ground truth.

    The ground truth label should be a JSON string or dict with:
    - "query": The original question
    - "rubrics": List of rubric dicts with "description" and "weight" keys

    Returns weighted average of rubric scores.

    Environment Variables:
        RUBRIC_JUDGE_MODEL: Override the LLM model used for rubric scoring.
            Defaults to RubricVerifierConfig.rubric_judge_model ("gpt-4.1").
        OPENAI_BASE_URL or OPENAI_API_BASE: OpenAI-compatible endpoint for rubric scoring.
        OPENAI_API_KEY: API key for rubric scoring unless --rubric_judge_api_key_env/--rubric_judge_api_key is used.
    """

    def __init__(self, verifier_config: RubricVerifierConfig) -> None:
        super().__init__("rubric", verifier_config=verifier_config, weight=1.0)
        self.config = verifier_config

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """Score response against all rubrics in the ground truth."""
        if isinstance(label, str):
            try:
                label = json.loads(label)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse rubric label as JSON: {label[:100]}")
                return VerificationResult(score=0.0)

        if not isinstance(label, dict):
            logger.warning(f"Rubric label is not a dict: {type(label)}")
            return VerificationResult(score=0.0)

        question = label.get("query") or label.get("Question") or query
        rubrics = label.get("rubrics", [])

        if not rubrics:
            logger.warning("No rubrics found in ground truth")
            return VerificationResult(score=0.0)

        # Extract content from <answer> tags if present, otherwise use full response.
        # Use the last match in case the model outputs multiple answer blocks.
        if answer_matches := re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL):
            response_for_scoring = answer_matches[-1].strip()
        else:
            response_for_scoring = prediction

        # Score each rubric in parallel
        async def score_rubric(rubric: dict) -> tuple[float, float]:
            description = rubric.get("description") or rubric.get("rubric_item") or rubric.get("Ingredient", "")
            weight = rubric.get("weight", 1.0)

            if not description:
                logger.warning("Rubric with empty description found, skipping.")
                return 0.0, 0.0

            user_prompt = f"<question>{question}</question>\n<response>{response_for_scoring}</response>\n<criterion>{description}</criterion>"

            try:
                model_name = os.environ.get("RUBRIC_JUDGE_MODEL", self.config.rubric_judge_model)
                client = LMJudgeVerifier._client_for_values(
                    self.config.resolved_base_url(),
                    self.config.resolved_api_key(),
                    self.config.rubric_judge_timeout,
                )
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": RUBRIC_SCORING_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.rubric_judge_temperature,
                    max_completion_tokens=self.config.rubric_judge_max_tokens,
                    timeout=self.config.rubric_judge_timeout,
                )
                resp = response.choices[0].message.content or ""

                obj = extract_json_from_response(resp)
                if obj and isinstance(obj, dict) and "score" in obj:
                    score = float(obj["score"]) / 2.0  # Normalize from 0-2 to 0-1
                    return score, weight
            except Exception as e:
                logger.warning(f"Error scoring rubric: {e}")

            return 0.0, weight

        # Run all rubric scoring in parallel
        results = await asyncio.gather(*[score_rubric(r) for r in rubrics])

        # Compute weighted average
        total_weight = sum(abs(w) for _, w in results)
        if total_weight == 0:
            return VerificationResult(score=0.0)

        weighted_sum = sum(s * w for s, w in results)
        final_score = weighted_sum / total_weight

        return VerificationResult(score=final_score)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """Synchronous wrapper for async_call."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Cannot call synchronous method from async context. Use async_call instead.")
        except RuntimeError:
            # No loop is running, which is expected for a sync call
            pass
        return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query, rollout_state))

    @classmethod
    def get_config_class(cls) -> type:
        return RubricVerifierConfig


def build_all_verifiers(args, streaming_config=None) -> dict[str, VerifierFunction]:
    """
    Build all verifiers with the given configs.
    Args:
        args: The main Args object
        streaming_config: Optional StreamingDataLoaderConfig for additional fields
    """
    verifiers: dict[str, VerifierFunction] = {}
    for subclass in VerifierFunction.__subclasses__():
        if subclass == LMJudgeVerifier:
            continue

        verifier_config = subclass.get_config_class().from_args(args, streaming_config)
        instance = subclass(verifier_config)
        verifiers[instance.name.lower()] = instance

        # add the code_stdio verifier
        if subclass == CodeVerifier:
            stdio_config = copy.deepcopy(verifier_config)
            stdio_config.code_api_url = stdio_config.code_api_url.replace("/test_program", "/test_program_stdio")
            instance = CodeVerifier(stdio_config)
            instance.name = "code_stdio"
            verifiers["code_stdio"] = instance

    for judge_type in JUDGE_PROMPT_MAP:
        instance = LMJudgeVerifier(judge_type, LMJudgeVerifierConfig.from_args(args, streaming_config))
        verifiers[instance.name.lower()] = instance

    judge_config = LMJudgeVerifierConfig.from_args(args, streaming_config)
    custom_prompt_template = LMJudgeVerifier.custom_prompt_template(judge_config)
    if custom_prompt_template is not None:
        instance = LMJudgeVerifier(
            "quality_rubric",
            judge_config,
            name="llm_judge",
            prompt_template=custom_prompt_template,
            extractor=EXTRACTOR_MAP["quality_rubric"],
        )
        verifiers[instance.name.lower()] = instance

    # if we have remap arg, remap!
    if streaming_config and streaming_config.remap_verifier:
        remap = streaming_config.remap_verifier.split("=")
        assert len(remap) == 2, "Remap must be in the format old_name=new_name"
        old_name, new_name = remap
        # map so that the old name calls the new verifier
        assert new_name.lower() in verifiers, f"{new_name} not found in verifiers during remapping"
        verifiers[old_name.lower()] = verifiers[new_name.lower()]

    return verifiers


# special case, we use this outside our general verifier loop.
def soft_format_reward_func(responses: list[str], reward_scale: float = 1.0) -> list[float]:
    """
    Check if the completion has a specific format defined by a pattern.

    Returns a list of rewards scaled by reward_scale.
    """
    pattern = r".*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [reward_scale if match else 0.0 for match in matches]


async def cleanup_all_llm_judge_clients():
    """
    Cleanup function to properly close all LLM judge clients before shutdown.
    """
    await LMJudgeVerifier.cleanup_all_clients()


async def apply_verifiable_reward(
    reward_fn_mapping: dict[str, VerifierFunction],
    responses: list,
    decoded_responses: list[str],
    ground_truths: list,
    datasets: list[str],
    reward_mult: float = 1.0,
    queries: list[str] | None = None,
    rollout_states: list[dict | None] | None = None,
    local_judge: Any | None = None,
):
    if queries is None:
        queries = [None] * len(responses)
    if rollout_states is None:
        rollout_states = [None] * len(responses)
    if local_judge is not None:
        rollout_states = [
            {**(state or {}), "_local_judge_actor": local_judge}
            for state in rollout_states
        ]

    async_tasks = []
    task_metadata = []

    for i, (tok_prediction, prediction, ground_truth, dataset, query, rollout_state) in enumerate(
        zip(responses, decoded_responses, ground_truths, datasets, queries, rollout_states)
    ):
        ground_truth_list = [ground_truth] if isinstance(ground_truth, str) else ground_truth
        dataset_list = [dataset] if isinstance(dataset, str) else dataset
        assert len(ground_truth_list) == len(dataset_list), "Ground truth and dataset list lengths do not match."

        for gt, ds in zip(ground_truth_list, dataset_list):
            reward_func = reward_fn_mapping.get(ds.lower())
            if reward_func is None:
                logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
                continue

            task = reward_func.async_call(
                tokenized_prediction=tok_prediction,
                prediction=prediction,
                label=gt,
                query=query,
                rollout_state=rollout_state,
            )
            async_tasks.append(task)
            task_metadata.append({"response_idx": i, "dataset": reward_func.name, "reward_weight": reward_func.weight})

    if async_tasks:
        reward_results = await asyncio.gather(*async_tasks)
        logger.debug(f"Applied {len(reward_results)} ground truth rewards in parallel")
    else:
        reward_results = []

    response_rewards = [0] * len(responses)
    response_per_func_rewards = [{} for _ in range(len(responses))]

    for result, metadata in zip(reward_results, task_metadata):
        response_idx = metadata["response_idx"]
        dataset = metadata["dataset"]
        reward_weight = metadata["reward_weight"]

        score = result.score if hasattr(result, "score") else result
        weighted_reward = reward_mult * score * reward_weight

        response_rewards[response_idx] += weighted_reward
        response_per_func_rewards[response_idx][dataset] = (
            response_per_func_rewards[response_idx].get(dataset, 0) + weighted_reward
        )

    return response_rewards, response_per_func_rewards


@dataclasses.dataclass
class RewardConfig:
    """Configuration for reward function computation."""

    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    apply_verifiable_reward: bool = True
    verification_reward: float = 10.0
    non_stop_penalty: bool = False
    non_stop_penalty_value: float = -10.0
    only_reward_good_outputs: bool = False
    additive_format_reward: bool = False
    verifier_functions: dict[str, VerifierFunction] = dataclasses.field(default_factory=dict)
    reward_aggregator: Literal["last", "sum"] = "last"
    """How to combine per-turn rewards: 'last' (use last turn reward) or 'sum' (sum all rewards across turns)."""

    def build(self) -> Callable:
        """Build and return the reward function."""
        aggregator: RewardAggregator = {"last": LastRewardAggregator(), "sum": SumRewardAggregator()}[
            self.reward_aggregator
        ]

        async def reward_fn(
            responses: list,
            decoded_responses: list[str],
            ground_truths: list[Any],
            datasets: list[str],
            finish_reasons: list[str],
            infos,
            queries: list[str] | None = None,
            local_judge: Any | None = None,
        ) -> tuple[list[float], dict[str, Any]]:
            timeouts = infos.timeouts
            tool_errors = infos.tool_errors
            tool_outputs = infos.tool_outputs
            tool_calleds = infos.tool_calleds
            rollout_states = infos.rollout_states or [{}] * len(decoded_responses)
            good_outputs = [
                len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
                for i in range(len(tool_outputs))
            ]
            scores = [0.0] * len(decoded_responses)
            metrics: dict[str, Any] = {}
            format_scores: list[float] = []

            if self.apply_r1_style_format_reward:
                format_scores = soft_format_reward_func(decoded_responses, self.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["val/format_scores"] = np.array(format_scores).mean()

            if self.apply_verifiable_reward:
                verifiable_rewards, per_func_rewards = await apply_verifiable_reward(
                    self.verifier_functions,
                    responses,
                    decoded_responses,
                    ground_truths,
                    datasets,
                    reward_mult=self.verification_reward,
                    queries=queries,
                    rollout_states=rollout_states,
                    local_judge=local_judge,
                )
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")

                for i in range(len(verifiable_rewards)):
                    if not self.only_reward_good_outputs or (good_outputs[i] and self.only_reward_good_outputs):
                        turn_rewards = list(rollout_states[i].get("rewards", []))
                        verifier_score = verifiable_rewards[i]

                        if turn_rewards:
                            turn_rewards[-1] += verifier_score
                        else:
                            turn_rewards = [verifier_score]

                        raw_score = aggregator(turn_rewards)

                        if self.apply_r1_style_format_reward and self.additive_format_reward:
                            scores[i] = raw_score + scores[i]
                        elif self.apply_r1_style_format_reward and not self.additive_format_reward:
                            scores[i] = raw_score if format_scores[i] == 1 else 0
                        else:
                            scores[i] = raw_score

                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                per_func_lists: dict[str, list] = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"objective/{key}_reward"] = np_value.mean()
                    metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()

            if self.non_stop_penalty:
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = self.non_stop_penalty_value

            return scores, metrics

        return reward_fn
