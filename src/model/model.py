from abc import ABC, abstractmethod
from copy import copy
from typing import Callable, List
import torch 
from transformers import AutoTokenizer

from src.task.task import Task
from src.utils.logger import get_logger 

import time

# try to import vllm
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.sampler import Sampler
except ImportError:
    get_logger().info("VLLM not installed")

from vertexai.preview import tokenization
import google.generativeai as genai
from google.generativeai.protos import Content, Part, Model, Schema
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from tqdm import tqdm

# System prompts for different scenarios
ICLF_SYSTEM_PROMPT = """You are an useful assistant. Answer the following questions. Feedback will indicate if you answered correctly. You must answer correctly, using previous feedback to make better predictions.{task_description}"""
ICL_SYSTEM_PROMPT = """You are an useful assistant. Answer the following questions.{task_description}"""

STANDARD_SYSTEM_PROMPT = """You are an useful assistant. Answer the following questions.{task_description}"""

POTENTIALLY_NEW_ICL_SYSTEM_PROMPT = """You are an useful assistant. Answer the following questions. Feedback will indicate if you answered correctly. You must answer correctly, using previous feedback to make better predictions. Be careful to interpret the feedback correctly: just because the feedback for an answer is positive does not mean it is the correct answer in all cases. You must use the feedback to learn the correct mapping between the questions and the answers.{task_description}"""

# Supported models and their specific configurations
SUPPORTED_MODELS = ["Llama-3", "Phi-3.5", "gemini-1.5-flash", "Qwen2.5"]
MODELS_SUPPORTING_SYSTEM_MESSAGE = ["Llama-3", "Phi-3.5", "Qwen2.5"]
MODEL_TO_TERMINATION_TOKEN_STR = {
    "Llama-3": "<|eot_id|>",
    "Phi-3.5": "<|end|>",
    "Qwen2.5": "<|im_end|>"
}
GEMINI_MODEL_TO_TOKENIZER_NAME = {
    "gemini-1.5-flash": "gemini-1.5-flash-001"
}

class ModelWrapper(ABC):
    def __init__(self, model_name: str, icl: bool, icrl: bool, temperature: float, verbose: bool):
        """
        Initialize the ModelWrapper with the given parameters.

        Args:
            model_name (str): Name of the model.
            icl (bool): Whether to use in-context learning.
            icrl (bool): Whether to use in-context reinforcement learning.
            temperature (float): Temperature for sampling.
            verbose (bool): Whether to enable verbose logging.
        """
        # Make sure the model is supported
        self.model_name = model_name
        self.model_family = None
        for supported_model in SUPPORTED_MODELS:
            if supported_model in model_name:
                self.model_family = supported_model
                break
        assert self.model_family is not None, f"Model family not found for model {model_name}"
        
        # Logging
        self.verbose = verbose
        
        # Generation related attributes
        self.temperature = temperature
        self.model_prediction_prefix = ""

        # Method related attributes
        self.icl = icl
        self.icrl = icrl

        # Task related attributes
        self.task_description = None
        self.task_labels = None
        self.task_prediction_prefix = None

        # Past
        self.past_token_ids = []
        self.past_task_input_list = []
        self.past_model_prediction_list = []
        self.past_task_feedback_list = []
        self.past_task_answer_list = []

    def get_name(self):
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.model_name

    def set_task(self, task: Task):
        """
        Set the task for the model.

        Args:
            task (Task): The task to be set.
        """
        self.task_description = task.get_description()
        self.task_labels = [label for label in task.get_labels()]
        self.task_prediction_prefix = task.get_prediction_prefix()
                

    def _format_model_prediction(self, model_prediction = None):
        """
        Format the model prediction.

        Args:
            model_prediction (str, optional): The model prediction. Defaults to None.

        Returns:
            str: The formatted model prediction.
        """
        if not model_prediction:
            return f"{self.task_prediction_prefix}"
        return f"{self.task_prediction_prefix} {model_prediction}"
    
    def _format_feedback(self, feedback, model_prediction):
        """
        Format the feedback for the model prediction.

        Args:
            feedback (bool): Whether the feedback is positive or negative.
            model_prediction (str): The model prediction.

        Returns:
            str: The formatted feedback.
        """
        return f"'{model_prediction}' is the correct answer! Good job!" if feedback else f"The answer '{model_prediction}' is wrong! You can do better!"
    
    def _format_messages(self, task_input_list: List[str], model_prediction_list: List[str], task_feedback_list: List[str], task_answer_list: List[str]):
        """
        Format the messages for the model.

        Args:
            task_input_list (List[str]): List of task inputs.
            model_prediction_list (List[str]): List of model predictions.
            task_feedback_list (List[str]): List of task feedbacks.
            task_answer_list (List[str]): List of task answers.

        Returns:
            List[dict]: The formatted messages.
        """
        if self.verbose:
            get_logger().info(f"Formatting messages with task input list: {task_input_list}, model prediction list: {model_prediction_list}, task feedback list: {task_feedback_list}, task answer list: {task_answer_list}")
        messages = []
        task_description = f"\n{self.task_description}"
        system_prompt = (STANDARD_SYSTEM_PROMPT if not self.icl else (ICLF_SYSTEM_PROMPT if self.icrl else ICL_SYSTEM_PROMPT)).format(
            task_description=task_description,
        )
        i = 0
        messages.append(
            {
                "role": "system" if self.supports_system_message else "user",
                "content": system_prompt
            }
        )
        while True:
            if len(task_input_list) > i:
                # If the last message was already from the user (i.e., previous feedback or system message), append the new task input to the last message
                # Otherwise, create a new message
                if len(messages) > 0 and messages[-1]["role"] == "user":
                    messages[-1]["content"] += f"\n\n{task_input_list[i]}"
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": task_input_list[i]
                        }
                    )
            if len(model_prediction_list) > i:
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._format_model_prediction(model_prediction_list[i])
                    }
                )
            if len(task_feedback_list) > i:
                messages.append(
                    {
                        "role": "user",
                        "content": self._format_feedback(task_feedback_list[i], model_prediction_list[i])
                    }
                )
            if len(task_answer_list) > i:
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._format_model_prediction(task_answer_list[i])
                    }
                )
            i += 1
            if i >= len(task_input_list) and i >= len(model_prediction_list) and i >= len(task_feedback_list) and i >= len(task_answer_list):
                break

        return messages

    @abstractmethod
    def get_maximum_length(self):
        """
        Get the maximum length of the model input.

        Returns:
            int: The maximum length of the model input.
        """
        pass
    
    @abstractmethod
    def get_number_tokens(self, messages):
        """
        Get the number of tokens in the messages.

        Args:
            messages (List[dict]): The messages.

        Returns:
            int: The number of tokens in the messages.
        """
        pass
    
    def _update_past_messages(self, task_input_list: List[str], model_prediction_list: List[str], task_feedback_list: List[int], task_answer_list: List[str]):
        """
        Update the past messages with the given lists.

        Args:
            task_input_list (List[str]): List of task inputs.
            model_prediction_list (List[str]): List of model predictions.
            task_feedback_list (List[int]): List of task feedbacks.
            task_answer_list (List[str]): List of task answers.
        """
        self.past_task_input_list = copy(task_input_list)
        self.past_model_prediction_list = copy(model_prediction_list)
        self.past_task_feedback_list = copy(task_feedback_list)
        self.past_task_answer_list = copy(task_answer_list)
    
    def refresh_cache(self, task_input_list: List[str] = [], model_prediction_list: List[str] = [], task_feedback_list: List[int] = [], task_answer_list: List[str] = []):
        """
        Refresh the cache with the all the task inputs, model predictions, task feedbacks and task answers.
        Note that the lists should contain all the past elements as well, not just the new ones.

        Args:
            task_input_list (List[str], optional): List of task inputs. Defaults to [].
            model_prediction_list (List[str], optional): List of model predictions. Defaults to [].
            task_feedback_list (List[int], optional): List of task feedbacks. Defaults to [].
            task_answer_list (List[str], optional): List of task answers. Defaults to [].
        """
        self._update_past_messages(task_input_list, model_prediction_list, task_feedback_list, task_answer_list)


    @abstractmethod
    def predict_labels(self, task_prompts: List[str], force_verbose=False) -> List[str]:
        """
        Predict the labels of the task given some prompts.
        We assume the cache has already been taken care of, and that the prompts contain only the new part of the prompts.
        
        Args:
            task_prompts (List[str]): The prompts to predict the labels of. 
            force_verbose (bool, optional): Whether to force verbose output. Defaults to False.
        
        Returns:
            str: The predicted label.
        """
        raise NotImplementedError("Method predict_labels not implemented")
    

def load_model(model_name: str, icl: bool, icrl: bool, temperature: float, verbose: bool) -> ModelWrapper:
    """
    Load the model with the given parameters.

    Args:
        model_name (str): Name of the model.
        icl (bool): Whether to use in-context learning.
        icrl (bool): Whether to use in-context reinforcement learning.
        temperature (float): Temperature for sampling.
        verbose (bool): Whether to enable verbose logging.

    Returns:
        ModelWrapper: The loaded model.
    """
    vllm_enabled = False
    try:
        import vllm
        vllm_enabled = True
    except ImportError:
        pass

    if "gemini" in model_name:
        return GeminiAPIWrapper(model_name, icl, icrl, temperature, verbose)

    if vllm_enabled:
        return VLLMModelWrapper(model_name, icl, icrl, temperature, verbose)
        
    raise NotImplementedError("VLLM not installed")

class VLLMModelWrapper(ModelWrapper):
    def __init__(self, model_name: str, icl: bool, icrl: bool, temperature: float, verbose: bool):
        """
        Initialize the VLLMModelWrapper with the given parameters.

        Args:
            model_name (str): Name of the model.
            icl (bool): Whether to use in-context learning.
            icrl (bool): Whether to use in-context reinforcement learning.
            temperature (float): Temperature for sampling.
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(model_name, icl, icrl, temperature, verbose)

        # Load model
        model_args = {
            "model": model_name,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "distributed_executor_backend": "mp",
            "tensor_parallel_size": torch.cuda.device_count(),
        }

        if "Qwen2.5" in self.model_family:
            model_args["rope_scaling"] = {
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn"
            }

        self.model = LLM(
            **model_args
        )
        if self.verbose:
            get_logger().info(f"Model {model_name} loaded on {torch.cuda.device_count()} GPUs.")


        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 

        self._apply_family_specific_fixes()

        # Tokenization - Model related attributes
        model_termination_token_str = MODEL_TO_TERMINATION_TOKEN_STR.get(self.model_family)  
        self.model_termination_token_id = self.tokenizer.convert_tokens_to_ids(model_termination_token_str)
        assert self.tokenizer.convert_ids_to_tokens(self.model_termination_token_id) == model_termination_token_str, f"Termination token id {self.model_termination_token_id} does not correspond to termination token string {model_termination_token_str}"
        if self.verbose:
            get_logger().info(f"Model termination token string: {model_termination_token_str}. Model termination token id: {self.model_termination_token_id}")
        self.supports_system_message = self.model_family in MODELS_SUPPORTING_SYSTEM_MESSAGE
        if self.verbose:
            get_logger().info(f"Model: {model_name}. Supports system message: {self.supports_system_message}")

        # Logit processor, used for constrained decoding on the task labels
        self.logit_processor = None

    def _apply_family_specific_fixes(self):
        """
        Apply family-specific fixes to the model.
        """
        pass
        # if self.model_family == "Llama-3.1":
        #     self.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"

    def set_task(self, task: Task):
        """
        Set the task for the model.

        Args:
            task (Task): The task to be set.
        """
        super().set_task(task)

        task_labels_with_whitespaces = [f" {label}" for label in self.task_labels]
        task_labels_tokens_ids = [self.tokenizer.encode(label, add_special_tokens=False) for label in task_labels_with_whitespaces]
        
        if self.verbose:
            get_logger().info(f"Task description: {self.task_description}. Task labels (without any whitespaces): {self.task_labels}. Task prediction prefix: {self.task_prediction_prefix}. Task labels token ids: {task_labels_tokens_ids}")
            for i in range(len(self.task_labels)):
                # Show task label, show encoded task label, show task label with whitespace, show ecnoded task label with whitespace
                get_logger().info(f"Task label: {self.task_labels[i]}. Encoded task label: {self.tokenizer.encode(self.task_labels[i], add_special_tokens=False)}. Task label with whitespace: {task_labels_with_whitespaces[i]}. Encoded task label with whitespace: {task_labels_tokens_ids[i]}")
                
        self.logit_processor = self._get_logit_processor(task_labels_tokens_ids)
    
    def _get_logit_processor(self, task_labels_tokens_ids) -> Callable[[List[int], torch.Tensor], torch.Tensor]:
        """
        Get the logit processor for constrained decoding on the task labels.

        Args:
            task_labels_tokens_ids (List[List[int]]): List of token ids for each task label.

        Returns:
            Callable[[List[int], torch.Tensor], torch.Tensor]: The logit processor.
        """
        task_labels_tokens_ids_with_termination = [label_token_ids + [self.model_termination_token_id] for label_token_ids in task_labels_tokens_ids] # * 2 because there apparently vllm makes one more forward pass after the stop token is generated
        
        def logit_processor(prompt_tokens: List[int], generated_label_tokens: List[int], logits: torch.Tensor) -> torch.Tensor:
            """
            Process the logits to constrain decoding on the task labels.

            Args:
                prompt_tokens (List[int]): List of prompt tokens.
                generated_label_tokens (List[int]): List of generated label tokens.
                logits (torch.Tensor): The logits.

            Returns:
                torch.Tensor: The processed logits.
            """
           
            generated_label_tokens = list(generated_label_tokens)

            num_generated_label_tokens = len(generated_label_tokens) 

            if num_generated_label_tokens > 0 and generated_label_tokens[-1] == self.model_termination_token_id:
                return logits

            task_labels_token_ids_until_now = [label_token_ids[:num_generated_label_tokens] for label_token_ids in task_labels_tokens_ids_with_termination]

            valid_labels = [label_token_ids == generated_label_tokens for label_token_ids in task_labels_token_ids_until_now]

            token_ids_to_consider = [label_token_ids[num_generated_label_tokens] for valid_label, label_token_ids in zip(valid_labels, task_labels_tokens_ids_with_termination) if valid_label]

            # Create a boolean mask for the tokens to consider
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[token_ids_to_consider] = True

            # Set all other logits to negative infinity
            logits[~mask] = -float('inf')

            return logits

        return logit_processor
    
    def get_maximum_length(self):
        """
        Get the maximum length of the model input.

        Returns:
            int: The maximum length of the model input.
        """
        return self.model.llm_engine.model_config.max_model_len

    def get_number_tokens(self, messages):
        """
        Get the number of tokens in the messages.

        Args:
            messages (List[dict]): The messages.

        Returns:
            int: The number of tokens in the messages.
        """
        messages_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        ) + self.model_prediction_prefix + self._format_model_prediction()
        return len(self.tokenizer.tokenize(messages_str))
    
    def predict_labels(self, task_prompts: List[str], generation_seed, force_verbose=False) -> List[str]:
        """
        Predict the labels of the task given some prompts.
        We assume the cache has already been taken care of, and that the prompts contain only the new part of the prompts.
        
        Args:
            task_prompts (List[str]): The prompts to predict the labels of. 
            generation_seed (int): The seed for generation.
            force_verbose (bool, optional): Whether to force verbose output. Defaults to False.
        
        Returns:
            List[str]: The predicted labels.
        """
        assert self.task_labels is not None, "Task labels not set"
        assert self.logit_processor is not None, "Logit processor not set"

        full_prompts = [
            self.tokenizer.apply_chat_template(
                self._format_messages(self.past_task_input_list + [task_prompt], self.past_model_prediction_list, self.past_task_feedback_list, self.past_task_answer_list), 
                    tokenize=False, 
                    add_generation_prompt=True
            ) + self.model_prediction_prefix + self._format_model_prediction()
            for task_prompt in task_prompts
        ]

        if self.verbose or force_verbose:
            get_logger().info(f"Predicting label for prompts: '{full_prompts}'")

        # get_logger().info(f"Stopping token id: {self.model_termination_token_id}")

        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            logits_processors=[self.logit_processor],
            stop_token_ids=[self.model_termination_token_id],
            n=1,
            seed=generation_seed,
            best_of=1
        )


        outputs = self.model.generate(
            prompts=full_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,

        )
        predicted_labels = [output.outputs[0].text.lstrip() for output in outputs]


        if self.verbose or force_verbose:
            # Given each prompt, show the prompt and the label
            for i in range(len(task_prompts)):
                get_logger().info(f"Predicted label for prompt `{task_prompts[i]}`: '{predicted_labels[i]}'")

        # Assert the predicted labels are in the task labels
        for predicted_label in predicted_labels:
            assert predicted_label in self.task_labels, f"Predicted label {predicted_label} not in task labels {self.task_labels}"

        return predicted_labels

class TrieNode:
    def __init__(self):
        """
        Initialize a TrieNode.
        """
        self.children = {}
        self.is_end = False

class VLLMModelTokenizerOnlyWrapper(ModelWrapper):
    def __init__(self, model_name: str, icl: bool, icrl: bool, temperature: float, verbose: bool):
        """
        Initialize the VLLMModelTokenizerOnlyWrapper with the given parameters.

        Args:
            model_name (str): Name of the model.
            icl (bool): Whether to use in-context learning.
            icrl (bool): Whether to use in-context reinforcement learning.
            temperature (float): Temperature for sampling.
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(model_name, icl, icrl, temperature, verbose)

        # Load model
        self.model = None
        if self.verbose:
            get_logger().info("Model not loaded")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 

        self._apply_family_specific_fixes()

        # Tokenization - Model related attributes
        model_termination_token_str = MODEL_TO_TERMINATION_TOKEN_STR.get(self.model_family)  
        self.model_termination_token_id = self.tokenizer.convert_tokens_to_ids(model_termination_token_str)

        assert self.tokenizer.convert_ids_to_tokens(self.model_termination_token_id) == model_termination_token_str, f"Termination token id {self.model_termination_token_id} does not correspond to termination token string {model_termination_token_str}"
        if self.verbose:
            get_logger().info(f"Model termination token string: {model_termination_token_str}. Model termination token id: {self.model_termination_token_id}")
        self.supports_system_message = self.model_family in MODELS_SUPPORTING_SYSTEM_MESSAGE
        if self.verbose:
            get_logger().info(f"Model: {model_name}. Supports system message: {self.supports_system_message}")

        # Logit processor, used for constrained decoding on the task labels
        self.logit_processor = None

        # Initialize the trie for past model calls
        self.past_model_calls_trie = TrieNode()

    def _apply_family_specific_fixes(self):
        """
        Apply family-specific fixes to the model.
        """
        pass
    
    def get_maximum_length(self):
        """
        Get the maximum length of the model input.

        Returns:
            int: The maximum length of the model input.
        """
        raise NotImplementedError("Maximum length not implemented")

    def get_number_tokens(self, messages):
        """
        Get the number of tokens in the messages.

        Args:
            messages (List[dict]): The messages.

        Returns:
            int: The number of tokens in the messages.
        """
        messages_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) + self.model_prediction_prefix + self._format_model_prediction()
        return len(self.tokenizer.tokenize(messages_str))

    def predict_labels(self, task_prompts: List[str], generation_seed, force_verbose=False) -> List[int]:
        """
        Predict the number of new tokens for the given task prompts.
        
        Args:
            task_prompts (List[str]): The prompts to predict the number of new tokens for.
            generation_seed (int): The seed for generation.
            force_verbose (bool, optional): Whether to force verbose output. Defaults to False.
        
        Returns:
            List[int]: The predicted number of new tokens for each prompt.
        """
        assert self.task_labels is not None, "Task labels not set"

        full_prompts = [
            self.tokenizer.apply_chat_template(
                self._format_messages(
                    self.past_task_input_list + [task_prompt],
                    self.past_model_prediction_list,
                    self.past_task_feedback_list,
                    self.past_task_answer_list
                ), 
                tokenize=False, 
                add_generation_prompt=True
            ) + self.model_prediction_prefix + self._format_model_prediction()
            for task_prompt in task_prompts
        ]

        if self.verbose or force_verbose:
            get_logger().info(f"Predicting label for prompts: '{full_prompts}'")

        # Tokenize prompts once and store
        tokenized_prompts = []
        new_tokens_counts = []
        for prompt in full_prompts:
            prompt_tokens = self.tokenizer.tokenize(prompt)
            # get_logger().info(f"Prompt: '{repr(prompt)}'")
            tokenized_prompts.append(prompt_tokens)
            prefix_length = self._find_longest_prefix_length(prompt_tokens)
            new_tokens_counts.append(len(prompt_tokens) - prefix_length)

        # Store all prefixes in trie
        for prompt_tokens in tokenized_prompts:
            self._store_all_prefixes(prompt_tokens)

        if self.verbose or force_verbose:
            for i, task_prompt in enumerate(task_prompts):
                get_logger().info(
                    f"Predicted number of new tokens for prompt `{task_prompts[i]}`: '{new_tokens_counts[i]}'"
                )

        return new_tokens_counts

    def _find_longest_prefix_length(self, tokens: List[str]) -> int:
        """
        Find the length of the longest prefix in the trie.

        Args:
            tokens (List[str]): The tokens to search for.

        Returns:
            int: The length of the longest prefix.
        """
        node = self.past_model_calls_trie
        length = 0
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                length += 1
            else:
                break
        return length

    def _store_all_prefixes(self, tokens: List[str]):
        """
        Store all prefixes of the given tokens in the trie.

        Args:
            tokens (List[str]): The tokens to store.
        """
        node = self.past_model_calls_trie
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True

class GeminiAPIWrapper(ModelWrapper):
    def __init__(self, model_name: str, icl: bool, icrl: bool, temperature: float, verbose: bool):
        super().__init__(model_name, icl, icrl, temperature, verbose)

        genai.configure(api_key=os.environ["API_KEY"])

        # Model will be loaded when the task is set
        self.model = None
        self.system_prompt = None
        self.model_maximum_length = None

        self.tokenizer_name = GEMINI_MODEL_TO_TOKENIZER_NAME[self.model_family]

        # Logit processor, used for constrained decoding on the task labels
        self.logit_processor = None

        # Cost debugging
        self.total_metadata = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "cached_content_token_count": 0
        }

        self.max_retries = 3
        self.retry_delay = 30

        self.tokenizer = tokenization.get_tokenizer_for_model(self.tokenizer_name)

    def _format_messages(self, task_input_list: List[str], model_prediction_list: List[str], task_feedback_list: List[str], task_answer_list: List[str]):
        if self.verbose:
            get_logger().info(f"Formatting messages with task input list: {task_input_list}, model prediction list: {model_prediction_list}, task feedback list: {task_feedback_list}, task answer list: {task_answer_list}")
        messages = []
        i = 0
        while True:
            if len(task_input_list) > i:
                # If the last message was already from the user (i.e., previous feedback or system message), append the new task input to the last message
                # Otherwise, create a new message
                if len(messages) > 0 and messages[-1]["role"] == "user":
                    messages[-1]["parts"][0]["text"] += f"\n\n{task_input_list[i]}"
                else:
                    messages.append(
                        {
                            "role": "user",
                            "parts": [{"text": task_input_list[i]}]
                        }
                    )
            if len(model_prediction_list) > i:
                messages.append(
                    {
                        "role": "model",
                        "parts": [{"text": self._format_model_prediction(self.label_to_model_output[model_prediction_list[i]])}]
                    }
                )
            if len(task_feedback_list) > i:
                messages.append(
                    {
                        "role": "user",
                        "parts": [{"text": self._format_feedback(task_feedback_list[i], model_prediction_list[i])}]
                    }
                )
            if len(task_answer_list) > i:
                messages.append(
                    {
                        "role": "model",
                        "parts": [{"text": self._format_model_prediction(task_answer_list[i])}]
                    }
                )
            i += 1
            if i >= len(task_input_list) and i >= len(model_prediction_list) and i >= len(task_feedback_list) and i >= len(task_answer_list):
                break

        return messages


    def set_task(self, task: Task):
        super().set_task(task)

        # Set correct system message 
        task_description = f"\n{self.task_description}"
        self.system_prompt = (STANDARD_SYSTEM_PROMPT if not self.icl else (ICLF_SYSTEM_PROMPT if self.icrl else ICL_SYSTEM_PROMPT)).format(
            task_description=task_description,
        )
        enum = [f"{self.task_prediction_prefix} {label}" for label in self.task_labels]


            
        self.model_output_to_label = {enum_elem: label for enum_elem, label in zip(enum, self.task_labels)}
        self.label_to_model_output = {label: label for enum_elem, label in zip(enum, self.task_labels)}

        response_schema = {
            "type": "STRING",
            "enum": enum,
        }

        self.generation_config = genai.GenerationConfig(
            candidate_count=1,
            temperature=self.temperature,
            response_mime_type="text/x.enum",
            response_schema=response_schema,
        )

        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config=self.generation_config,
            system_instruction=self.system_prompt
        )    
    
    def get_maximum_length(self):
        if self.model_maximum_length is None:
            model = genai.get_model(f"models/{self.model_name}")
            self.model_maximum_length = model.input_token_limit

        return self.model_maximum_length

    def get_number_tokens(self, messages):
        number_tokens = self.tokenizer.count_tokens(messages).total_tokens
        return number_tokens
    
    def predict_labels(self, task_prompts: List[str], generation_seed: int, force_verbose=False): # seed can't be used in Gemini API, currently
        assert self.model is not None, "Task not set"

        messages_list = [
            self._format_messages(self.past_task_input_list + [task_prompt], self.past_model_prediction_list, self.past_task_feedback_list, self.past_task_answer_list)
            for task_prompt in task_prompts
        ]

        predicted_labels = []
        # if len(messages_list) > 0:
        #     cache_tokens = self.get_number_tokens(messages_list[0][:-1])
            # if cache_tokens > 32000:
            #     self.num_cache_tokens += cache_tokens
            #     messages_list = [messages[-1:] for messages in messages_list]
        if self.verbose or force_verbose:
            get_logger().info(f"Predicting label for prompts:")
            for i, prompt in enumerate(task_prompts):
                get_logger().info(f"Prompt {i+1}:")
                get_logger().info(f"- '{prompt}'")
                get_logger().info("")  # empty line for better separation
            get_logger().info(f"Messages list:")
            for i, messages in enumerate(messages_list):
                get_logger().info(f"Messages for Prompt {i+1}:")
                get_logger().info(f"- {messages}")
                get_logger().info("")  # empty line for better separation

        for messages in tqdm(messages_list, leave=False, desc="Predicting labels"):    
            for retry_count in range(self.max_retries):
                try:
                    response = self.model.generate_content(
                        messages,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        })
                    
                    predicted_label = self.model_output_to_label[response.text.strip()]
                    predicted_labels.append(predicted_label)
                    
                    # Update metadata
                    metadata = response.usage_metadata
                    self.total_metadata["prompt_token_count"] += metadata.prompt_token_count
                    self.total_metadata["candidates_token_count"] += metadata.candidates_token_count
                    self.total_metadata["total_token_count"] += metadata.total_token_count
                    self.total_metadata["cached_content_token_count"] += metadata.cached_content_token_count
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry_count < self.max_retries - 1:  # Don't sleep on last attempt
                        if self.verbose or force_verbose:
                            get_logger().warning(f"API request failed (attempt {retry_count + 1}/{self.max_retries}): {str(e)}")
                            get_logger().info(f"Waiting {self.retry_delay} seconds before retrying...")
                            # Show question and response
                            get_logger().info(f"Question: {messages}")
                            get_logger().info(f"Response: {response}")

                

                        # If the reason is PROHIBITED_CONTENT, we can't retry
                        if 'response' in locals() and hasattr(response, "prompt_feedback") and hasattr(response.prompt_feedback, "block_reason"):
                            # Show block reason
                            get_logger().error(f"Prompt feedback block reason: {response.prompt_feedback.block_reason}")
                            get_logger().error("Prompt feedback indicates prohibited content. In this case the prompt is unsafe to the model. We answer with a default value.")

                            # There is this inappropriate in the one specific prompt. In this case, we just answer with label 0 (wrong label) so that will never be used again.
                            if "inappropriate_content_here_to_modify_each_time" in messages[-1]["parts"][0]["text"]:
                                predicted_labels.append(self.task_labels[0])
                                break
                        
                        time.sleep(self.retry_delay)


                    else:
                        get_logger().error(f"API request failed after {self.max_retries} attempts: {str(e)}")
                        raise

        get_logger().info(f"Total metadata: {self.total_metadata}")

        if self.verbose or force_verbose:
            # Given each prompt, show the prompt and the label
            for i in range(len(task_prompts)):
                get_logger().info(f"Predicted label for prompt {task_prompts[i]}: '{predicted_labels[i]}'")

        # Assert the predicted labels are in the task labels
        for predicted_label in predicted_labels:
            assert predicted_label in self.task_labels, f"Predicted label {predicted_label} not in task labels {self.task_labels}"

        return predicted_labels
