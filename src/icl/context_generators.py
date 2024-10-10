from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple
import numpy as np

from src.utils.logger import get_logger


class ContextGenerator(ABC):
    """
    Abstract base class for context generators.
    """
    def __init__(self, max_examples: int, verbose: bool):
        """
        Initialize the context generator.

        Args:
            max_examples (int): Maximum number of examples to keep.
            verbose (bool): Whether to print verbose logs.
        """
        self.max_examples = max_examples
        self.verbose = verbose

    @abstractmethod
    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        pass

    def get_context_additional_metrics(self):
        """
        Get additional metrics for the context.

        Returns:
            None
        """
        return None


class RandomContextGenerator(ContextGenerator, ABC):
    """
    Abstract base class for random context generators.
    """
    def __init__(self, max_examples: int, p_keep: float, verbose: bool):
        """
        Initialize the random context generator.

        Args:
            max_examples (int): Maximum number of examples to keep.
            p_keep (float): Probability of keeping an example.
            verbose (bool): Whether to print verbose logs.
        """
        super().__init__(max_examples, verbose)
        self.p_keep = p_keep
        self.rng = None

    def set_random_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed.
        """
        self.rng = np.random.default_rng(seed)

    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        assert self.rng is not None

        # Flip coin for all examples in parallel
        training_keep_examples = self.rng.binomial(1, self.p_keep, size=len(training_task_prompts))
        indices_to_keep = np.where(training_keep_examples == 1)[0]
        indices_to_keep = self.rng.permutation(indices_to_keep)
        indices_to_keep = self.choose_indices_to_keep(indices_to_keep)

        if self.verbose:
            get_logger().info(f"Keeping {len(indices_to_keep)} examples out of {len(training_task_prompts)}: {indices_to_keep}")

        # We only keep the examples to keep
        context_task_prompts = [training_task_prompts[i] for i in indices_to_keep]
        context_model_predictions = [training_model_predictions[i] for i in indices_to_keep]
        context_task_feedbacks = [training_task_feedbacks[i] for i in indices_to_keep]
        context_task_answers = [training_task_answers[i] for i in indices_to_keep]
        context_task_accuracies = [training_task_accuracies[i] for i in indices_to_keep]

        self.rng = None

        return context_task_prompts, context_model_predictions, context_task_feedbacks, context_task_answers, context_task_accuracies
    
    @abstractmethod
    def choose_indices_to_keep(self, indices_to_keep):
        """
        Choose indices to keep based on the specific strategy.

        Args:
            indices_to_keep (list): List of indices to keep.

        Returns:
            list: Chosen indices to keep.
        """
        pass


class RandomContextGeneratorBiasedStart(RandomContextGenerator):
    """
    Random context generator biased towards the start.
    """
    def choose_indices_to_keep(self, indices_to_keep):
        """
        Choose indices to keep, biased towards the start.

        Args:
            indices_to_keep (list): List of indices to keep.

        Returns:
            list: Chosen indices to keep.
        """
        indices_to_keep = np.sort(indices_to_keep)
        indices_to_keep = indices_to_keep[:self.max_examples]
        return indices_to_keep


class RandomContextGeneratorBiasedEnd(RandomContextGenerator):
    """
    Random context generator biased towards the end.
    """
    def choose_indices_to_keep(self, indices_to_keep):
        """
        Choose indices to keep, biased towards the end.

        Args:
            indices_to_keep (list): List of indices to keep.

        Returns:
            list: Chosen indices to keep.
        """
        indices_to_keep = np.sort(indices_to_keep)
        indices_to_keep = indices_to_keep[-self.max_examples:]
        return indices_to_keep


class RandomContextGeneratorUnbiased(RandomContextGenerator):
    """
    Random context generator unbiased.
    """
    def choose_indices_to_keep(self, indices_to_keep):
        """
        Choose indices to keep, unbiased.

        Args:
            indices_to_keep (list): List of indices to keep.

        Returns:
            list: Chosen indices to keep.
        """
        indices_to_keep = indices_to_keep[:self.max_examples]
        indices_to_keep = np.sort(indices_to_keep)  # we still sort the indices to keep 
        return indices_to_keep


class RandomContextGeneratorUnbiasedOnlyPositive(RandomContextGeneratorUnbiased):
    """
    Random context generator unbiased, keeping only positive feedback examples.
    """
    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data, keeping only positive feedback examples.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        assert self.rng is not None

        # Find the indices of the positive feedback
        positive_indices = [i for i, feedback in enumerate(training_task_feedbacks) if feedback == 1]
        
        # Keep only the training examples with positive feedback
        filtered_training_task_prompts = [training_task_prompts[i] for i in positive_indices]
        filtered_training_model_predictions = [training_model_predictions[i] for i in positive_indices]
        filtered_training_task_feedbacks = [training_task_feedbacks[i] for i in positive_indices]
        filtered_training_task_answers = [training_task_answers[i] for i in positive_indices]
        filtered_training_task_accuracies = [training_task_accuracies[i] for i in positive_indices]

        # Call the parent class to do the rest of the work, exactly like before, but with the filtered data with only positive feedback
        return super().generate(filtered_training_task_prompts, filtered_training_model_predictions, filtered_training_task_feedbacks, filtered_training_task_answers, filtered_training_task_accuracies)


class RandomContextGeneratorUnbiasedOnlyNegative(RandomContextGeneratorUnbiased):
    """
    Random context generator unbiased, keeping only negative feedback examples.
    """
    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data, keeping only negative feedback examples.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        assert self.rng is not None

        # Find the indices of the negative feedback
        negative_indices = [i for i, feedback in enumerate(training_task_feedbacks) if feedback == 0]
        
        # Keep only the training examples with negative feedback
        filtered_training_task_prompts = [training_task_prompts[i] for i in negative_indices]
        filtered_training_model_predictions = [training_model_predictions[i] for i in negative_indices]
        filtered_training_task_feedbacks = [training_task_feedbacks[i] for i in negative_indices]
        filtered_training_task_answers = [training_task_answers[i] for i in negative_indices]
        filtered_training_task_accuracies = [training_task_accuracies[i] for i in negative_indices]

        # Call the parent class to do the rest of the work, exactly like before, but with the filtered data with only negative feedback
        return super().generate(filtered_training_task_prompts, filtered_training_model_predictions, filtered_training_task_feedbacks, filtered_training_task_answers, filtered_training_task_accuracies)


class RandomContextGeneratorBiasedStartOnlyPositive(RandomContextGeneratorBiasedStart):
    """
    Random context generator biased towards the start, keeping only positive feedback examples.
    """
    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data, keeping only positive feedback examples.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        assert self.rng is not None

        # Find the indices of the positive feedback
        positive_indices = [i for i, feedback in enumerate(training_task_feedbacks) if feedback == 1]
        
        # Keep only the training examples with positive feedback
        filtered_training_task_prompts = [training_task_prompts[i] for i in positive_indices]
        filtered_training_model_predictions = [training_model_predictions[i] for i in positive_indices]
        filtered_training_task_feedbacks = [training_task_feedbacks[i] for i in positive_indices]
        filtered_training_task_answers = [training_task_answers[i] for i in positive_indices]
        filtered_training_task_accuracies = [training_task_accuracies[i] for i in positive_indices]

        # Call the parent class to do the rest of the work, exactly like before, but with the filtered data with only positive feedback
        return super().generate(filtered_training_task_prompts, filtered_training_model_predictions, filtered_training_task_feedbacks, filtered_training_task_answers, filtered_training_task_accuracies)


class RandomContextGeneratorBiasedEndOnlyPositive(RandomContextGeneratorBiasedEnd):
    """
    Random context generator biased towards the end, keeping only positive feedback examples.
    """
    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data, keeping only positive feedback examples.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        assert self.rng is not None

        # Find the indices of the positive feedback
        positive_indices = [i for i, feedback in enumerate(training_task_feedbacks) if feedback == 1]
        
        # Keep only the training examples with positive feedback
        filtered_training_task_prompts = [training_task_prompts[i] for i in positive_indices]
        filtered_training_model_predictions = [training_model_predictions[i] for i in positive_indices]
        filtered_training_task_feedbacks = [training_task_feedbacks[i] for i in positive_indices]
        filtered_training_task_answers = [training_task_answers[i] for i in positive_indices]
        filtered_training_task_accuracies = [training_task_accuracies[i] for i in positive_indices]

        # Call the parent class to do the rest of the work, exactly like before, but with the filtered data with only positive feedback
        return super().generate(filtered_training_task_prompts, filtered_training_model_predictions, filtered_training_task_feedbacks, filtered_training_task_answers, filtered_training_task_accuracies)


class ApproximateContextGenerator(ContextGenerator, ABC):
    """
    Abstract base class for approximate context generators.
    """
    def __init__(self, max_examples: int, p_keep: float, max_contexts: int, verbose: bool):
        """
        Initialize the approximate context generator.

        Args:
            max_examples (int): Maximum number of examples to keep.
            p_keep (float): Probability of keeping an example.
            max_contexts (int): Maximum number of contexts.
            verbose (bool): Whether to print verbose logs.
        """
        super().__init__(max_examples, verbose)
        self.p_keep = p_keep
        self.max_contexts = max_contexts

        self.contexts_storage = [(list(), list(), list(), list(), list()) for _ in range(self.max_contexts)]
        self.log_contexts_probs = [0.0 for _ in range(self.max_contexts)]  # Initialize with log(1.0) = 0.0

        self.metrics_to_report = {
            "approx_index": None,
            "approx_lengths": {i: 0 for i in range(self.max_contexts)}
        }

        self.rng = None

    def get_context_additional_metrics(self):
        """
        Get additional metrics for the context.

        Returns:
            dict: Additional metrics for the context.
        """
        return deepcopy(self.metrics_to_report)

    def set_random_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed.
        """
        self.rng = np.random.default_rng(seed)

    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str] | List[int] | List[bool]]:
        """
        Generate context from training data.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        if len(training_task_prompts) == 0:
            context_index = self.rng.choice(range(self.max_contexts))
            self.metrics_to_report["approx_index"] = int(context_index)
            return [], [], [], [], []
        
        last_training_task_prompt = training_task_prompts[-1]
        last_training_model_prediction = training_model_predictions[-1]
        last_training_task_feedback = training_task_feedbacks[-1]
        last_training_task_answer = training_task_answers[-1]
        last_training_task_accuracy = training_task_accuracies[-1]

        for context_index in range(self.max_contexts):
            if len(self.contexts_storage[context_index][0]) >= self.max_examples:
                continue

            keep_example = self.rng.binomial(1, self.p_keep)
            if keep_example:
                self.contexts_storage[context_index][0].append(last_training_task_prompt)
                self.contexts_storage[context_index][1].append(last_training_model_prediction)
                self.contexts_storage[context_index][2].append(last_training_task_feedback)
                self.contexts_storage[context_index][3].append(last_training_task_answer)
                self.contexts_storage[context_index][4].append(last_training_task_accuracy)

                self.log_contexts_probs[context_index] += np.log(self.p_keep)

                self.metrics_to_report["approx_lengths"][context_index] += 1
            else:
                self.log_contexts_probs[context_index] += np.log(1 - self.p_keep)

        context_index = self.sample_context()
        self.metrics_to_report["approx_index"] = int(context_index)
        context = self.contexts_storage[context_index]
        
        return context
    
    @abstractmethod
    def sample_context(self):
        """
        Sample a context based on the specific strategy.

        Returns:
            int: Index of the sampled context.
        """
        raise NotImplementedError


class UniformApproximateContextGenerator(ApproximateContextGenerator):
    """
    Uniform approximate context generator.
    """
    def sample_context(self):
        """
        Sample a context uniformly.

        Returns:
            int: Index of the sampled context.
        """
        context_index = self.rng.choice(range(self.max_contexts))
        return context_index  


class ExactApproximateContextGenerator(ApproximateContextGenerator):
    """
    Exact approximate context generator.
    """
    def sample_context(self):
        """
        Sample a context based on exact probabilities.

        Returns:
            int: Index of the sampled context.
        """
        log_normalized_probs = self.log_contexts_probs - np.logaddexp.reduce(self.log_contexts_probs)
        normalized_probs = np.exp(log_normalized_probs)
        context_index = self.rng.choice(range(self.max_contexts), p=normalized_probs)  # not uniformly random
        return context_index


class UniformApproximateContextGeneratorOnlyPositive(UniformApproximateContextGenerator):
    """
    Uniform approximate context generator, keeping only positive feedback examples.
    """
    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data, keeping only positive feedback examples.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        if len(training_task_prompts) > 0:
            positive_indices = [i for i, feedback in enumerate(training_task_feedbacks) if feedback == 1]
            filtered_training_task_prompts = [training_task_prompts[i] for i in positive_indices]
            filtered_training_model_predictions = [training_model_predictions[i] for i in positive_indices]
            filtered_training_task_feedbacks = [training_task_feedbacks[i] for i in positive_indices]
            filtered_training_task_answers = [training_task_answers[i] for i in positive_indices]
            filtered_training_task_accuracies = [training_task_accuracies[i] for i in positive_indices]
        else:
            filtered_training_task_prompts = []
            filtered_training_model_predictions = []
            filtered_training_task_feedbacks = []
            filtered_training_task_answers = []
            filtered_training_task_accuracies = []

        return super().generate(filtered_training_task_prompts, filtered_training_model_predictions, filtered_training_task_feedbacks, filtered_training_task_answers, filtered_training_task_accuracies)


class ExactApproximateContextGeneratorOnlyPositive(ExactApproximateContextGenerator):
    """
    Exact approximate context generator, keeping only positive feedback examples.
    """
    def generate(self, training_task_prompts, training_model_predictions, training_task_feedbacks, training_task_answers, training_task_accuracies) -> Tuple[List[str], List[str], List[int], List[str], List[bool]]:
        """
        Generate context from training data, keeping only positive feedback examples.

        Args:
            training_task_prompts (list): List of training task prompts.
            training_model_predictions (list): List of training model predictions.
            training_task_feedbacks (list): List of training task feedbacks.
            training_task_answers (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.

        Returns:
            Tuple: Generated context consisting of prompts, predictions, feedbacks, answers, and accuracies.
        """
        if len(training_task_prompts) > 0:
            positive_indices = [i for i, feedback in enumerate(training_task_feedbacks) if feedback == 1]
            filtered_training_task_prompts = [training_task_prompts[i] for i in positive_indices]
            filtered_training_model_predictions = [training_model_predictions[i] for i in positive_indices]
            filtered_training_task_feedbacks = [training_task_feedbacks[i] for i in positive_indices]
            filtered_training_task_answers = [training_task_answers[i] for i in positive_indices]
            filtered_training_task_accuracies = [training_task_accuracies[i] for i in positive_indices]
        else:
            filtered_training_task_prompts = []
            filtered_training_model_predictions = []
            filtered_training_task_feedbacks = []
            filtered_training_task_answers = []
            filtered_training_task_accuracies = []

        return super().generate(filtered_training_task_prompts, filtered_training_model_predictions, filtered_training_task_feedbacks, filtered_training_task_answers, filtered_training_task_accuracies)


def load_context_generator(context_generator_name: str, max_examples: int, p_keep: float, max_contexts: int, approximate_context_sampling_method: str, verbose: bool) -> RandomContextGenerator:
    """
    Load the appropriate context generator based on the provided parameters.

    Args:
        context_generator_name (str): Name of the context generator.
        max_examples (int): Maximum number of examples to keep.
        p_keep (float): Probability of keeping an example.
        max_contexts (int): Maximum number of contexts.
        approximate_context_sampling_method (str): Sampling method for approximate context generator.
        verbose (bool): Whether to print verbose logs.

    Returns:
        RandomContextGenerator: The appropriate context generator.
    """
    if context_generator_name == "random_biased_start":
        return RandomContextGeneratorBiasedStart(max_examples, p_keep, verbose)
    elif context_generator_name == "random_biased_end":
        return RandomContextGeneratorBiasedEnd(max_examples, p_keep, verbose)
    elif context_generator_name == "random_unbiased":
        return RandomContextGeneratorUnbiased(max_examples, p_keep, verbose)
    elif context_generator_name == "random_biased_start_only_positive":
        return RandomContextGeneratorBiasedStartOnlyPositive(max_examples, p_keep, verbose)
    elif context_generator_name == "random_biased_end_only_positive":
        return RandomContextGeneratorBiasedEndOnlyPositive(max_examples, p_keep, verbose)
    elif context_generator_name == "random_unbiased_only_positive":
        return RandomContextGeneratorUnbiasedOnlyPositive(max_examples, p_keep, verbose)  
    elif context_generator_name == "random_unbiased_only_negative":
        return RandomContextGeneratorUnbiasedOnlyNegative(max_examples, p_keep, verbose)  
    elif context_generator_name == "approximate":
        assert max_contexts > 0, "You need to specify a number of maximum contexts for the approximate context generator"
        if approximate_context_sampling_method == "uniform":
            return UniformApproximateContextGenerator(max_examples, p_keep, max_contexts, verbose)
        elif approximate_context_sampling_method == "exact":
            return ExactApproximateContextGenerator(max_examples, p_keep, max_contexts, verbose)
    elif context_generator_name == "approximate_only_positive":
        assert max_contexts > 0, "You need to specify a number of maximum contexts for the approximate context generator"
        if approximate_context_sampling_method == "uniform":
            return UniformApproximateContextGeneratorOnlyPositive(max_examples, p_keep, max_contexts, verbose)
        elif approximate_context_sampling_method == "exact":
            return ExactApproximateContextGeneratorOnlyPositive(max_examples, p_keep, max_contexts, verbose)
    else:
        raise ValueError(f"Unknown context generator {context_generator_name} or approximate context sampling method {approximate_context_sampling_method}")