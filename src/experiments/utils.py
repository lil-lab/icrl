from contextlib import AbstractContextManager
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple


from src.utils.logger import get_data_dir, get_logger, read, save

def get_experiment_dir(args):
    """
    Constructs the directory path for the experiment based on the provided arguments.

    Args:
        args: Experiment arguments.

    Returns:
        str: The directory path for the experiment.
    """
    context_stategy_str = f"{args.context_strategy_name}"
    if args.context_strategy_name.startswith("approximate"):
        context_stategy_str += f"_{args.approximate_context_sampling_method}_{str(args.max_contexts)}"
    elif args.context_strategy_name.startswith("ucb"):
        context_stategy_str += f"_{args.ucb_alpha}"
    elif "reset" in args.context_strategy_name:
        context_stategy_str += f"_reset{args.prob_reset}"
    elif "mixed" in args.context_strategy_name:
        context_stategy_str += f"_explor{args.p_exploration}"

    if args.exemplars_per_label > 0:
        context_stategy_str += f"_exemplars{args.exemplars_per_label}"

    return os.path.join(
        get_data_dir(), 
        f"{args.model_name}", 
        f"{args.task_name}", 
        f"{args.max_context_examples}", 
        f"{args.icrl}", 
        f"{context_stategy_str}",
        f"{args.context_p_keep}", 
        f"{args.icrl_omit_feedback}", 
        f"{args.icrl_flip_feedback}" if not args.icrl_flip_feedback else f"{args.icrl_flip_feedback}_{args.icrl_flip_feedback_prob}" , 
        f"{args.temperature}", 
        f"{args.training_seed}")

DATA_FILENAME = "data.pickle"
RESULTS_FILENAME = "results.json"
TIMES_FILENAME = "times.json"
SAVE_INTERVAL = 100
MAXIMUM_CONTEXT_LENGTH_FILENAME = "maximum_context_length.json"

def get_maximum_context_name_path(args):
    """
    Constructs the file path for the maximum context length file based on the provided arguments.

    Args:
        args: Experiment arguments.

    Returns:
        str: The file path for the maximum context length file.
    """
    return os.path.join(get_data_dir(), f"{args.model_name}", f"{args.task_name}", MAXIMUM_CONTEXT_LENGTH_FILENAME)

def generate_metrics(gold_labels, accuracies, additional_metrics=None):
    """
    Generates metrics based on the provided gold labels and accuracies.

    Args:
        gold_labels (list): List of gold labels.
        accuracies (list): List of accuracies.
        additional_metrics (dict, optional): Additional metrics to include.

    Returns:
        dict: Generated metrics.
    """
    assert len(gold_labels) == len(accuracies)
    metrics = {
        "accuracy": sum(accuracies) / len(accuracies) if len(accuracies) > 0 else None,
        "per_class_accuracies": [
            {
                "class": class_name,
                "accuracy": sum([1 for gold_label, accurate in zip(gold_labels, accuracies) if accurate == 1 and gold_label == class_name]) / sum([1 for gold_label in gold_labels if gold_label == class_name])
            }
            for class_name in sorted(list(set(gold_labels)))
        ] if len(accuracies) > 0 else None,
        "context_length": len(accuracies),
    }
    if additional_metrics is not None:
        metrics.update(additional_metrics)
    return metrics

@dataclass
class StepData:
    """
    Data class to store information for each step in the experiment.
    """
    context: Optional[Tuple[List]] = None
    context_metrics: Optional[Dict] = None
    training_data: Optional[Tuple[Any]] = None
    test_data: Optional[Tuple[List]] = None
    test_metrics: Optional[Dict] = None
    time: Optional[float] = None

    def set_context(self, icl_task_prompt_list, icl_model_prediction_list, icl_task_feedback_list, icl_task_answer_list, icl_task_accuracies, additional_metrics=None):
        """
        Sets the context data and generates context metrics.

        Args:
            icl_task_prompt_list (list): List of task prompts.
            icl_model_prediction_list (list): List of model predictions.
            icl_task_feedback_list (list): List of task feedbacks.
            icl_task_answer_list (list): List of task answers.
            icl_task_accuracies (list): List of task accuracies.
            additional_metrics (dict, optional): Additional metrics to include.
        """
        self.context = (icl_task_prompt_list, icl_model_prediction_list, icl_task_feedback_list, icl_task_answer_list, icl_task_accuracies)
        self.context_metrics = generate_metrics(icl_task_answer_list, icl_task_accuracies, additional_metrics)

    def get_context(self):
        """
        Returns the context data.

        Returns:
            tuple: The context data.
        """
        return self.context

    def get_context_metrics(self):
        """
        Returns the context metrics.

        Returns:
            dict: The context metrics.
        """
        return self.context_metrics
    
    def set_training_data(self, training_task_prompt, training_model_prediction, training_task_feedback, training_task_answer, training_task_accuracies):
        """
        Sets the training data.

        Args:
            training_task_prompt (list): List of training task prompts.
            training_model_prediction (list): List of training model predictions.
            training_task_feedback (list): List of training task feedbacks.
            training_task_answer (list): List of training task answers.
            training_task_accuracies (list): List of training task accuracies.
        """
        self.training_data = (training_task_prompt, training_model_prediction, training_task_feedback, training_task_answer, training_task_accuracies)

    def get_training_data(self):
        """
        Returns the training data.

        Returns:
            tuple: The training data.
        """
        return self.training_data
    
    def get_test_data(self):
        """
        Returns the test data.

        Returns:
            tuple: The test data.
        """
        return self.test_data
    
    def set_test_data(self, test_task_prompts, test_model_predictions, test_task_feedbacks, test_task_answers, test_task_accuracies):
        """
        Sets the test data and generates test metrics.

        Args:
            test_task_prompts (list): List of test task prompts.
            test_model_predictions (list): List of test model predictions.
            test_task_feedbacks (list): List of test task feedbacks.
            test_task_answers (list): List of test task answers.
            test_task_accuracies (list): List of test task accuracies.
        """
        self.test_data = (test_task_prompts, test_model_predictions, test_task_feedbacks, test_task_answers, test_task_accuracies)
        self.test_metrics = generate_metrics(test_task_answers, test_task_accuracies)
    
    def get_test_metrics(self):
        """
        Returns the test metrics.

        Returns:
            dict: The test metrics.
        """
        return self.test_metrics

    def training_step_processed(self):
        """
        Checks if the training step has been processed.

        Returns:
            bool: True if the training step has been processed, False otherwise.
        """
        return self.training_data is not None

    def test_step_processed(self):
        """
        Checks if the test step has been processed.

        Returns:
            bool: True if the test step has been processed, False otherwise.
        """
        return self.test_data is not None
    
    def context_processed(self):
        """
        Checks if the context has been processed.

        Returns:
            bool: True if the context has been processed, False otherwise.
        """
        return self.context is not None
    
    def get_time(self):
        """
        Returns the time.

        Returns:
            float: The time.
        """
        return self.time
    
    def set_time(self, time):
        """
        Sets the time.

        Args:
            time (float): The time.
        """
        self.time = time


class ExperimentDataManager:
    """
    Manages the data for an experiment.
    """
    def __init__(self, args):
        self.args = args
        self.experiment_dir = get_experiment_dir(args)
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads the data from the experiment directory.

        Returns:
            dict: The loaded data.
        """
        try:
            data = read(os.path.join(self.experiment_dir, DATA_FILENAME))
            # Delete hf_token from args, if it exists
            if "hf_token" in self.args.__dict__:
                print(f"Deleting hf_token from {self.experiment_dir}")
                del self.args.hf_token
            # Delete hf_token from data, if it exists
            if "hf_token" in data["args"]:
                del data["args"]["hf_token"]
            get_logger().info(f"Loaded data from {self.experiment_dir}. Number of past steps: {len(data['steps'])}")
            return data
        except OSError:
            # Delete hf_token before saving
            if "hf_token" in self.args.__dict__:
                del self.args.hf_token
            print(f"Data not found in {os.path.join(self.experiment_dir, DATA_FILENAME)}. Starting new data.")
            return {
                "args": self.args.__dict__,
                "steps": dict()
            }

    def get_step_data(self, step) -> StepData:
        """
        Returns the data for a specific step. If the step does not exist, it creates a new StepData object.

        Args:
            step (int): The step number.

        Returns:
            StepData: The data for the step.
        """
        if step not in self.data["steps"]:
            self.data["steps"][step] = StepData()
        return self.data["steps"][step]

    def set_maximum_context_length(self, maximum_context_length):
        """
        Sets the maximum context length and saves it to a file.

        Args:
            maximum_context_length (int): The maximum context length.
        """
        save(maximum_context_length, get_maximum_context_name_path(self.args))

    def get_maximum_context_length(self):
        """
        Returns the maximum context length.

        Returns:
            int: The maximum context length.
        """
        try:
            return read(get_maximum_context_name_path(self.args))
        except OSError:
            return None

    def save_data(self):
        """
        Saves the experiment data to a file.
        """
        save(self.data, os.path.join(self.experiment_dir, DATA_FILENAME))

    def process_results(self):
        """
        Processes the results of the experiment.

        Returns:
            dict: The processed results.
        """
        results = {
            "args": self.data["args"],
            "test_results": {
                step: {
                    "test_metrics": step_data.get_test_metrics(),
                    "context_metrics": step_data.get_context_metrics(),
                }
                for step, step_data in self.data["steps"].items() if step_data.test_step_processed()
            },
            "training_examples": {
                step: step_data.get_training_data()
                for step, step_data in self.data["steps"].items() if step_data.training_step_processed()
            }
        }
        return results

    def process_times(self):
        """
        Processes the times of the experiment.

        Returns:
            dict: The processed times.
        """
        times = {
            "args": self.data["args"],
            "times": {
                step: step_data.get_time()
                for step, step_data in self.data["steps"].items()
            }
        }
        return times

    def save_results(self, results):
        """
        Saves the results to a file.

        Args:
            results (dict): The results to save.
        """
        save(results, os.path.join(self.experiment_dir, RESULTS_FILENAME))

    def save_times(self, times):
        """
        Saves the times to a file.

        Args:
            times (dict): The times to save.
        """
        save(times, os.path.join(self.experiment_dir, TIMES_FILENAME))

class ExperimentContextManager(AbstractContextManager):
    """
    Context manager for managing experiment data.
    """
    def __init__(self, args, save_interval=SAVE_INTERVAL):
        self.data_manager = ExperimentDataManager(args)
        # If steps data not empty, raise an error (to avoid overwriting)
        if len(self.data_manager.data["steps"]) > 0: 
            get_logger().info(f"Data already exists in {self.data_manager.experiment_dir}.")
            raise ValueError(f"Data already exists in {self.data_manager.experiment_dir}.")
        self.save_interval = save_interval
        self.new_step_data: Dict[int, StepData] = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Saves the data when exiting the context.

        Args:
            exc_type: Exception type.
            exc_value: Exception value.
            traceback: Traceback object.

        Returns:
            bool: False to propagate the exception.
        """
        if exc_type is not None:
            get_logger().info(f"Output data saved in the following location due to an exception: {self.data_manager.experiment_dir}")
        else:
            self._save_changes()
            get_logger().info(f"Output data saved in the following location: {self.data_manager.experiment_dir}")
        self._save_to_disk()
        return False

    def get_step_data(self, step) -> StepData:
        """
        Returns the data for a specific step. If the step does not exist, it creates a new StepData object.

        Args:
            step (int): The step number.

        Returns:
            StepData: The data for the step.
        """
        if step not in self.new_step_data:
            self.new_step_data[step] = self.data_manager.get_step_data(step)
        return self.new_step_data[step]

    def save_changes(self):
        """
        Saves the changes to the data if the save interval is reached.
        """
        if len(self.new_step_data) == self.save_interval:
            self._save_changes()
            self._save_to_disk()

    def _save_changes(self):
        """
        Updates the data manager with the new step data and clears the new step data.
        """
        self.data_manager.data["steps"].update(self.new_step_data)
        self.new_step_data.clear()

    def _save_to_disk(self):
        """
        Saves the data, results, and times to disk.
        """
        self.data_manager.save_data()
        results = self.data_manager.process_results()
        self.data_manager.save_results(results)
        times = self.data_manager.process_times()
        self.data_manager.save_times(times)