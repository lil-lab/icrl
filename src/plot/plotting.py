import os
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLineCollection
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
from src.experiments.utils import DATA_FILENAME, ExperimentDataManager
from src.task.task import load_task
from src.utils.logger import get_plots_dir, get_data_dir
import seaborn as sns
from matplotlib.colors import to_rgba
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec

import matplotlib.ticker as tick
import re
from scipy.stats import spearmanr

from matplotlib.lines import Line2D


MODEL_TO_NAME = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 'Llama-3.1 8B',
    "microsoft/Phi-3.5-mini-instruct": 'Phi-3.5-Mini',
    "Qwen/Qwen2.5-0.5B-Instruct": 'Qwen2.5 0.5B',
    "Qwen/Qwen2.5-1.5B-Instruct": 'Qwen2.5 1.5B',
    "Qwen/Qwen2.5-3B-Instruct": 'Qwen2.5 3B',
    "Qwen/Qwen2.5-7B-Instruct": 'Qwen2.5 7B',
    "Qwen/Qwen2.5-14B-Instruct": 'Qwen2.5 14B',
    "Qwen/Qwen2.5-32B-Instruct": 'Qwen2.5 32B',
    "Qwen/Qwen2.5-72B-Instruct": 'Qwen2.5 72B',
    "gemini-1.5-flash-8b": "Gemini-1.5 8B",
}

TASK_TO_NAME = {
    "banking77": "Banking77",
    "banking77_unsemantic": "Abs. Banking77",
    "clinic150": "CLINC150",
    "clinic150_unsemantic": "Abs. CLINC150", 
    "nlu": "NLU",
    "nlu_unsemantic": "Abs. NLU",
    "trec_coarse": "TREC",
    "trec_coarse_unsemantic": "Abs. TREC",
    "trec_fine": "TREC Fine",
    "trec_fine_unsemantic": "Abs. TREC Fine",
}

STRATEGY_TO_NAME = {
    "random_unbiased_only_positive": "Unbiased",
    "random_biased_end_only_positive": "Start-Biased",
    "random_biased_start_only_positive": "End-Biased",
}

APPROXIMATE_CONTEXT_SAMPLING_METHOD_TO_NAME = {
    "uniform": "Uniform",
    "exact": "Exact",
}

# EXPERIMENT SUBDIRECTORIES
MAIN_RESULTS_SUBDIR = "main_results"
P_SEARCH_SUBDIR = "p_keep_search"
CONTEXT_LENGTH_STRATEGY_SUBDIR = "context_length_strategy_comparison"
ABLATIONS_SUBDIR = "ablations"
APPROX_BEAM_SIZES_SUBDIR = "approximate_beam_sizes"
APPROXIMATE_CONTEXT_STRATEGY_SUBDIR = "approximate_sampling_methods"
APPROXIMATE_DETAILED_RESULTS_SUBDIR = "approximate_detailed_results"
NAIVE_PLUS_COMPARISON_SUBDIR = "naive_plus_comparison"
NAIVE_HEATMAPS_SUBIDR = "naive_heatmaps"
UNSEMANTIC_TASKS_SUBDIR = "unsemantic_tasks"


# DATA-PLOTTING CONSTANTS
WINDOW_SIZE = 256
 
# STYLE AND OUTPUT
COLOR_PALETTE = 'deep'
BARS_TEXT_SIZE = 'x-large' 
LABEL_SIZE = 'x-large' 
TITLE_SIZE = 'xx-large'
LEGEND_FONT_SIZE = 'x-large'
BARS_TEXT_SIZE = 'large'
FONT_FAMILY = 'Times'
FORMAT = 'pdf'
BARS_TEXT_ROTATION = 60

def get_colors(num_colors, gradient=False):
    if not gradient:
        ALL_COLORS = [
            '#377eb8', 
            '#ff7f00', 
            '#4daf4a',
            '#f781bf', 
            '#a65628', 
            '#984ea3',
            '#999999', 
            '#e41a1c', 
            '#dede00'
        ]
        return ALL_COLORS[:num_colors]
    else:
        return sns.color_palette("magma", n_colors=num_colors)
    

def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format
    
class Args:
    """
    A class to represent the arguments for an experiment.

    Attributes:
    ----------
    model_name : str
        The name of the model.
    task_name : str
        The name of the task.
    max_context_examples : int or str
        The maximum number of context examples.
    icrl : bool
        Whether ICRL is enabled.
    context_strategy_name : str
        The name of the context strategy.
    context_p_keep : float
        The probability of keeping a context.
    icrl_omit_feedback : bool
        Whether to omit feedback in ICRL.
    icrl_flip_feedback : bool
        Whether to flip feedback in ICRL.
    icrl_flip_feedback_prob : float or None
        The probability of flipping feedback in ICRL.
    temperature : float
        The temperature for the model.
    training_seed : int
        The seed for training.
    max_contexts : int or None
        The maximum number of contexts.
    approximate_context_sampling_method : str or None
        The method for approximate context sampling.
    """

    def __init__(self, model_name, task_name, max_context_examples, icrl, context_strategy_name, context_p_keep, icrl_omit_feedback, icrl_flip_feedback, icrl_flip_feedback_prob, temperature, training_seed, max_contexts, approximate_context_sampling_method, ucb_alpha, prob_reset, p_exploration, exemplars_per_label):
        self.model_name = model_name
        self.task_name = task_name
        self.max_context_examples = max_context_examples
        self.icrl = icrl
        self.context_strategy_name = context_strategy_name
        self.context_p_keep = context_p_keep
        self.icrl_omit_feedback = icrl_omit_feedback
        self.icrl_flip_feedback = icrl_flip_feedback
        self.icrl_flip_feedback_prob = icrl_flip_feedback_prob
        self.temperature = temperature
        self.training_seed = training_seed
        self.max_contexts = max_contexts
        self.approximate_context_sampling_method = approximate_context_sampling_method
        self.ucb_alpha = ucb_alpha
        self.prob_reset = prob_reset
        self.p_exploration = p_exploration
        self.exemplars_per_label = exemplars_per_label

def extract_data():
    """
    Extracts experiment data from the data directory.

    This function walks through the data directory, processes the directory structure to extract
    experiment parameters, and creates an Args object for each experiment. The Args objects are
    collected in a list and returned.

    Returns:
        list: A list of Args objects representing the experiments.
    """
    data_dir = get_data_dir()
    all_experiments = []

    for root, dirs, files in os.walk(data_dir):
        if DATA_FILENAME in files:
            path_parts = os.path.relpath(root, data_dir).split(os.sep)
            # print(f"Processing {path_parts}")
            if len(path_parts) >= 10:
                training_seed = path_parts[-1]
                temperature = path_parts[-2]
                icrl_flip_feedback = path_parts[-3]
                icrl_omit_feedback = path_parts[-4] == "True"
                context_p_keep = path_parts[-5]
                context_strategy_name = path_parts[-6]
                icrl = path_parts[-7] == "True"
                max_context_examples = path_parts[-8]
                task_name = path_parts[-9]
                model_name = os.path.join(*path_parts[-11:-9])

                exemplars_per_label = 0
                if "_exemplars" in context_strategy_name:
                    exemplars_match = re.search(r'_exemplars\d+', context_strategy_name)
                    if exemplars_match:
                        exemplars_substring = exemplars_match.group()
                        context_strategy_name = context_strategy_name.replace(exemplars_substring, "")
                        exemplars_per_label = int(exemplars_substring.split("_exemplars")[-1])

                if "resets" in context_strategy_name:
                    # Find substring of type _reset<float> and remove it. Use it as prob_reset. To find the substring, use regex
                    reset_substring = re.search(r'_reset\d+\.\d+', context_strategy_name).group()
                    context_strategy_name = context_strategy_name.replace(reset_substring, "")
                    
                    
                    # Prob reset is the substring after '_reset' and before the end of the string
                    prob_reset = reset_substring.split("_reset")[-1]
                else:
                    prob_reset = None

                if "mixed" in context_strategy_name:
                    p_exploration_substring = re.search(r'_explor\d+\.\d+', context_strategy_name).group()
                    context_strategy_name = context_strategy_name.replace(p_exploration_substring, "")

                    p_exploration = p_exploration_substring.split("_explor")[-1]
                else:
                    p_exploration = "0.0"

                if context_strategy_name.startswith("approximate"):
                    approx_parts = context_strategy_name.split("_")
                    max_contexts = approx_parts[-1]
                    approximate_context_sampling_method = approx_parts[-2]
                    context_strategy_name = "_".join(approx_parts[:-2])
                else:
                    max_contexts = None
                    approximate_context_sampling_method = None

                if context_strategy_name.startswith("ucb"):
                    ucb_alpha = context_strategy_name.split("_")[-1]
                    context_strategy_name = "_".join(context_strategy_name.split("_")[:-1])
                else:
                    ucb_alpha = None

                if icrl_flip_feedback.startswith("True"):
                    icrl_flip_feedback_prob = float(icrl_flip_feedback.split("_")[-1])
                    icrl_flip_feedback = True
                else:
                    icrl_flip_feedback_prob = None
                    icrl_flip_feedback = False

                if model_name not in MODEL_TO_NAME.keys():
                    continue

                args = Args(
                    model_name, 
                    task_name, 
                    max_context_examples, 
                    icrl, 
                    context_strategy_name, 
                    context_p_keep, 
                    icrl_omit_feedback,
                    icrl_flip_feedback,
                    icrl_flip_feedback_prob,
                    temperature,
                    training_seed,
                    max_contexts,
                    approximate_context_sampling_method,
                    ucb_alpha,
                    prob_reset, 
                    p_exploration,
                    exemplars_per_label
                )
                
                all_experiments.append(args)

    return all_experiments

class ExperimentStorage:
    """
    A class to manage the storage and lazy loading of experiment data.
    """

    def __init__(self):
        """
        Initializes the ExperimentStorage instance.
        """
        # Dictionary to hold loaded data, keys are args (as tuples) and values are the loaded data
        self.data_cache = {}
        # List to track order of experiments
        self._access_order = []

    def load_data(self, args):
        """
        Loads data lazily from the args if it hasn't been loaded yet.
        If the data has already been loaded, it returns the cached data.
        
        Args:
            args (Args): An instance of Args class containing the parameters.
        
        Returns:
            ExperimentDataManager: The data manager containing the experiment data.
        """
        # Convert the args object into a tuple to use as a key
        args_key = self._args_to_key(args)

        # If the data is already cached, update access order and return it
        if args_key in self.data_cache:
            # Move to end of access order
            self._access_order.remove(args_key)
            self._access_order.append(args_key)
            return self.data_cache[args_key]

        # Otherwise, create a new ExperimentDataManager instance and load the data
        data_manager = ExperimentDataManager(args)
        self.data_cache[args_key] = data_manager.data
        self._access_order.append(args_key)

        # Keep only last 20 experiments
        while len(self.data_cache) > 20:
            oldest_key = self._access_order.pop(0)
            del self.data_cache[oldest_key]

        return data_manager.data

    def refresh(self):
        """
        Clears all cached data.
        """
        self.data_cache.clear()
        self._access_order.clear()

    def _args_to_key(self, args):
        """
        Converts Args object into a tuple that can be used as a key in the data cache.
        
        Args:
            args (Args): An instance of the Args class containing the experiment parameters.
        
        Returns:
            tuple: A tuple representation of the args object to be used as a key.
        """
        return (
            args.model_name, args.task_name, args.max_context_examples, args.icrl,
            args.context_strategy_name, args.context_p_keep, args.icrl_omit_feedback,
            args.icrl_flip_feedback, args.icrl_flip_feedback_prob, args.temperature,
            args.training_seed, args.max_contexts, args.approximate_context_sampling_method,
            args.exemplars_per_label, args.ucb_alpha, args.prob_reset, args.p_exploration
        )

experiment_storage = ExperimentStorage()


def compute_tables_figures(args, data):
    """
    Computes total regret and test accuracies for an experiment.
    
    Args:
        args: Args object containing experiment parameters
        data: Dictionary containing experiment data with 'steps' key
        
    Returns:
        tuple: (total_regret, first_test_acc, last_test_acc) containing:
            - Total regret (count of incorrect predictions) across all training steps
            - First test accuracy 
            - Last test accuracy
    """
    total_regret = 0
    steps = sorted(int(s) for s in data['steps'].keys())
    
    # Get test accuracies
    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
    test_accuracies = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]
    first_test_acc = test_accuracies[0] if test_accuracies else None
    last_test_acc = test_accuracies[-1] if test_accuracies else None
    
    # Calculate regret
    for step in steps:
        step_info = data['steps'][step]
        if step_info.training_step_processed():
            training_data = step_info.get_training_data()
            accuracies = training_data[4]
            if isinstance(accuracies, list):
                total_regret += accuracies.count(0)
            elif isinstance(accuracies, np.ndarray):
                total_regret += np.sum(accuracies == 0)
            else:
                # single numeric value
                if accuracies == 0:
                    total_regret += 1
                    
    return total_regret, first_test_acc, last_test_acc
    

def init_fig_settings(font_family=FONT_FAMILY):
    """
    Initializes the figure settings for plotting.

    Args:
        font_family (str): The font family to be used in the plots. Defaults to FONT_FAMILY.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": font_family
    })

def save_fig(fig, path, format=FORMAT, dpi=400):
    """
    Saves a figure to the specified path in the given format and dpi.

    Args:
        fig (matplotlib.figure.Figure): The figure to be saved.
        path (str): The path where the figure will be saved.
        format (str, optional): The format in which to save the figure. Defaults to FORMAT.
        dpi (int, optional): The resolution in dots per inch. Defaults to 400.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{path}.{format}", format=format, dpi=dpi, bbox_inches='tight')
    plt.close()
    
def plot_main_results(all_experiments):
    """
    Plots the main results of the experiments, including test accuracy and regret for different experiment types.

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    init_fig_settings()
    experiment_storage.refresh()
    linewidth = 2.0

    # Define experiment types and their order
    exp_order = ["Upper Bound", "Naive ICRL", "Stochastic ICRL", "Approximate ICRL", "Zero-shot", "Naive+ ICRL"]
    legend_order = ["Zero-shot", "Naive ICRL", "Naive+ ICRL", "Stochastic ICRL", "Upper Bound"]

    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        # Filter experiments
        if "unsemantic" in args.task_name:
            continue
        if "phi" in args.model_name.lower():
            continue
        if "qwen" in args.model_name.lower() and "7b" not in args.model_name.lower():
            continue

        # Determine experiment type based on parameters
        if not args.icrl:
            exp_type = "Upper Bound"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end" and args.temperature == "1.0":
            exp_type = "Naive ICRL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end_only_positive" and args.temperature == "2.0":
            exp_type = "Naive+ ICRL"
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.temperature == "1.0":
            exp_type = "Stochastic ICRL"
        else:
            continue

        # Store experiment data
        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = (args, experiment_storage.load_data(args))

    # Get filtered model and task names
    model_names = [model for model in MODEL_TO_NAME.keys() if ('7b' in model.lower() or "qwen" not in model.lower()) and "gemini" not in model.lower() and "phi" not in model.lower()]
    task_names = sorted(set(task for tasks in model_task_experiments.values() for task in tasks))

    assert len(model_names) <= 3, f"Expected at most models, got {len(model_names)}"
    assert len(task_names) == 5, f"Expected 5 tasks, got {len(task_names)}"

    # Set up colors and create figure
    exp_colors = {exp_type: get_colors(6)[i] for i, exp_type in enumerate(exp_order)}
    fig, axs = plt.subplots(
        len(model_names), 
        len(task_names), 
        figsize=(13, 5),
        tight_layout=True,
    )

    # Plot results for each model and task
    for i, model_name in enumerate(model_names):
        experiment_storage.refresh()
        for j, task_name in enumerate(task_names):
            experiments = model_task_experiments[model_name][task_name]
            ax_test = axs[i, j]  

            # Add row labels
            if j == 0:
                ax_test.set_ylabel('')
                ax_test.text(-0.6, 0.5, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', 
                            fontsize=TITLE_SIZE, 
                            rotation=90,
                            verticalalignment='center',
                            transform=ax_test.transAxes)
                ax_test.text(-0.4, 0.5, 'Accuracy', 
                            fontsize=LEGEND_FONT_SIZE,
                            rotation=90,
                            verticalalignment='center',
                            transform=ax_test.transAxes)

            # Add column labels
            if i == 0:
                ax_test.set_title(r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, pad=10)

            # Plot experiment data
            regrets = []
            max_steps = 0
            zero_shot_acc = None
            oracle_acc = None
            for exp_type in legend_order:
                if exp_type in experiments:
                    args, data = experiments[exp_type]
                    color = exp_colors[exp_type]

                    regret, first_test_acc, last_test_acc = compute_tables_figures(args, data)
                    print(args.task_name, args.model_name, exp_type, f"Regret: {regret}, 0-step test acc: {first_test_acc}, last test acc: {last_test_acc}")

                    # Get test accuracy data
                    steps = sorted(int(step) for step in data['steps'].keys())
                    max_steps = max(max_steps, max(steps))
                    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                    test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                    # Calculate regret
                    training_acc = [data['steps'][step].get_training_data()[4] for step in steps if data['steps'][step].training_step_processed()]
                    regret = len(training_acc) - sum(training_acc)
                    regrets.append((exp_type, regret))

                    # Plot accuracy curves
                    if exp_type == "Upper Bound":
                        oracle_acc = test_acc[-1] if test_acc else None
                    elif exp_type == "Naive ICRL":
                        zero_shot_acc = test_acc[0] if test_acc else 0
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=linewidth)
                    else:
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=linewidth)

            # Plot zero-shot and oracle lines
            if zero_shot_acc is not None:
                ax_test.plot([0, max_steps], [zero_shot_acc, zero_shot_acc], color=exp_colors["Zero-shot"], label="Zero-shot", alpha=0.9, linewidth=linewidth, linestyle='--')
            if oracle_acc:
                ax_test.plot([0, max_steps], [oracle_acc, oracle_acc], color=exp_colors["Upper Bound"], label="Upper Bound", alpha=0.9, linewidth=linewidth, linestyle='--')

            # Configure axes
            ax_test.set_ylim(0, 1)
            if i == len(model_names) - 1:
                ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE, labelpad=10)
            ax_test.set_xlim(0, max_steps + 100)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            # Remove unnecessary spines
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)

            # Set axis ticks
            ticks = [1000, 5000, 9000] if j <= 2 else [1000, 2500, 4000]
            if i == len(model_names) - 1:
                ax_test.set_xticks(ticks)
            else:
                ax_test.set_xticks(ticks, ["", "", ""])

            if j != 0:
                ax_test.set_yticks([0.3, 0.6, 0.9], ["", "", ""])
            else:
                ax_test.set_yticks([0.3, 0.6, 0.9])

    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles_labels = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]))
    handles, labels = zip(*handles_labels)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=6, fancybox=True, fontsize=LEGEND_FONT_SIZE)

    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'main_results')
    save_fig(fig, pdf_path, FORMAT)

def plot_main_results_with_phi_and_approximate(all_experiments):
    """
    Plots the main results of the experiments, including test accuracy and regret for different experiment types.

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    init_fig_settings()
    experiment_storage.refresh()
    linewidth = 2.0

    # Define experiment types and their order
    exp_order = ["Upper Bound", "Naive ICRL", "Stochastic ICRL", "Approximate ICRL", "Zero-shot", "Naive+ ICRL"]
    legend_order = ["Zero-shot", "Naive ICRL", "Stochastic ICRL", "Approximate ICRL", "Upper Bound"]

    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        # Filter experiments
        if "unsemantic" in args.task_name or "qwen" in args.model_name.lower():
            continue

        # Determine experiment type based on parameters
        if not args.icrl:
            exp_type = "Upper Bound"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end" and args.temperature == "1.0":
            exp_type = "Naive ICRL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end_only_positive" and args.temperature == "2.0":
            exp_type = "Naive+ ICRL"
            continue 
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.temperature == "1.0":
            exp_type = "Stochastic ICRL"
        elif args.context_strategy_name == "approximate_only_positive" and int(args.max_contexts) == 8 and args.approximate_context_sampling_method == "uniform":
            exp_type = "Approximate ICRL"
        else:
            continue

        # Ensure no duplicate experiments
        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = (args, experiment_storage.load_data(args))

    # Get unique model and task names
    model_names = [model for model in MODEL_TO_NAME.keys() if ("qwen" not in model.lower()) and "gemini" not in model.lower()]
    task_names = sorted(set(task for tasks in model_task_experiments.values() for task in tasks))

    assert len(model_names) <= 3, f"Expected at most models, got {len(model_names)}"
    assert len(task_names) == 5, f"Expected 5 tasks, got {len(task_names)}"

    # Set up colors and figure
    exp_colors = {exp_type: get_colors(6)[i] for i, exp_type in enumerate(exp_order)}
    fig, axs = plt.subplots(len(model_names), len(task_names), figsize=(13, 5), tight_layout=True)

    # Plot results for each model and task
    for i, model_name in enumerate(model_names):
        experiment_storage.refresh()
        for j, task_name in enumerate(task_names):
            experiments = model_task_experiments[model_name][task_name]
            ax_test = axs[i, j]

            # Add row labels
            if j == 0:
                ax_test.set_ylabel('')
                ax_test.text(-0.6, 0.5, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', 
                            fontsize=TITLE_SIZE, 
                            rotation=90,
                            verticalalignment='center',
                            transform=ax_test.transAxes)
                ax_test.text(-0.4, 0.5, 'Accuracy', 
                            fontsize=LEGEND_FONT_SIZE,
                            rotation=90,
                            verticalalignment='center',
                            transform=ax_test.transAxes)

            # Add column labels
            if i == 0:
                ax_test.set_title(r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, pad=10)

            # Plot accuracy curves
            max_steps = 0
            zero_shot_acc = None
            oracle_acc = None
            for exp_type in legend_order:
                if exp_type in experiments:
                    args, data = experiments[exp_type]
                    color = exp_colors[exp_type]

                    steps = sorted(int(step) for step in data['steps'].keys())
                    max_steps = max(max_steps, max(steps))
                    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                    test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                    if exp_type == "Upper Bound":
                        oracle_acc = test_acc[-1] if test_acc else None
                    elif exp_type == "Naive ICRL":
                        zero_shot_acc = test_acc[0] if test_acc else 0
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=linewidth)
                    else:
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=linewidth)

            # Add reference lines
            if zero_shot_acc is not None:
                ax_test.plot([0, max_steps], [zero_shot_acc, zero_shot_acc], color=exp_colors["Zero-shot"], label="Zero-shot", alpha=0.9, linewidth=linewidth, linestyle='--')
            if oracle_acc:
                ax_test.plot([0, max_steps], [oracle_acc, oracle_acc], color=exp_colors["Upper Bound"], label="Upper Bound", alpha=0.9, linewidth=linewidth, linestyle='--')

            # Configure axes
            ax_test.set_ylim(0, 1)
            if i == len(model_names) - 1:
                ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE, labelpad=10)
            ax_test.set_xlim(0, max_steps + 100)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            # Clean up plot appearance
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)

            # Set axis ticks
            ticks = [1000, 5000, 9000] if j <= 2 else [1000, 2500, 4000]
            if i == len(model_names) - 1:
                ax_test.set_xticks(ticks)
            else:
                ax_test.set_xticks(ticks, ["", "", ""])

            if j != 0:
                ax_test.set_yticks([0.3, 0.6, 0.9], ["", "", ""])
            else:
                ax_test.set_yticks([0.3, 0.6, 0.9])

    # Add legend and save
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles_labels = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]))
    handles, labels = zip(*handles_labels)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=6, fancybox=True, fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'main_results_phi_and_approximate')
    save_fig(fig, pdf_path, FORMAT)

def plot_p_keep_search_results(all_experiments):
    """
    Plots the results of the p_keep search, showing accuracy curves and regret bars for different p_keep values.
    Creates a figure with two models side by side, each having an accuracy plot and regret bar plot.

    Args:
        all_experiments (list): List of all experiment arguments.
    """
    linewidth = 2.0
    init_fig_settings()
    experiment_storage.refresh()

    # Filter experiments to get only relevant ones
    context_strategy = "random_unbiased_only_positive"
    filtered_experiments = [
        args for args in all_experiments 
        if args.context_strategy_name == context_strategy and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.task_name == "banking77" and args.max_context_examples == "None" and args.temperature == "1.0" and ("phi" in args.model_name.lower() or ("llama" in args.model_name.lower() and "8b" in args.model_name.lower()))
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, P_SEARCH_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name
    model_experiments = {}
    for args, data in filtered_experiments:
        if args.model_name not in model_experiments:
            model_experiments[args.model_name] = []
        model_experiments[args.model_name].append((args, data))

    if len(model_experiments) != 2:
        print(f"Expected 2 models, but found {len(model_experiments)}. Adjust the filtering if necessary.")
        return

    # Set up figure with gridspec for layout
    fig = plt.figure(figsize=(12, 3))
    gs = fig.add_gridspec(1, 5, width_ratios=[7, 8, 1, 7, 8], hspace=0.2)

    # Add task name label
    fig.text(0.04, 0.5, r'\textbf{' + TASK_TO_NAME['banking77'] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

    color_cycle = get_colors(max(len(exps) for exps in model_experiments.values()))
    model_experiments = dict(sorted(model_experiments.items(), key=lambda x: x[0]))

    for i, (model_name, experiments) in enumerate(model_experiments.items()):
        ax1 = fig.add_subplot(gs[0, i*3])
        ax2 = fig.add_subplot(gs[0, i*3+1])
        
        # Add model name title
        if i == 0:
            fig.text(0.305, 1.05, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
        else:
            fig.text(0.75, 1.05, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
        
        regret_dict = {}
        p_keep_values = []

        ax1.set_title("Accuracy", fontsize=LABEL_SIZE)
        ax2.set_title("Regret", fontsize=LABEL_SIZE)
        
        # Calculate regret and collect p_keep values
        for args, data in experiments:
            steps = sorted(int(step) for step in data['steps'].keys())
            if not steps:
                print(f"No steps found for {model_name} with p_keep={args.context_p_keep}")
                continue

            p_keep = args.context_p_keep
            p_keep_values.append(p_keep)
            print(f"Processing {model_name} with p_keep={p_keep}")
            
            total_regret = 0
            for step in steps:
                if data['steps'][step].training_step_processed():
                    training_data = data['steps'][step].get_training_data()
                    accuracies = training_data[4]
                    if isinstance(accuracies, list):
                        total_regret += accuracies.count(0)
                    elif isinstance(accuracies, np.ndarray):
                        total_regret += np.sum(accuracies == 0)
                    else:
                        if accuracies == 0:
                            total_regret += 1
            regret_dict[p_keep] = total_regret

        p_keep_values.sort()

        # Plot accuracy curves
        for j, p_keep in enumerate(p_keep_values):
            color = color_cycle[j]
            label = f'$p_{{keep}}={p_keep}$'
            
            args, data = next((args, data) for args, data in experiments if args.context_p_keep == p_keep)
            steps = sorted(int(step) for step in data['steps'].keys())
            
            test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
            test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
            if test_acc:
                ax1.plot(
                    test_steps, test_acc, 
                    label=label, 
                    linewidth=linewidth,
                    color=color, alpha=0.8
                )
            else:
                print(f"No test accuracy data for {model_name} with {label}")

        # Configure accuracy plot
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
        ax1.set_xlim(0, len(steps) + 100)
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([1000, 5000, 9000])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax1.set_yticks([0.3, 0.6, 0.9])

        # Create regret bar plot
        if regret_dict:
            regrets = [regret_dict[p] for p in p_keep_values]
            bars = ax2.bar(
                [f'$p_{{keep}}={p}$' for p in p_keep_values],
                regrets,
                color=[color_cycle[j] for j in range(len(p_keep_values))],
                edgecolor='black',
                width=0.4
            )
            
            ax2.grid(False)
            ax2.set_ylim(0, 12000)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax2.set_xticks([])

            # Add regret values above bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2 + 0.20, 
                    height + 40, 
                    f'{int(height)}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=BARS_TEXT_SIZE,
                    rotation=BARS_TEXT_ROTATION
                )
        else:
            ax2.text(0.5, 0.5, 'No Regret Data Available', 
                     horizontalalignment='center', 
                     verticalalignment='center', 
                     fontsize=14, 
                     color='red')
            ax2.axis('off')

    # Add legend below plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1),
               ncols=5, fancybox=True)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    pdf_filename = 'p_keep_search_results'
    pdf_path = os.path.join(plot_dir, pdf_filename)
    save_fig(fig, pdf_path)

    print(f"Saved plot for p_keep search as {pdf_path}")


def plot_ablations_without_regret(all_experiments):
    """
    Produces two rows of plots:
    - row 0 (Stochastic): random_unbiased, context_p_keep=0.1
    - row 1 (Naive+): biased_end, context_p_keep=1.0
    """
    init_fig_settings()
    experiment_storage.refresh()

    linewidth = 2.0

    # Filter experiments for stochastic and naive+ rows
    filtered_experiments_stochastic = [
        args for args in all_experiments
        if args.context_strategy_name.startswith("random_unbiased")
        and args.icrl
        and (args.task_name == "banking77" or args.task_name == "clinic150")
        and args.max_context_examples == "None"
        and "llama" in args.model_name
        and args.context_p_keep == "0.1"
        and args.temperature == "1.0"
    ]
    filtered_experiments_stochastic = [
        args
        for args in filtered_experiments_stochastic
    ]

    filtered_experiments_naiveplus = [
        args for args in all_experiments
        if args.context_strategy_name.startswith("random_biased_end")
        and args.icrl
        and (args.task_name == "banking77" or args.task_name == "clinic150")
        and args.max_context_examples == "None"
        and "llama" in args.model_name
        and args.context_p_keep == "1.0"
        and args.temperature == "2.0"
    ]
    filtered_experiments_naiveplus = [
        args
        for args in filtered_experiments_naiveplus
    ]

    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, ABLATIONS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    def group_by_task(experiments):
        task_exps = {}
        for args in experiments:
            if args.task_name not in task_exps:
                task_exps[args.task_name] = []
            task_exps[args.task_name].append(args)
        return task_exps

    task_experiments_stochastic = group_by_task(filtered_experiments_stochastic)
    task_experiments_naiveplus = group_by_task(filtered_experiments_naiveplus)

    colors = get_colors(6)
    strategies = [
        ("Only pos. reward", colors[0]),
        ("Both pos. and neg. reward", colors[1]),
        ("Only neg. reward", colors[2]),
        ("No reward", colors[3]),
        ("Inverted pos. and neg. reward", colors[4]),
        ("Noisy pos. and neg. reward", colors[5]),
    ]

    fig, axs = plt.subplots(
        2, 
        2, 
        figsize=(6, 5),
        tight_layout=True,
    )


    def plot_row(task_experiments_dict, row_index, method):
        """
        Plot all tasks for a single row (row_index=0 => top, row_index=1 => bottom).
        Returns (handles, labels) from the last ax1 so we can create a legend outside.
        """
        final_handles, final_labels = [], []

        for i, (task_name, experiments) in enumerate(reversed(task_experiments_dict.items())):
            ax1 = fig.add_subplot(axs[row_index, i])

            experiment_storage.refresh()

            # Add row and column labels
            if i == 0:
                ax1.set_ylabel('')  
                ax1.text(-0.6, 0.5, r'\textbf{' + method + '}', 
                            fontsize=TITLE_SIZE, 
                            rotation=90,
                            verticalalignment='center',
                            transform=ax1.transAxes)
                ax1.text(-0.4, 0.5, 'Accuracy', 
                            fontsize=LEGEND_FONT_SIZE,
                            rotation=90,
                            verticalalignment='center',
                            transform=ax1.transAxes)
                
            if row_index == 0:
                ax1.set_title(r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, pad=20)

            regret_dict = {s[0]: 0 for s in strategies}

            for args in experiments:
                # Determine strategy based on feedback configuration
                if (not args.icrl_omit_feedback
                    and not args.icrl_flip_feedback
                    and (args.context_strategy_name == "random_unbiased_only_positive" or args.context_strategy_name == "random_biased_end_only_positive")):
                    strategy = "Only pos. reward"
                elif (not args.icrl_omit_feedback
                      and not args.icrl_flip_feedback
                      and (args.context_strategy_name == "random_unbiased" or args.context_strategy_name == "random_biased_end")):
                    strategy = "Both pos. and neg. reward"
                elif (not args.icrl_omit_feedback
                      and not args.icrl_flip_feedback
                      and (args.context_strategy_name == "random_unbiased_only_negative" or args.context_strategy_name == "random_biased_end_only_negative")):
                    strategy = "Only neg. reward"
                elif (args.icrl_omit_feedback
                      and not args.icrl_flip_feedback):
                    strategy = "No reward"
                elif (not args.icrl_omit_feedback
                      and args.icrl_flip_feedback
                      and args.icrl_flip_feedback_prob == 1.0):
                    strategy = "Inverted pos. and neg. reward"
                elif (not args.icrl_omit_feedback
                      and args.icrl_flip_feedback
                      and args.icrl_flip_feedback_prob == 0.1):
                    strategy = "Noisy pos. and neg. reward"
                else:
                    print(f"Unrecognized strategy configuration for {args.model_name}")
                    continue

                data = experiment_storage.load_data(args)
                steps = sorted(int(step) for step in data['steps'].keys())

                color = next(s[1] for s in strategies if s[0] == strategy)

                regret, first_test_acc, last_test_acc = compute_tables_figures(args, data)

                print(method, args.task_name, strategy, f"Regret: {regret}, 0-step test acc: {first_test_acc}, last test acc: {last_test_acc}")

                # Calculate total regret
                total_regret = 0
                for step in steps:
                    if data['steps'][step].training_step_processed():
                        training_data = data['steps'][step].get_training_data()
                        accuracies = training_data[4]
                        if isinstance(accuracies, list):
                            total_regret += accuracies.count(0)
                        elif isinstance(accuracies, np.ndarray):
                            total_regret += np.sum(accuracies == 0)
                        else:
                            if accuracies == 0:
                                total_regret += 1
                regret_dict[strategy] = total_regret

                # Plot test accuracy
                test_steps = [s for s in steps if data['steps'][s].test_step_processed()]
                test_acc = [data['steps'][s].get_test_metrics().get('accuracy', 0) for s in test_steps]
                if test_acc:
                    ax1.plot(
                        test_steps, test_acc,
                        label=strategy,
                        linewidth=linewidth,
                        color=color, alpha=0.8
                    )
                else:
                    print(f"No test accuracy data for {args.model_name} with {strategy}")

                # Get training accuracy data
                training_acc = [
                    data['steps'][s].get_training_data()[4]
                    for s in steps if data['steps'][s].training_step_processed()
                ]
                if training_acc:
                    window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
                else:
                    print(f"Insufficient training accuracy data for {args.model_name} with {strategy}")

            # Configure axes
            ax1.set_ylim(0, 1)
            if row_index == 1:
                ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE, labelpad=10)
            ax1.set_xlim(0, max(steps) + 300)
            ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

            ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            ticks = [1000, 5000, 9000]

            if row_index == 1:
                ax1.set_xticks(ticks)
            else:
                ax1.set_xticks(ticks, ["", "", ""])

            ticks = [0.2, 0.5, 0.8]
            if i == 0:
                ax1.set_yticks(ticks)
            else:
                ax1.set_yticks(ticks, ["", "", ""])

            final_handles, final_labels = ax1.get_legend_handles_labels()

        return final_handles, final_labels

    # Plot both rows and create legend
    plot_row(task_experiments_naiveplus, row_index=0, method='Naive ICRL')
    handles, labels = plot_row(task_experiments_stochastic, row_index=1, method='Stochastic ICRL')

    fig.legend(
        handles, labels,
        loc='lower center',
        fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, 1.005),
        ncols=2,
        fancybox=True
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    pdf_filename = 'ablations_results_no_regret'
    pdf_path = os.path.join(plot_dir, pdf_filename)
    save_fig(fig, pdf_path)

    print("Saved plot for ablations")



def plot_approximate_beam_sizes_results(all_experiments):
    """
    Plots the results of the approximate beam sizes experiments, showing test accuracy and regret 
    for different beam sizes across tasks and models.

    Args:
        all_experiments (list): List of all experiment arguments.
    """
    experiment_storage.refresh()
    linewidth = 2.0
    context_strategy = "approximate_only_positive"
    tasks = ["banking77", "clinic150"]
    
    # Filter experiments to get approximate and stochastic baseline runs
    filtered_experiments = [
        args for args in all_experiments 
        if (context_strategy in args.context_strategy_name and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.task_name in tasks and args.approximate_context_sampling_method == "uniform") 
        or (args.task_name in tasks and args.context_strategy_name == "random_unbiased_only_positive" and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.context_p_keep == "0.1" and args.max_context_examples == "None" and args.temperature=="1.0")
    ]

    print(f"Found {len(filtered_experiments)} experiments for approximate beam sizes")
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, APPROX_BEAM_SIZES_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Group experiments by task and model
    task_model_experiments = {}
    for args in filtered_experiments:
        key = (args.task_name, args.model_name)
        if key not in task_model_experiments:
            task_model_experiments[key] = []
        task_model_experiments[key].append(args)
    
    task_names = sorted(set(task_name for task_name, _ in task_model_experiments.keys()))
    model_names = sorted(set(model_name for _, model_name in task_model_experiments.keys() if 'llama' in model_name.lower() or 'phi' in model_name.lower()))[:2]

    init_fig_settings()
    
    # Create figure with 2x5 grid - test accuracy and regret plots for each task/model
    fig, axs = plt.subplots(2, 5, figsize=(12, 5), gridspec_kw={'width_ratios': [6, 4, 0.5, 6, 4]})

    for row in range(2):
        axs[row, 2].set_visible(False)

    for i, model_name in enumerate(model_names):
        print(f"Processing model {model_name}")

        # Add model name labels
        if i == 0:
            fig.text(-0.04, 0.77, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)
        else:
            fig.text(-0.04, 0.3, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

        for j, task_name in enumerate(task_names):
            experiment_storage.refresh()

            experiments = task_model_experiments.get((task_name, model_name), [])
            if not experiments:
                continue

            ax_test = axs[i, j*3]
            ax_regret = axs[i, j*3+1]

            # Add task name titles on first row only
            if i == 0:
                if j == 0:
                    fig.text(0.3, 1.05, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
                else:
                    fig.text(0.82, 1.05, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)

            # Add subplot titles on first row
            if i == 0:
                ax_test.set_title("Test Accuracy", fontsize=LABEL_SIZE)
                ax_regret.set_title("Regret", fontsize=LABEL_SIZE)
    
            # Set up colors for different beam sizes
            beam_sizes = []
            for args in experiments:
                if context_strategy in args.context_strategy_name:
                    beam_size = int(args.max_contexts)
                else:
                    beam_size = None  # Infinite beam size
                beam_sizes.append(beam_size)
            beam_sizes = sorted(set(beam_sizes), key=lambda x: x if x is not None else float('inf'))
            color_cycle = get_colors(len(beam_sizes))
            color_map = {beam_size: color for beam_size, color in zip(beam_sizes, color_cycle)}
    
            regret_dict = {}
            experiments = sorted(experiments, key=lambda x: int(x.max_contexts) if context_strategy in x.context_strategy_name else float('inf'))
    
            # Plot test accuracy and calculate regret for each experiment
            for args in experiments:
                data = experiment_storage.load_data(args)
                if context_strategy in args.context_strategy_name:
                    beam_size = int(args.max_contexts)
                else:
                    beam_size = None
                color = color_map[beam_size]
                label = f'$K = {beam_size}$' if beam_size is not None else '$K = \infty$'
    
                steps = sorted(int(step) for step in data['steps'].keys())
                if not steps:
                    print(f"No steps found for {task_name} with beam size={beam_size}")
                    continue
    
                # Calculate total regret
                total_regret = 0
                for step in steps:
                    if data['steps'][step].training_step_processed():
                        training_data = data['steps'][step].get_training_data()
                        accuracies = training_data[4]
                        if isinstance(accuracies, list):
                            total_regret += accuracies.count(0)
                        elif isinstance(accuracies, np.ndarray):
                            total_regret += np.sum(accuracies == 0)
                        else:
                            if accuracies == 0:
                                total_regret += 1
                regret_dict[beam_size] = total_regret
    
                # Plot test accuracy curve
                test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
                if test_acc:
                    ax_test.plot(
                        test_steps, test_acc, 
                        label=label, 
                        color=color, alpha=0.8, linewidth=linewidth
                    )
                else:
                    print(f"No test accuracy data for {task_name} with {label}")
    
            # Configure test accuracy plot
            ax_test.set_ylim(0, 1)
            ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
            ax_test.set_xlim(0, len(steps) + 100)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)
            ax_test.set_xticks([1000, 5000, 9000])
            ax_test.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
            ax_test.set_yticks([0.3, 0.6, 0.9])
    
            # Create regret bar plot
            if regret_dict:
                regrets = [regret_dict[b] for b in beam_sizes]
                bars = ax_regret.bar(
                    [f'Beam size = {b}' if b is not None else 'Beam size = ∞' for b in beam_sizes],
                    regrets, 
                    color=[color_map[b] for b in beam_sizes],
                    edgecolor='black',
                    width=0.4
                )
                    
                ax_regret.grid(False)
                ax_regret.set_ylim(0, 12000)
                ax_regret.spines['right'].set_visible(False)
                ax_regret.spines['top'].set_visible(False)
                ax_regret.spines['left'].set_visible(False)
                ax_regret.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                ax_regret.set_xticks([])

                # Add regret values above bars
                for bar in bars:
                    height = bar.get_height()
                    ax_regret.text(
                        bar.get_x() + bar.get_width() / 2 + 0.20, 
                        height + 40, 
                        f'{int(height)}', 
                        ha='center', 
                        va='bottom', 
                        fontsize=BARS_TEXT_SIZE,
                        rotation=BARS_TEXT_ROTATION
                    )
            else:
                ax_regret.text(0.5, 0.5, 'No Regret Data Available', 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        fontsize=14, 
                        color='red')
                ax_regret.axis('off')
    
    # Add legend below figure
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1),
        ncols=7, fancybox=True)        

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, f'approximate_beam_sizes_results')
    save_fig(fig, pdf_path)



def plot_context_length_and_strategy_comparison(all_experiments):
    """
    Plots the comparison of context length and strategy for the given experiments.
    Creates a figure with two tasks side by side, each showing accuracy curves and regret bars.

    Args:
        all_experiments (list): List of all experiment arguments.
    """
    linewidth = 2.0
    init_fig_settings()
    experiment_storage.refresh()

    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and (args.task_name == "banking77" or args.task_name == "clinic150") and "llama" in args.model_name and "approximate" not in args.context_strategy_name and "only_positive" in args.context_strategy_name and args.context_p_keep == "0.1" and args.temperature == "1.0"
    ]
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, CONTEXT_LENGTH_STRATEGY_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by task
    task_experiments = {}
    for args in filtered_experiments:
        if args.task_name not in task_experiments:
            task_experiments[args.task_name] = []
        task_experiments[args.task_name].append(args)
    task_experiments = dict(sorted(task_experiments.items()))
    

    # Define color palette and line styles
    color_palette = get_colors(len(set(args.max_context_examples for args in filtered_experiments)))
    strategy_names = sorted(set(args.context_strategy_name for args in filtered_experiments))
    line_styles = ['-', '--', ':', '-.']
    line_style_map = dict(zip(strategy_names, line_styles[:len(strategy_names)]))

    fig = plt.figure(figsize=(10, 6))  
    gs = fig.add_gridspec(7, 2, width_ratios=[5, 3], height_ratios=[1, 1, 1, 1, 1, 1, 1], hspace=0.6)

    for task_idx, (task_name, experiments) in enumerate(task_experiments.items()):
        experiment_storage.refresh()

        # Define color map specific to max_context_examples
        max_context_examples_values = sorted(set(args.max_context_examples for args in experiments), key=lambda x: int(x) if x != "None" else float('inf'))
        color_map = dict(zip(max_context_examples_values, color_palette))

        # Initialize maximum regret for scaling bar plots
        max_regret = 0

        # Main accuracy plot
        ax1 = fig.add_subplot(gs[4*task_idx:4*task_idx+3, 0])

        # Dictionary to store regret data for bar plots
        regret_data = {strategy: {} for strategy in strategy_names}

        for args in experiments:
            data = experiment_storage.load_data(args)
            steps = sorted(int(step) for step in data['steps'].keys())  
            if not steps:
                print(f"No steps found for {args.model_name} with max_context_examples={args.max_context_examples}, strategy={args.context_strategy_name}")
                continue

            # Calculate regret
            total_regret = 0
            for step in steps:
                if data['steps'][step].training_step_processed():
                    training_data = data['steps'][step].get_training_data()
                    accuracies = training_data[4]
                    if isinstance(accuracies, list):
                        total_regret += accuracies.count(0)
                    elif isinstance(accuracies, np.ndarray):
                        total_regret += np.sum(accuracies == 0)
                    else:
                        if accuracies == 0:
                            total_regret += 1
            max_regret = max(max_regret, total_regret)
            regret_data[args.context_strategy_name][args.max_context_examples] = total_regret

            # Plot test accuracy
            test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
            test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
            if test_acc:
                ax1.plot(
                    test_steps, test_acc, 
                    label=f'W={args.max_context_examples}, strategy={args.context_strategy_name}', 
                    linestyle=line_style_map[args.context_strategy_name],
                    color=color_map[args.max_context_examples],
                    linewidth=linewidth,
                    alpha=0.8
                )
            else:
                print(f"No test accuracy data for {args.model_name} with max_context={args.max_context_examples}, strategy={args.context_strategy_name}")

        # Configure accuracy plot
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
        ax1.set_xlim(0, len(steps) + 100)
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.set_ylabel('Accuracy', fontsize=LABEL_SIZE)
        ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([1000, 5000, 9000])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax1.set_yticks([0.3, 0.6, 0.9])

        # Add task name label
        ax1.text(-0.2, 0.5, r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, transform=ax1.transAxes, rotation=90, ha='center', va='center')
        ax1.text(1.1, 0.5, 'Regret', ha='center', va='center', fontsize=LABEL_SIZE, transform=ax1.transAxes, rotation=90)

        # Create regret bar plots for each strategy
        all_regret_bars = []
        for i, strategy in enumerate(strategy_names[:3]):
            ax_bar = fig.add_subplot(gs[task_idx*4+i, 1])
            regrets = [regret_data[strategy][max_context_examples] for max_context_examples in max_context_examples_values[:2]]
            bars = ax_bar.barh(
                [f'{max_context}' for max_context in max_context_examples_values[:2]],
                regrets,
                color=[color_map[max_context] for max_context in max_context_examples_values[:2]],
                edgecolor='black',
                height=0.6
            )

            # Configure regret bar plot
            ax_bar.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)
            ax_bar.set_title(STRATEGY_TO_NAME[strategy], fontsize=LABEL_SIZE)
            ax_bar.grid(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['bottom'].set_visible(False)
            ax_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_bar.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            all_regret_bars.append(ax_bar)

            # Add regret values as text
            for bar in bars:
                width = bar.get_width()
                ax_bar.text(
                    width + 100,
                    bar.get_y() + bar.get_height() / 2,
                    f'{int(bar.get_width())}',
                    ha='left',
                    va='center',
                    fontsize=LABEL_SIZE
                )

        # Set consistent x-limits for regret bars
        for regret_bar in all_regret_bars:
            regret_bar.set_xlim(0, max_regret + 100)

        # Create legend with colors and line styles
        color_handles = [plt.Line2D([0], [0], color=color_palette[i], lw=linewidth) for i in range(len(max_context_examples_values))]
        color_labels = [f'$W={max_tokens}$' for max_tokens in ["4K", "8K", "128K"]]
        line_handles = [plt.Line2D([0], [0], color='black', linestyle=line_style_map[strategy], lw=2) for strategy in strategy_names[:3]]
        line_labels = [STRATEGY_TO_NAME[strategy] for strategy in strategy_names[:3]]

        handles = [color_handles[i // 2] if i % 2 == 0 else line_handles[i // 2] for i in range(6)]
        labels = [color_labels[i // 2] if i % 2 == 0 else line_labels[i // 2] for i in range(6)]

        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1), fancybox=True)

    plt.tight_layout(rect=[0.03, 0.0, 1, 0.95])

    # Save figure
    pdf_filename = f'context_length_and_strategy_comparison'
    pdf_path = os.path.join(plot_dir, pdf_filename)
    save_fig(fig, pdf_path)

    print(f"Saved plot as {pdf_path}")


def plot_unsemantic_results(all_experiments):
    """
    Plots the unsemantic results of the experiments, including test accuracy and regret for different experiment types.

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    init_fig_settings()

    linewidth = 2.0

    # Define the order of experiment types
    exp_order = ["Upper Bound", "Naive ICRL", "Stochastic ICRL", "Approximate ICRL", "Zero-shot",  "Naive+ ICRL"]
    legend_order = ["Naive ICRL", "Naive+ ICRL", "Stochastic ICRL", "Approximate ICRL", "Upper Bound"]
    
    # Define the directory where plots will be saved
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, UNSEMANTIC_TASKS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        # Skip all normal tasks
        if "unsemantic" not in args.task_name:
            continue
        if not args.icrl:
            exp_type = "Upper Bound"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end" and args.temperature == "1.0":
            exp_type = "Naive ICRL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end_only_positive" and args.temperature == "2.0":
            exp_type = "Naive+ ICRL"
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.temperature == "1.0":
            exp_type = "Stochastic ICRL"
        elif args.context_strategy_name == "approximate_only_positive" and int(args.max_contexts) == 8 and args.approximate_context_sampling_method == "uniform":
            exp_type = "Approximate ICRL"
        else:
            continue  # Skip experiments that don't match any of the four types

        # Add exemplar suffix if using exemplars (except for Upper Bound and Zero-shot)
        if args.exemplars_per_label == 1 and exp_type not in ["Upper Bound", "Zero-shot"]:
            exp_type = f"{exp_type} (w/ exemplars)"
            print("Changing exp type")

        # Assert there is exactly one experiment per model and task and experiment type
        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = args
        
    # Get unique model names and task names, ensuring specific model order
    model_names = [model for model in MODEL_TO_NAME.keys() if ('7b' in model.lower() or "qwen" not in model.lower()) and "phi" not in model.lower()]
    task_names = sorted(set(task for tasks in model_task_experiments.values() for task in tasks))

    assert len(model_names) <= 3, f"Expected at most 3 models, got {len(model_names)}"
    assert len(task_names) <= 5, f"Expected at most 5 tasks, got {len(task_names)}: {task_names}"

    # Define colors for each experiment type (same color for exemplar/non-exemplar versions)
    exp_colors = {}
    for i, exp_type in enumerate(exp_order):
        color = get_colors(6)[i]
        exp_colors[exp_type] = color
        if exp_type not in ["Upper Bound", "Zero-shot"]:  # No exemplar version for Upper Bound and Zero-shot
            exp_colors[f"{exp_type} (w/ exemplars)"] = color

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(
        3, 
        5, 
        figsize=(13, 6),
        gridspec_kw={'height_ratios': [3, 3, 3]},
        tight_layout=True,
    )

    # Iterate over each model to create separate plots
    for i, model_name in enumerate(model_names):
        # Iterate over tasks for the current model
        for j, task_name in enumerate(task_names):
            experiment_storage.refresh()

            experiments = model_task_experiments[model_name][task_name]

            # Select the upper axes for line plot
            ax_test = axs[0+(i), j]  # Upper plot for test accuracies (line plot)

            if i == 0:
                # Set task label at the top of each column
                ax_test.set_title(r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, pad=20)

            # Plot data for each experiment type
            regrets = []
            max_steps = 0
            oracle_acc = None
            for exp_type in legend_order:
                base_color = exp_colors[exp_type]
                
                # Handle both regular and exemplar versions
                versions = [exp_type]
                if exp_type not in ["Upper Bound", "Zero-shot"]:
                    versions.append(f"{exp_type} (w/ exemplars)")
                
                for version in versions:
                    if version not in experiments:
                        continue

                    
                    args = experiments[version]
                    data = experiment_storage.load_data(args)

                    regret, first_test_acc, last_test_acc = compute_tables_figures(args, data)

                    print(args.model_name, args.task_name, version, f"Regret: {regret}, 0-step test acc: {first_test_acc}, last test acc: {last_test_acc}")

                    
                    # Compute test accuracy and regret
                    steps = sorted(int(step) for step in data['steps'].keys())
                    max_steps = max(max_steps, max(steps))
                    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                    test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                    training_acc = [data['steps'][step].get_training_data()[4] for step in steps if data['steps'][step].training_step_processed()]
                    regret = len(training_acc) - sum(training_acc)
                    regrets.append((version, regret))

                    # Plot test accuracy
                    if exp_type == "Upper Bound":
                        oracle_acc = test_acc[-1] if test_acc else None
                    elif exp_type == "Naive ICRL" and version == exp_type:
                        ax_test.plot(test_steps, test_acc, color=base_color, label=version, alpha=0.9, linewidth=linewidth)
                    else:
                        linestyle = ':' if '(w/ exemplars)' in version else '-'
                        ax_test.plot(test_steps, test_acc, color=base_color, label=version, alpha=0.9, linewidth=linewidth, linestyle=linestyle)

            # Plot oracle line
            if oracle_acc:
                ax_test.plot([0, max_steps], [oracle_acc, oracle_acc], color=exp_colors["Upper Bound"], label="Upper Bound", alpha=0.9, linewidth=linewidth, linestyle='--')

            # Set y-axis limits for test accuracy
            if j == 0:
                ax_test.set_ylabel('')  # Clear existing label
                ax_test.text(-0.6, 0.5, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', 
                            fontsize=TITLE_SIZE, 
                            rotation=90,
                            verticalalignment='center',
                            transform=ax_test.transAxes)
                ax_test.text(-0.4, 0.5, 'Accuracy', 
                            fontsize=LEGEND_FONT_SIZE,  # Using smaller font size for "Accuracy"
                            rotation=90,
                            verticalalignment='center',
                            transform=ax_test.transAxes)
            
            ax_test.set_ylim(0, 1)
            if i == 2:
                ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE, labelpad=10)
            ax_test.set_xlim(0, max_steps + 100)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

            # Make ticks larger
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            # Remove top and right spines
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)

            offset = 300
            ax_test.set_xlim(0, max_steps + offset)
            if j <= 2:
                ticks = [1000, 5000, 9000]
            else:
                ticks = [1000, 2500, 4000]

            if i == len(model_names) - 1:
                ax_test.set_xticks(ticks)
            else:
                ax_test.set_xticks(ticks, ["", "", ""])

            # Set y-ticks
            y_ticks = [0.3, 0.6, 0.9]
            if j !=0:
                ax_test.set_yticks(y_ticks, ["", "", ""])
            else:
                ax_test.set_yticks(y_ticks)

    # Update handles and labels so that the versions w/ exemplars are not included
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles, labels = zip(*[(h, l) for h, l in zip(handles, labels) if " (w/ exemplars)" not in l])
    
    # Reorder the handles and labels to match the legend_order
    handles_labels = sorted(zip(handles, labels), key=lambda x: legend_order.index(x[1]))
    handles, labels = zip(*handles_labels)

    # Create additional handles for line styles with and without exemplars
    exemplar_handle = Line2D([], [], color='black', linestyle='-', linewidth=linewidth, label='Without Exemplars')
    no_exemplar_handle = Line2D([], [], color='black', linestyle=':', linewidth=linewidth, label='With Exemplars')

    # Combine original handles with the new handles for line styles
    combined_handles = list(handles) + [exemplar_handle, no_exemplar_handle]
    combined_labels = list(labels) + ['Without Exemplars', 'With Exemplars']

    fig.legend(combined_handles, combined_labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=6, fancybox=True, fontsize=LEGEND_FONT_SIZE)

    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'all_results')
    save_fig(fig, pdf_path, FORMAT)


def plot_scaling_qwen_results(all_experiments):
    """
    Plots the scaling qwen results of the experiments, including test accuracy and regret for different experiment types.

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    init_fig_settings()

    # Define the order of experiment types
    exp_order = ["Upper Bound", "Naive ICRL", "Stochastic ICRL", "Approximate ICRL", "Zero-shot", "Naive+ ICRL"]
    legend_order = ["Zero-shot", "Naive ICRL", "Naive+ ICRL", "Stochastic ICRL", "Approximate ICRL", "Upper Bound"]
    
    # Define the directory where plots will be saved
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        # Skip all unsemantic tasks
        if "unsemantic" in args.task_name:
            continue
        if "qwen2.5" not in args.model_name.lower():
            continue
        if not args.icrl:
            exp_type = "Upper Bound"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end" and args.temperature == "1.0":
            exp_type = "Naive ICRL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end_only_positive" and args.temperature == "2.0":
            exp_type = "Naive+ ICRL"
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.temperature == "1.0":
            exp_type = "Stochastic ICRL"
        else:
            continue  # Skip experiments that don't match any of the four types

        # Assert there is exactly one experiment per model and task and experiment type
        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = args

    # Get unique model names and task names, ensuring specific model order
    models = [(model_key, model_name) for (model_key, model_name) in MODEL_TO_NAME.items() if 'qwen' in model_key.lower()]
    tasks = [(task_key, task_name) for (task_key, task_name) in TASK_TO_NAME.items() if "unsemantic" not in task_key]

    # Define colors for each experiment type
    exp_colors = {exp_type: get_colors(6)[i] for i, exp_type in enumerate(exp_order)}

    # Define shorter labels for the bar plots
    short_labels = {
        "Upper Bound": "Upper Bound",
        "Zero-shot": "Zero-shot",
        "Naive ICRL": "Naive ICRL",
        "Naive+ ICRL": "Naive+ ICRL",
        "Stochastic ICRL": "Explor. ICRL",
        "Approximate ICRL": "Approx. ICRL"
    }

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(
        len(tasks) * 2, 
        len(models), 
        figsize=(22, len(tasks) * 5),
        gridspec_kw={'height_ratios': [3, 2] * len(tasks)},
        tight_layout=True,
    )

    for i, (model_key, model_name) in enumerate(models):
        print("Processing model " + model_name, i)
        
        # Iterate over tasks for the current model
        for j, (task_key, task_name) in enumerate(tasks):
            experiment_storage.refresh()
            experiments = model_task_experiments[model_key][task_key]

            # Select the upper and lower axes for line plot and bar plot respectively
            ax_test = axs[0+(j*2), i]  # Upper plot for test accuracies (line plot)
            ax_regret = axs[1+(j*2), i]  # Lower plot for regrets (bar plot)

            # Add subtitles for rows
            if i == 0:
                ax_test.set_ylabel('Test Accuracy', fontsize=LABEL_SIZE)
                ax_test.get_yaxis().set_label_coords(-0.25,.6)
                ax_regret.set_ylabel('Regret', fontsize=LABEL_SIZE)
                ax_regret.get_yaxis().set_label_coords(-0.25,0.6)
          
            # Set task label at the left of each row
            if i == 0:
                ax_test.text(
                    -0.6, -0.3, r'\textbf{' + task_name + '}', 
                    rotation=90, 
                    va='center', 
                    ha='center', 
                    transform=ax_test.transAxes,
                    fontsize=TITLE_SIZE
                )

            # Set model label at the top of each column
            if j == 0:
                ax_test.text(
                    0.5, 1.7, r'\textbf{' + model_name + '}', 
                    va='top', 
                    ha='center', 
                    transform=ax_test.transAxes,
                    fontsize=TITLE_SIZE
                )

            # Plot data for each experiment type
            regrets = []
            max_steps = 0
            zero_shot_acc = None
            oracle_acc = None
            for exp_type in legend_order:
                if exp_type in experiments:
                    args = experiments[exp_type]
                    data = experiment_storage.load_data(args)
                    color = exp_colors[exp_type]

                    # Compute test accuracy and regret
                    steps = sorted(int(step) for step in data['steps'].keys())
                    max_steps = max(max_steps, max(steps))

                    print("Plotting", len(steps), "steps")
                    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                    test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                    training_acc = [data['steps'][step].get_training_data()[4] for step in steps if data['steps'][step].training_step_processed()]
                    regret = len(training_acc) - sum(training_acc)  # Number of wrong training examples
                    regrets.append((exp_type, regret))

                    # Plot test accuracy (line plot)
                    if exp_type == "Upper Bound":
                        # Plot a continuous line for Upper Bound with the last step's test accuracy
                        oracle_acc = test_acc[-1] if test_acc else None
                    elif exp_type == "Naive ICRL":
                        # Store first test accuracy for zero-shot line
                        zero_shot_acc = test_acc[0] if test_acc else 0
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=4.)
                        print(f"Plotting {str(args.__dict__)}", i, j)
                    else:
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=4.)
                        print(f"Plotting {str(args.__dict__)}", i, j)

            # Plot zero-shot line after getting the accuracy from Naive ICRL
            if zero_shot_acc is not None:
                ax_test.plot([0, max_steps], [zero_shot_acc, zero_shot_acc], color=exp_colors["Zero-shot"], label="Zero-shot", alpha=0.9, linewidth=4., linestyle='--')

            # Plot oracle line
            if oracle_acc:
                ax_test.plot([0, max_steps], [oracle_acc, oracle_acc], color=exp_colors["Upper Bound"], label="Upper Bound", alpha=0.9, linewidth=4., linestyle='--')

            # Set y-axis limits for test accuracy
            ax_test.set_ylim(0, 1)
            ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
            ax_test.set_xlim(0, max_steps + 100)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

            # Make ticks larger
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            # Remove top and right spines
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)

            # Set y-ticks
            if j <= 2:
                ax_test.set_xticks([1000, 5000, 9000])
            else:
                ax_test.set_xticks([1000, 2500, 4000])

            # Hide x-ticks
            ax_test.set_yticks([0.3, 0.6, 0.9])

            # Plot regret (horizontal bar plot), omitting standard ICL and zero-shot
            regrets = [r for r in regrets if r[0] not in ["Upper Bound", "Zero-shot"]]
            if regrets:
                # Sort the regrets based on the fixed order
                sorted_regrets = sorted([r for r in regrets if r[0] not in ["Upper Bound", "Zero-shot"]], key=lambda x: legend_order.index(x[0]) if x[0] in legend_order else len(legend_order), reverse=True)
                
                bars = ax_regret.barh(
                    [short_labels[r[0]] for r in sorted_regrets],
                    [r[1] for r in sorted_regrets],
                    color=[exp_colors[r[0]] for r in sorted_regrets],
                    edgecolor='black',
                    height=0.5
                )

                # Add text annotations next to each bar
                for bar in bars:
                    width = bar.get_width()
                    ax_regret.text(
                        width + 0.03 * ax_regret.get_xlim()[1],  # Add a small offset proportional to the x-axis range
                        bar.get_y() + bar.get_height() / 2 - 0.03,
                        f'{int(bar.get_width())}',
                        ha='left',
                        va='center',
                        fontsize=LABEL_SIZE
                    )

            # Set x-axis limits for the regret plot
            ax_regret.set_xlim(0, max_steps + 100)
            ax_regret.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)  # Reduce y-tick label size

            # Remove spines and ticks for regret plot
            ax_regret.grid(False)
            ax_regret.spines['right'].set_visible(False)
            ax_regret.spines['top'].set_visible(False)
            ax_regret.spines['bottom'].set_visible(False)
            ax_regret.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_regret.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Show legend below the entire figure
    all_handles = []
    all_labels = []
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    # Reorder the handles and labels to match the legend_order
    handles_labels = sorted(zip(all_handles, all_labels), key=lambda x: legend_order.index(x[1]))
    handles, labels = zip(*handles_labels)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=6, fancybox=True, fontsize=LEGEND_FONT_SIZE)

    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'qwen_scaling')
    save_fig(fig, pdf_path, FORMAT)


def plot_scaling_qwen_results_compact(all_experiments):
    """
    Plots the scaling results for Qwen models, showing test accuracy across different model sizes and tasks.

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    init_fig_settings()

    linewidth = 2.0
    alpha = 0.7

    exp_order = ["Stochastic ICRL", "Naive+ ICRL"] 
    size_order = ['0.5', '1.5', '3', '7', '14', '32', '72']
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        if "unsemantic" in args.task_name:
            continue
        if "qwen2.5" not in args.model_name.lower():
            continue
            
        if not args.icrl:
            exp_type = "Upper Bound"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end" and args.temperature == "1.0":
            exp_type = "Naive ICRL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end_only_positive" and args.temperature == "2.0":
            exp_type = "Naive+ ICRL"
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.temperature == "1.0":
            exp_type = "Stochastic ICRL"
        else:
            continue

        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = args

    models = [(model_key, model_name) for (model_key, model_name) in MODEL_TO_NAME.items() if 'qwen' in model_key.lower()]
    tasks = [(task_key, task_name) for (task_key, task_name) in TASK_TO_NAME.items() if "unsemantic" not in task_key]

    size_colors = {size: get_colors(7, gradient=True)[i] for i, size in enumerate(size_order)}    

    linestyles = ["-", "-"]
    exp_linestyle = {exp_type: linestyle for exp_type, linestyle in zip(exp_order, linestyles)}
    
    fig, axs = plt.subplots(
        2, 
        len(tasks), 
        figsize=(13, 5),
        tight_layout=True,
    )
    
    for exp_index, exp_type in enumerate(exp_order):
        for i, (model_key, model_name) in enumerate(models):
            print(model_name)
            model_size = re.search(r'-(.*?)b-', model_key, re.IGNORECASE).group(1).lower()

            experiment_storage.refresh()
            
            for j, (task_key, task_name) in enumerate(tasks):
                if exp_type not in model_task_experiments[model_key][task_key]:
                    continue
                args = model_task_experiments[model_key][task_key][exp_type]
                data = experiment_storage.load_data(args)

                regret, first_test_acc, last_test_acc = compute_tables_figures(args, data)

                print(model_key, task_key, exp_type, f"Regret: {regret}, 0-step test acc: {first_test_acc}, last test acc: {last_test_acc}")

                ax_test = axs[exp_index][j]

                if exp_index == 0:
                    ax_test.set_title(r'\textbf{' + task_name + '}', fontsize=TITLE_SIZE, pad=20)

                max_steps = 0
 
                color = size_colors[model_size]
                linestyle = exp_linestyle[exp_type]

                steps = sorted(int(step) for step in data['steps'].keys())
                max_steps = max(max_steps, max(steps))
                test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
            
                if j == 0:
                    ax_test.text(-0.6, 0.5, r'\textbf{' + exp_order[exp_index] + '}', 
                               fontsize=TITLE_SIZE, 
                               rotation=90,
                               verticalalignment='center',
                               transform=ax_test.transAxes)
                    ax_test.text(-0.4, 0.5, 'Accuracy', 
                               fontsize=LEGEND_FONT_SIZE,
                               rotation=90,
                               verticalalignment='center',
                               transform=ax_test.transAxes)

                ax_test.set_ylim(0, 1)
                if exp_index == len(exp_order) - 1:
                    ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE, labelpad=10)
                    ax_test.get_xaxis().set_label_coords(.5,-0.2)

                ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
                ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

                ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

                ax_test.spines['top'].set_visible(False)
                ax_test.spines['right'].set_visible(False)

                offset = 300
                ax_test.set_xlim(0, max_steps + offset)

                if j <= 2:
                    ticks = [1000, 5000, 9000]
                else:
                    ticks = [1000, 2500, 4000]

                if exp_index == len(exp_order) - 1:
                    ax_test.set_xticks(ticks)
                else:
                    ax_test.set_xticks(ticks, ["", "", ""])

                if j !=0:
                    ax_test.set_yticks([0.3, 0.6, 0.9], ["", "", ""])
                else:
                    ax_test.set_yticks([0.3, 0.6, 0.9])

    size_handles = [plt.Line2D([0], [0], color=size_colors[size], lw=linewidth, alpha=alpha,) for size in size_order]
    size_labels = [f"{size}B" for size in size_order]

    
    all_handles = size_handles
    all_labels = size_labels

    fig.legend(
        all_handles,
        all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(size_handles), 
        fancybox=True,
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'qwen_scaling_compact')
    save_fig(fig, pdf_path, FORMAT)


def plot_scaling_qwen_results_compact_context_efficiency(all_experiments):
    """
    Plots test accuracy vs number of in-context examples for different model sizes and tasks.
    Shows how efficiently each model size uses context examples.

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    init_fig_settings()
    experiment_storage.refresh()
    markersize = 9.0
    alpha = 0.6

    exp_order = ["Upper Bound", "Approximate ICRL", "Stochastic ICRL", "Naive ICRL", "Zero-shot", "Naive+ ICRL"]
    legend_order = ["Naive ICRL", "Naive+ ICRL", "Stochastic ICRL"]
    size_order = ['0.5', '1.5', '3', '7', '14', '32', '72']
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        if "unsemantic" in args.task_name:
            continue
        if "qwen2.5" not in args.model_name.lower():
            continue
            
        if not args.icrl:
            exp_type = "Upper Bound"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end" and args.temperature == "1.0":
            exp_type = "Naive ICRL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end_only_positive" and args.temperature == "2.0":
            exp_type = "Naive+ ICRL"
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.temperature == "1.0":
            exp_type = "Stochastic ICRL"
        else:
            continue

        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = args

    models = [(model_key, model_name) for (model_key, model_name) in MODEL_TO_NAME.items() if 'qwen' in model_key.lower()]
    tasks = [(task_key, task_name) for (task_key, task_name) in TASK_TO_NAME.items() if "unsemantic" not in task_key]

    size_colors = {size: get_colors(7)[i] for i, size in enumerate(size_order)}
    
    markers = ['o', 's', '^', 'D', 'P', '*', 'h']
    exp_markers = {exp_type: marker for exp_type, marker in zip(exp_order, markers)}
    
    fig, axs = plt.subplots(
        1, 
        len(tasks), 
        figsize=(13, 5),
        tight_layout=True,
    )
    
    for i, (model_key, model_name) in enumerate(models):
        print(model_name)
        model_size = re.search(r'-(.*?)b-', model_key, re.IGNORECASE).group(1).lower()
        
        for j, (task_key, task_name) in enumerate(tasks):
            experiment_storage.refresh()
            experiments = model_task_experiments[model_key][task_key]
            ax_test = axs[j]

            ax_test.set_ylabel('Test Accuracy', fontsize=LABEL_SIZE)
            ax_test.get_yaxis().set_label_coords(-0.15,.5)

            if i == 0:
                ax_test.text(
                    0.5, 1.05, r'\textbf{' + task_name + '}', 
                    va='center', 
                    ha='center', 
                    transform=ax_test.transAxes,
                    fontsize=TITLE_SIZE
                )

            max_steps = 0
            max_overall_examples = 0
            zero_shot_acc = None
            for exp_type in legend_order:
                if exp_type in experiments:
                    args = experiments[exp_type]
                    data = experiment_storage.load_data(args)
                    color = size_colors[model_size]
                    marker = exp_markers[exp_type]

                    steps = sorted(int(step) for step in data['steps'].keys())
                    max_steps = max(max_steps, max(steps))
                    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]

                    test_steps = [step for step in test_steps if data['steps'][step].get_context_metrics()['context_length'] > 0]
                   
                    test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]
                    num_incontext_examples = [data['steps'][step].get_context_metrics()['context_length'] for step in test_steps]
                    max_overall_examples = max(max(num_incontext_examples), max_overall_examples)

                    if exp_type == "Upper Bound":
                        continue
                    elif exp_type == "Naive ICRL":
                        zero_shot_acc = test_acc[0] if test_acc else 0
                        ax_test.scatter(num_incontext_examples, test_acc, color=color, label=exp_type, alpha=alpha, s=markersize**2, marker=marker)
                        ax_test.scatter(0, zero_shot_acc, color=color, label="Zero-shot", alpha=alpha, s=markersize**2, marker=marker)
                    else:
                        ax_test.scatter(num_incontext_examples, test_acc, color=color, label=exp_type, alpha=alpha, s=markersize**2, marker=marker)

            ax_test.set_ylim(0, 1)
            ax_test.set_xlabel('In-Context Examples', fontsize=LEGEND_FONT_SIZE)
            max_xlim = 3000
            ax_test.set_xticks([max_xlim/3, max_xlim*2/3, max_xlim])
            ax_test.set_xlim(2, max_xlim)
            ax_test.set_xscale('log')
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)

            ax_test.set_yticks([0.3, 0.6, 0.9])

    # Create legend handles for model sizes and experiment types
    size_handles = [plt.Line2D([0], [0], color=size_colors[size], alpha=alpha, marker='o', markersize=markersize, linestyle='None') for size in size_order]
    size_labels = [f"{size}B" for size in size_order]

    exp_handles = [plt.Line2D([0], [0], color='black', marker=exp_markers[exp_type], markersize=markersize, alpha=alpha, linestyle='None') for exp_type in legend_order]
    exp_labels = legend_order

    # Pad handles and labels to align legend columns
    max_len = max(len(size_handles), len(exp_handles))
    dummy_handle = plt.Line2D([], [], color='none')

    left_pad_handles = (max_len - len(size_handles)) // 2
    right_pad_handles = max_len - len(size_handles) - left_pad_handles
    left_pad_labels = (max_len - len(size_labels)) // 2
    right_pad_labels = max_len - len(size_labels) - left_pad_labels

    size_handles_padded = [dummy_handle] * left_pad_handles + size_handles + [dummy_handle] * right_pad_handles
    size_labels_padded = [''] * left_pad_labels + size_labels + [''] * right_pad_labels

    left_pad_exp_handles = (max_len - len(exp_handles)) // 2
    right_pad_exp_handles = max_len - len(exp_handles) - left_pad_exp_handles
    left_pad_exp_labels = (max_len - len(exp_labels)) // 2
    right_pad_exp_labels = max_len - len(exp_labels) - left_pad_exp_labels

    exp_handles_padded = [dummy_handle] * left_pad_exp_handles + exp_handles + [dummy_handle] * right_pad_exp_handles
    exp_labels_padded = [''] * left_pad_exp_labels + exp_labels + [''] * right_pad_exp_labels

    all_handles = [h for pair in zip(size_handles_padded, exp_handles_padded) for h in pair]
    all_labels = [l for pair in zip(size_labels_padded, exp_labels_padded) for l in pair]

    fig.legend(
        all_handles,
        all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=max_len, 
        fancybox=True,
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'qwen_scaling_compact_context_efficiency')
    save_fig(fig, pdf_path, FORMAT)

def plot_scaling_qwen_results_stability(all_experiments):
    """
    Plots the scaling qwen results of the experiments, including test accuracy and regret for different experiment types.

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    init_fig_settings()

    # Define the order of experiment types
    exp_order = ["Upper Bound", "Naive ICRL", "Stochastic ICRL", "Approximate ICRL", "Zero-shot", "Naive+ ICRL"]
    legend_order = ["Naive ICRL", "Naive+ ICRL", "Stochastic ICRL"]
    
    # Define the directory where plots will be saved
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        # Skip all unsemantic tasks
        if "unsemantic" in args.task_name:
            continue
        if "qwen2.5" not in args.model_name.lower():
            continue
        if not args.icrl:
            exp_type = "Upper Bound"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end" and args.temperature == "1.0":
            exp_type = "Naive ICRL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end_only_positive" and args.temperature == "2.0":
            exp_type = "Naive+ ICRL"
            # continue # TODO remove
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.temperature == "1.0":
            exp_type = "Stochastic ICRL"
        else:
            continue  # Skip experiments that don't match any of the four types

        # Assert there is exactly one experiment per model and task and experiment type
        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = args

    # Get unique model names and task names, ensuring specific model order
    models = [(model_key, model_name) for (model_key, model_name) in MODEL_TO_NAME.items() if 'qwen' in model_key.lower()]
    tasks = [(task_key, task_name) for (task_key, task_name) in TASK_TO_NAME.items() if "unsemantic" not in task_key]
    # assert len(model_names) <= 7, f"Expected at most 4 models, got {len(model_names)}"
    # assert len(task_names) <= 5, f"Expected at most 5 tasks, got {len(task_names)}"

    # Define colors for each experiment type
    exp_colors = {exp_type: get_colors(6)[i] for i, exp_type in enumerate(exp_order)}
    markers = ['o', 's', '^', 'D', 'P', '*', 'h']
    
    # Define symbols for different model sizes
    exp_markers = {exp_type: marker for exp_type, marker in zip(exp_order, markers)}
    
    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(
        1, 
        len(tasks), 
        figsize=(13, 3),
        tight_layout=True,
    )

   
    for i, (model_key, model_name) in enumerate(models):
        print(model_name)
        model_size = re.search(r'-(.*?)b-', model_key, re.IGNORECASE).group(1).lower()
        
        # Iterate over tasks for the current model
        for j, (task_key, task_name) in enumerate(tasks):
            experiments = model_task_experiments[model_key][task_key]

            experiment_storage.refresh()

            # Select the upper and lower axes for line plot and bar plot respectively
            ax_test = axs[j]  # Upper plot for test accuracies (line plot)

            # Add subtitles for rows
            if j == 0:
                ax_test.set_ylabel(r'Spearman $\rho$', fontsize=LABEL_SIZE, labelpad=20)

            # Set task label  at the top of each column
            if i == 0:
                ax_test.set_title(r'\textbf{' + task_name + '}', fontsize=TITLE_SIZE, pad=10)


            # Plot data for each experiment type
            max_steps = 0
            for exp_type in legend_order:
                if exp_type in experiments:
                    args = experiments[exp_type]
                    data = experiment_storage.load_data(args)
                    color = exp_colors[exp_type]
                    # color = exp_colors[exp_type]
                    # marker = size_markers[model_size]
                    marker = exp_markers[exp_type]
                    model_size_b = float(model_size) * 1e9

                    # Compute test accuracy and regret
                    steps = sorted(int(step) for step in data['steps'].keys())
                    max_steps = max(max_steps, max(steps))
                    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                    test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                    # Compute Spearman correlation between test_steps and test_acc
                    correlation, _ = spearmanr(test_steps, test_acc)
                    print(f"Spearman correlation for {model_name} {task_name} {exp_type}: {correlation}")
                   
                    ax_test.plot(model_size_b, correlation, color=color, alpha=0.8, marker=marker, markersize=10.)
                

            ax_test.set_xlim(0.1 * 1e9, 100 * 1e9)
            ax_test.set_xticks([val * 1e9 for val in [0.1, 1.0, 10.0, 100.0]])
            ax_test.set_xscale('log') 
            ax_test.set_xticks([val * 1e9 for val in [0.1, 1.0, 10.0, 100.0]])
            # Hide x-ticks
            # Set y-axis limits for test accuracy
            ax_test.set_ylim(-1.1, 1.1)
            


            ax_test.set_xlabel('Model Size', fontsize=LEGEND_FONT_SIZE)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

            # Make ticks larger
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            # Remove top and right spines
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)
            ax_test.spines['bottom'].set_visible(False)

            yticks = [-1.0, -0.5, 0, 0.5, 1.0]
            if j == 0:
                ax_test.set_yticks(yticks)
            else:
                ax_test.set_yticks(yticks, [""] * len(yticks))


    # Show legend with two rows: one for colors (model sizes) and one for symbols (experiment types)
    # Gather all unique handles and labels
    # Gather unique handles and labels

    exp_handles = [plt.Line2D([0], [0], color=exp_colors[exp_type], marker=exp_markers[exp_type], alpha=0.8,
                            linestyle='None', markersize=10) for exp_type in legend_order]
    exp_labels = legend_order

    # Now create the legend
    fig.legend(
        exp_handles,
        exp_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(exp_handles), 
        fancybox=True,
        fontsize=LEGEND_FONT_SIZE,
    )


    # Adjust layout and save figure per model
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'qwen_scaling_stability')
    save_fig(fig, pdf_path, FORMAT)

def plot_naive_plus_temperature_sensitivity(all_experiments):
    """
    Plots the test accuracy and regret for Llama 7B and Qwen 8B (different columns) for Naive+ for different levels of temperature

    Args:
        all_experiments (list): A list of Args objects containing the experiment parameters.
    """
    for strategy in ["random_biased_end", "random_biased_end_only_positive" ]:
        init_fig_settings()
        experiment_storage.refresh()

        experiment_storage.refresh()

        linewidth = 2.0
        markersize = 9.0
        alpha = 0.9

        temperature_order = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']

        plots_dir = get_plots_dir()
        plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
        os.makedirs(plot_dir, exist_ok=True)

        # Group experiments by model name and task name
        model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
        for args in all_experiments:
            if "banking77" != args.task_name:
                continue
            if "Qwen2.5-7B" not in args.model_name and "Llama-3.1-8B" not in args.model_name:
                continue
            
            if not (args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == strategy):
                continue

            if args.icrl_omit_feedback or args.icrl_flip_feedback:
                continue
            
            assert model_task_experiments[args.model_name][args.task_name].get(args.temperature) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {args.temperature}"
            model_task_experiments[args.model_name][args.task_name][args.temperature] = (args, experiment_storage.load_data(args))

        models = [(model_key, model_name) for (model_key, model_name) in MODEL_TO_NAME.items() if 'Qwen2.5-7B' in model_key or 'Llama-3.1-8B' in model_key]
        tasks = [(task_key, task_name) for (task_key, task_name) in TASK_TO_NAME.items() if "banking77" == task_key.lower()]

        temperature_colors = {temperature: get_colors(len(temperature_order))[i] for i, temperature in enumerate(temperature_order)}

        # Create figure with test accuracy and regret subplots
        fig, axs = plt.subplots(
            2, 
            2, 
            figsize=(10, 5),
            gridspec_kw={'height_ratios': [9, 7]},
            tight_layout=True,
        )

        for i, (model_key, model_name) in enumerate(models):
            print(model_name)

            experiment_storage.refresh()

            ax_test = axs[0, i]  # Upper plot for test accuracies
            ax_regret = axs[1, i]  # Lower plot for regrets

            # Add labels
            if i == 0:
                ax_test.set_ylabel('Test Accuracy', fontsize=LABEL_SIZE)
                ax_test.get_yaxis().set_label_coords(-0.15,.5)
                ax_regret.set_ylabel('Regret', fontsize=LABEL_SIZE)
                ax_regret.get_yaxis().set_label_coords(-0.15,0.5)
            
            ax_test.set_title(r'\textbf{' + model_name + '}', fontsize=TITLE_SIZE, pad=10)

            for j, (task_key, task_name) in enumerate(tasks):
                print(task_name)
                fig.text(-0.02, 0.5, r'\textbf{' + task_name + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

                experiments = model_task_experiments[model_key][task_key]

                print(experiments.keys())
                print(temperature_order)

                regrets = []
                max_steps = 0
                for temperature in temperature_order:
                    print(temperature)
                    if temperature in experiments:
                        args, data = experiments[temperature]
                        color = temperature_colors[temperature]

                        steps = sorted(int(step) for step in data['steps'].keys())
                        max_steps = max(max_steps, max(steps))
                        test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                        test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                        training_acc = [data['steps'][step].get_training_data()[4] for step in steps if data['steps'][step].training_step_processed()]
                        regret = len(training_acc) - sum(training_acc)  # Number of wrong training examples
                        regrets.append((temperature, regret))

                        ax_test.plot(test_steps, test_acc, color=color, label=temperature, alpha=alpha, linewidth=linewidth, markersize=markersize)
                    
                sorted_regrets = sorted(regrets, key=lambda x: temperature_order.index(x[0]), reverse=True)
                    
                bars = ax_regret.barh(
                    [r[0] for r in sorted_regrets],
                    [r[1] for r in sorted_regrets],
                    color=[temperature_colors[r[0]] for r in sorted_regrets],
                    edgecolor='black',
                    height=0.5
                )

                # Add regret values next to bars
                for bar in bars:
                    width = bar.get_width()
                    ax_regret.text(
                        width + 0.03 * ax_regret.get_xlim()[1],
                        bar.get_y() + bar.get_height() / 2 - 0.03,
                        f'{int(bar.get_width())}',
                        ha='left',
                        va='center',
                        fontsize=LABEL_SIZE
                    )

                ax_regret.set_xlim(0, max_steps + 100)
                ax_regret.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)

                # Configure regret plot appearance
                ax_regret.grid(False)
                ax_regret.spines['right'].set_visible(False)
                ax_regret.spines['top'].set_visible(False)
                ax_regret.spines['bottom'].set_visible(False)
                ax_regret.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax_regret.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                # Configure test accuracy plot appearance
                ax_test.set_ylim(0, 1)
                ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
                ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
                ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
                ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)
                ax_test.spines['top'].set_visible(False)
                ax_test.spines['right'].set_visible(False)

                offset = 300
                ax_test.set_xticks([1000, 5000, 9000])
                ax_test.set_xlim(0, 10000 + offset)

                y_ticks = [0.3, 0.6, 0.9]
                if i == 0:
                    ax_test.set_yticks(y_ticks)
                else:
                    ax_test.set_yticks(y_ticks, [""] * len(y_ticks))

        # Create temperature legend
        handles = [plt.Line2D([0], [0], color=temperature_colors[temperature], lw=linewidth, alpha=alpha,) for temperature in temperature_order]
        labels = [f"{temperature}" for temperature in temperature_order]

        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.10),
            ncol=len(temperature_order), 
            fancybox=True,
            fontsize=LEGEND_FONT_SIZE,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdf_path = os.path.join(plot_dir, f'temp_sensitivity_{strategy}')
        save_fig(fig, pdf_path, FORMAT)


def plot_approximate_context_sampling_method_comparison(all_experiments):
    """
    Plots the comparison of approximate context sampling methods for two tasks (banking77 and clinic150).
    Each task gets two plots - an accuracy plot showing test accuracy over time, and a regret bar plot.
    
    Args:
        all_experiments (list): List of all experiment arguments.
    """
    linewidth = 2.0
    init_fig_settings()
    experiment_storage.refresh()

    # Filter experiments to get only the ones we want to compare
    tasks = ["banking77", "clinic150"]
    filtered_exps = [
        args for args in all_experiments 
        if (args.context_strategy_name == "approximate_only_positive" and
            args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and
            args.task_name in tasks and
            args.approximate_context_sampling_method in ["uniform", "exact"] and
            "llama" in args.model_name and
            args.max_contexts == "8")
    ]
    filtered_exps = [(args, experiment_storage.load_data(args)) for args in filtered_exps]
    
    # Group experiments by task
    task_experiments = {}
    for args, data in filtered_exps:
        if args.task_name not in task_experiments:
            task_experiments[args.task_name] = []
        task_experiments[args.task_name].append((args, data))

    if len(task_experiments) != 2:
        print(f"Expected 2 tasks, but found {len(task_experiments)}. Adjust filtering if needed.")
        return

    sorted_tasks = sorted(task_experiments.keys(), key=lambda x: tasks.index(x))

    # Set up colors for the sampling methods
    sampling_methods = ["uniform", "exact"]
    color_cycle = get_colors(len(sampling_methods))
    color_map = {m: c for m, c in zip(sampling_methods, color_cycle)}
    
    # Collect data for plotting
    data_to_plot = {}
    for task in sorted_tasks:
        data_to_plot[task] = {}
        for method in sampling_methods:
            data_to_plot[task][method] = {
                "steps": [],
                "test_acc": [],
                "regret": 0
            }
            
        for args, data in task_experiments[task]:
            method = args.approximate_context_sampling_method
            steps_all = sorted(int(s) for s in data['steps'].keys())
            if not steps_all:
                print(f"No steps found for {args.model_name}, task={task}, method={method}")
                continue
            
            # Calculate total regret
            total_regret = 0
            for step in steps_all:
                step_info = data['steps'][step]
                if step_info.training_step_processed():
                    training_data = step_info.get_training_data()
                    accuracies = training_data[4]
                    if isinstance(accuracies, list):
                        total_regret += accuracies.count(0)
                    elif isinstance(accuracies, np.ndarray):
                        total_regret += np.sum(accuracies == 0)
                    else:
                        if accuracies == 0:
                            total_regret += 1
                            
            # Collect test accuracy data
            test_steps = []
            test_acc = []
            for step in steps_all:
                step_info = data['steps'][step]
                if step_info.test_step_processed():
                    test_steps.append(step)
                    test_acc.append(step_info.get_test_metrics().get('accuracy', 0))

            data_to_plot[task][method]["steps"] = test_steps
            data_to_plot[task][method]["test_acc"] = test_acc
            data_to_plot[task][method]["regret"] = total_regret

    # Create figure with 5 columns layout: (acc plot, bar plot), spacer, (acc plot, bar plot)
    fig = plt.figure(figsize=(10, 2))
    gs = fig.add_gridspec(1, 5, width_ratios=[7, 8, 2, 7, 8], hspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])  # First task accuracy
    ax2 = fig.add_subplot(gs[0, 1])  # First task regret
    ax3 = fig.add_subplot(gs[0, 3])  # Second task accuracy  
    ax4 = fig.add_subplot(gs[0, 4])  # Second task regret

    # Plot first task (left block)
    task_left = sorted_tasks[0]
    fig.text(
        0.3, 1.15,
        r'\textbf{' + TASK_TO_NAME[task_left] + '}',
        ha='center',
        va='center',
        fontsize=TITLE_SIZE
    )

    # Accuracy plot for first task
    for method in sampling_methods:
        steps = data_to_plot[task_left][method]["steps"]
        test_acc = data_to_plot[task_left][method]["test_acc"]
        if steps and test_acc:
            ax1.plot(
                steps,
                test_acc,
                label=f'{APPROXIMATE_CONTEXT_SAMPLING_METHOD_TO_NAME[method]}',
                color=color_map[method],
                linewidth=linewidth,
                alpha=0.8
            )
        else:
            print(f"No test accuracy data for task={task_left}, method={method}")

    ax1.set_title("Accuracy", fontsize=LABEL_SIZE)
    ax1.set_xlabel("Step", fontsize=LABEL_SIZE)
    ax1.set_ylim(0, 1.0)
    
    left_max_steps = max((max(data_to_plot[task_left][m]["steps"]) 
                          if data_to_plot[task_left][m]["steps"] else 0)
                         for m in sampling_methods)
    ax1.set_xlim(0, left_max_steps + 100)

    ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_xticks([1000, 5000, 9000])
    ax1.set_yticks([0.3, 0.6, 0.9])

    # Bar plot for first task regret
    regrets_left = [data_to_plot[task_left][m]["regret"] for m in sampling_methods]
    max_regret_left = max(regrets_left) if regrets_left else 0

    bars_left = ax2.bar(
        range(len(sampling_methods)),
        regrets_left,
        color=[color_map[m] for m in sampling_methods],
        edgecolor='black',
        width=0.4 
    )
    ax2.set_title("Regret", fontsize=LABEL_SIZE)

    ax2.grid(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax2.set_xticks([])
    ax2.set_ylim(0, max_regret_left * 1.2)

    # Add regret values above bars
    for bar in bars_left:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.06 * max_regret_left),
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=BARS_TEXT_SIZE,
            rotation=0
        )

    # Plot second task (right block)
    task_right = sorted_tasks[1]
    fig.text(
        0.73, 1.15,
        r'\textbf{' + TASK_TO_NAME[task_right] + '}',
        ha='center',
        va='center',
        fontsize=TITLE_SIZE
    )

    # Accuracy plot for second task
    for method in sampling_methods:
        steps = data_to_plot[task_right][method]["steps"]
        test_acc = data_to_plot[task_right][method]["test_acc"]
        if steps and test_acc:
            ax3.plot(
                steps,
                test_acc,
                label=f'{APPROXIMATE_CONTEXT_SAMPLING_METHOD_TO_NAME[method]}',
                color=color_map[method],
                linewidth=linewidth,
                alpha=0.8
            )
        else:
            print(f"No test accuracy data for task={task_right}, method={method}")

    ax3.set_title("Accuracy", fontsize=LABEL_SIZE)
    ax3.set_xlabel("Step", fontsize=LABEL_SIZE)
    ax3.set_ylim(0, 1.0)
    right_max_steps = max((max(data_to_plot[task_right][m]["steps"]) 
                           if data_to_plot[task_right][m]["steps"] else 0)
                          for m in sampling_methods)
    ax3.set_xlim(0, right_max_steps + 100)

    ax3.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax3.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    ax3.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xticks([1000, 5000, 9000])
    ax3.set_yticks([0.3, 0.6, 0.9])

    # Bar plot for second task regret
    regrets_right = [data_to_plot[task_right][m]["regret"] for m in sampling_methods]
    max_regret_right = max(regrets_right) if regrets_right else 0

    bars_right = ax4.bar(
        range(len(sampling_methods)),
        regrets_right,
        color=[color_map[m] for m in sampling_methods],
        edgecolor='black',
        width=0.4  
    )
    ax4.set_title("Regret", fontsize=LABEL_SIZE)
    ax4.grid(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax4.set_ylim(0, max_regret_right * 1.2)
    ax4.set_xticks([])

    for bar in bars_right:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.06 * max_regret_right),
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=BARS_TEXT_SIZE,
            rotation=0
        )

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(
        handles,
        labels,
        loc='lower center',
        fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, 1.3),
        ncol=len(sampling_methods),
        fancybox=True
    )

    plt.tight_layout(rect=[0.03, 0.0, 1, 0.95])

    # Save figure
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, APPROXIMATE_CONTEXT_STRATEGY_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    pdf_filename = 'approximate_context_sampling_method_comparison'
    pdf_path = os.path.join(plot_dir, pdf_filename)
    save_fig(fig, pdf_path)

    print(f"Saved plot as {pdf_path}")


def plot_approximate_detailed_results(all_experiments):
    """
    Plots detailed results for approximate context sampling methods, showing:
    - Overall test and training accuracy
    - Per-context training accuracy
    - Context sizes over time
    - Context usage patterns
    
    Args:
        all_experiments (list): List of all experiment arguments.
    """
    init_fig_settings()
    experiment_storage.refresh()

    context_strategy = "approximate_only_positive"
    tasks = ["banking77", "clinic150"]
    
    # Filter to get experiments with approximate context sampling
    filtered_experiments = [
        args for args in all_experiments 
        if context_strategy in args.context_strategy_name and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.task_name in tasks and args.max_contexts == "8"
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]
    
    print(f"Number of filtered experiments: {len(filtered_experiments)}")
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, APPROXIMATE_DETAILED_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    for experiment in filtered_experiments:
        args, data = experiment
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.2, wspace=0.2)
        axes = [fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]

        accuracy_palette = get_colors(2)
        max_context_count = len(data['steps'][list(data['steps'].keys())[0]].get_context_metrics()['approx_lengths'])
        context_palette = get_colors(max_context_count)

        accuracy_handles, accuracy_labels = [], []
        context_handles, context_labels = [], []

        print(f"Processing {args.model_name} with max_contexts={args.max_contexts}")

        steps = sorted(int(step) for step in data['steps'].keys())  
        if not steps:
            print("No valid steps found")
            continue

        # Plot test accuracy
        test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
        test_acc = [data['steps'][step].get_test_metrics()["accuracy"] for step in test_steps]
        if test_acc:
            line, = axes[0].plot(test_steps, test_acc, label='Test Accuracy', color=accuracy_palette[0], linewidth=2, alpha=0.7)
            accuracy_handles.append(line)
            accuracy_labels.append('Test Accuracy')

        # Plot training accuracy with windowed average for smoothing
        train_steps = [step for step in steps if data['steps'][step].training_step_processed()]
        train_acc = [data['steps'][step].get_training_data()[4] for step in train_steps]
        if train_acc:
            train_acc_windowed_avg = [np.mean(train_acc[max(0, i - WINDOW_SIZE + 1):i + 1]) for i in range(len(train_acc))]
            line, = axes[0].plot(train_steps, train_acc_windowed_avg, label=f'Train Accuracy ({WINDOW_SIZE}-step avg)', color=accuracy_palette[1], linewidth=3, alpha=0.7)
            accuracy_handles.append(line)
            accuracy_labels.append(f'Train Accuracy ({WINDOW_SIZE}-step avg)')

        # Initialize data structures for tracking context usage
        approx_index_data = {}
        for step in steps:
            if data['steps'][step].context_processed():
                approx_indices = data['steps'][step].get_context_metrics()['approx_lengths'].keys()
                for approx_idx in approx_indices:
                    approx_index_data[approx_idx] = {
                        'usage_window': deque(maxlen=WINDOW_SIZE),
                        'hit_rate': [], 'steps': [], 'context_size': [],
                        'usage_timeline': np.zeros(len(steps)),
                        'train_acc': [], 'train_acc_steps': []
                    }
                break

        # Process context usage data
        for step_idx, step in enumerate(steps):
            context_used = data['steps'][step].get_context_metrics()['approx_index'] if data['steps'][step].context_processed() else None
            
            for approx_idx in approx_index_data.keys():
                if context_used == approx_idx:
                    approx_index_data[approx_idx]['usage_window'].append(1)
                    approx_index_data[approx_idx]['usage_timeline'][step_idx] = 1
                    
                    if data['steps'][step].training_step_processed():
                        train_acc = data['steps'][step].get_training_data()[4]
                        approx_index_data[approx_idx]['train_acc'].append(train_acc)
                        approx_index_data[approx_idx]['train_acc_steps'].append(step)
                else:
                    approx_index_data[approx_idx]['usage_window'].append(0)

                approx_index_data[approx_idx]['hit_rate'].append(np.mean(approx_index_data[approx_idx]['usage_window']))
                approx_index_data[approx_idx]['steps'].append(step)
                approx_index_data[approx_idx]['context_size'].append(data['steps'][step].get_context_metrics()['approx_lengths'][approx_idx])

        # Plot per-context metrics
        for i, (approx_idx, info) in enumerate(approx_index_data.items()):
            color = context_palette[i % len(context_palette)]

            axes[2].plot(info['steps'], info['context_size'], color=color, linewidth=2, alpha=0.7)
            context_handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
            context_labels.append(f'Context #{approx_idx}')

            # Plot context usage timeline
            usage_timeline = info['usage_timeline']
            diff = np.diff(usage_timeline, prepend=0, append=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            durations = ends - starts

            axes[3].broken_barh(list(zip(starts, durations)), (i, 0.8), 
                                facecolors=to_rgba(color, alpha=0.8), 
                                edgecolors='none')

            if info['train_acc']:
                train_acc_windowed = [np.mean(info['train_acc'][max(0, i - WINDOW_SIZE + 1):i + 1]) 
                                      for i in range(len(info['train_acc']))]
                axes[1].plot(info['train_acc_steps'], train_acc_windowed, color=color, linewidth=2, alpha=0.7)

        # Configure axes
        titles = ['Overall Test and Training Accuracy', 
                  f'Per-Context Train Accuracy ({WINDOW_SIZE}-step avg)',
                  'Context Sizes', 'Context Hit Rate']
        
        for i, (ax, title) in enumerate(zip(axes, titles)):
            if i in [2, 3]:  # Bottom row
                ax.set_xlabel('Time Step', fontsize=LABEL_SIZE)
            if i in [0, 2]:  # Left column
                if i == 0:  # Top left
                    ax.set_ylabel('Accuracy', fontsize=LABEL_SIZE)
                    ax.yaxis.set_label_coords(-0.25, 0.5)
                else:  # Bottom left
                    ax.set_ylabel('Context Size', fontsize=LABEL_SIZE)
                    ax.yaxis.set_label_coords(-0.25, 0.5)
            elif i == 3:  # Bottom right
                ax.set_ylabel('Hit Rate', fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        # Standardize y-axis for accuracy plots
        axes[0].set_ylim(0, 1.0)
        axes[1].set_ylim(0, 1.0)
        axes[0].set_yticks([0.3, 0.6, 0.9])
        axes[1].set_yticks([0.3, 0.6, 0.9])

        axes[3].set_yticks([])
        axes[3].set_yticklabels([])

        # Clean up plot appearance
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        # Format axis ticks
        axes[0].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        axes[1].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        axes[2].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        axes[3].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

        # Standardize x-axis limits and ticks
        for ax in axes:
            ax.set_xlim(0, 10000)
            ax.set_xticks([1000, 5000, 9000])

        # Create custom legend handler for context lines
        class HandlerMultiLineCollection(HandlerLineCollection):
            def create_artists(self, legend, artist, xdescent, ydescent, width, height, fontsize, trans):
                x = np.linspace(0, width, 4)
                y = np.zeros_like(x) + height / 2. - ydescent
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=artist.get_colors(),
                                    transform=trans, linewidth=artist.get_linewidth())
                return [lc]

        colors = [context_palette[i] for i in [0, len(context_palette) // 2, len(context_palette) - 1]]
        context_lines = LineCollection([[(0, 0), (1, 0)] for _ in range(3)], 
                                       colors=colors, linewidth=4)

        combined_handles = accuracy_handles + [context_lines]
        combined_labels = accuracy_labels + ['Contexts']

        fig.legend(combined_handles, combined_labels,
                   handler_map={context_lines: HandlerMultiLineCollection()},
                   loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 0.9),
                   ncols=5, fancybox=True)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        pdf_filename = f'{args.model_name.replace("/","_")}_{args.task_name}_{args.approximate_context_sampling_method}'
        pdf_path = os.path.join(plot_dir, pdf_filename)

        # Add title with model, task and sampling method
        fig.text(0.5, 1.05, f'\\textbf{{{MODEL_TO_NAME[args.model_name]}}} - \\textbf{{{TASK_TO_NAME[args.task_name]}}} - \\textbf{{{APPROXIMATE_CONTEXT_SAMPLING_METHOD_TO_NAME[args.approximate_context_sampling_method]}}}', 
                 fontsize=TITLE_SIZE, ha='center', va='top')

        save_fig(fig, pdf_path)

        print(f"Saved plot for approximate_context_sampling_method={args.approximate_context_sampling_method} as {pdf_path}")
