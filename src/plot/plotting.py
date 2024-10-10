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

# PROPER NAMES OF ARGUMENTS
MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
LLAMA_NAME = 'Llama-3.1 8B Instruct'
PHI_NAME = 'Phi-3.5-Mini Instruct'

TASK_TO_NAME = {
    "banking77": "Banking77",
    "clinic150": "CLINIC150",
    "trec_coarse": "TREC",
    "trec_fine": "TREC Fine",
    "nlu": "NLU"
}

MODEL_TO_NAME = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": LLAMA_NAME,
    "microsoft/Phi-3.5-mini-instruct": PHI_NAME
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

    def __init__(self, model_name, task_name, max_context_examples, icrl, context_strategy_name, context_p_keep, icrl_omit_feedback, icrl_flip_feedback, icrl_flip_feedback_prob, temperature, training_seed, max_contexts, approximate_context_sampling_method):
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
                model_name = os.path.join(path_parts[-11], path_parts[-10])

                if context_strategy_name.startswith("approximate"):
                    approx_parts = context_strategy_name.split("_")
                    max_contexts = approx_parts[-1]
                    approximate_context_sampling_method = approx_parts[-2]
                    context_strategy_name = "_".join(approx_parts[:-2])
                else:
                    max_contexts = None
                    approximate_context_sampling_method = None

                if icrl_flip_feedback.startswith("True"):
                    icrl_flip_feedback_prob = float(icrl_flip_feedback.split("_")[-1])
                    icrl_flip_feedback = True
                else:
                    icrl_flip_feedback_prob = None
                    icrl_flip_feedback = False

                if model_name not in MODELS:
                    continue

                args = Args(model_name, task_name, max_context_examples, icrl, context_strategy_name, 
                            context_p_keep, icrl_omit_feedback, icrl_flip_feedback, icrl_flip_feedback_prob, temperature, training_seed, max_contexts, approximate_context_sampling_method)
                
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

        # If the data is already cached, return it
        if args_key in self.data_cache:
            return self.data_cache[args_key]

        # Otherwise, create a new ExperimentDataManager instance and load the data
        data_manager = ExperimentDataManager(args)
        self.data_cache[args_key] = data_manager.data

        return data_manager.data

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
            args.training_seed, args.max_contexts, args.approximate_context_sampling_method
        )

experiment_storage = ExperimentStorage()

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

    # Define the order of experiment types
    exp_order = ["Supervised ICL", "Naive ICRL", "Explorative ICRL", "Approximate ICRL"]

    # Define the directory where plots will be saved
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, MAIN_RESULTS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_task_experiments = defaultdict(lambda: defaultdict(lambda: dict()))
    for args in all_experiments:
        if not args.icrl:
            exp_type = "Supervised ICL"
        elif args.icrl and float(args.context_p_keep) == 1.0 and args.context_strategy_name == "random_biased_end":
            exp_type = "Naive ICRL"
        elif args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback:
            exp_type = "Explorative ICRL"
        elif args.context_strategy_name == "approximate_only_positive" and int(args.max_contexts) == 8 and args.approximate_context_sampling_method == "uniform":
            exp_type = "Approximate ICRL"
        else:
            continue  # Skip experiments that don't match any of the four types

        # Assert there is exactly one experiment per model and task and experiment type
        assert model_task_experiments[args.model_name][args.task_name].get(exp_type) is None, f"Duplicate experiment for {args.model_name}, {args.task_name}, {exp_type}"
        model_task_experiments[args.model_name][args.task_name][exp_type] = (args, experiment_storage.load_data(args))

    # Get unique model names and task names
    model_names = sorted(model_task_experiments.keys())
    task_names = sorted(set(task for tasks in model_task_experiments.values() for task in tasks))

    # Assert in total 10 experiments
    assert len(model_names) == 2, f"Expected 2 models, got {len(model_names)}"
    assert len(task_names) == 5, f"Expected 5 tasks, got {len(task_names)}"

    # Define colors for each experiment type
    exp_colors = {exp_type: sns.color_palette(COLOR_PALETTE, 4)[i] for i, exp_type in enumerate(exp_order)}

    # Define shorter labels for the bar plots
    short_labels = {
        "Supervised ICL": "ICL",
        "Naive ICRL": "Naive ICRL",
        "Explorative ICRL": "Explor. ICRL",
        "Approximate ICRL": "Approx. ICRL"
    }

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(
        4, 
        len(task_names), 
        figsize=(13, 8),
        gridspec_kw={'height_ratios': [3, 2, 3, 2]},
        tight_layout=True,
    )

    # Iterate over each model to create separate plots
    for i, model_name in enumerate(model_names):
        # Add subtitles for rows
        fig.text(-0.018, 0.75, r'\textbf{' + LLAMA_NAME + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)
        fig.text(-0.018, 0.25, r'\textbf{' + PHI_NAME + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

        # Iterate over tasks for the current model
        for j, task_name in enumerate(task_names):
            experiments = model_task_experiments[model_name][task_name]

            # Select the upper and lower axes for line plot and bar plot respectively
            ax_test = axs[0+(i*2), j]  # Upper plot for test accuracies (line plot)
            ax_regret = axs[1+(i*2), j]  # Lower plot for regrets (bar plot)

            # Add subtitles for rows
            if j == 0:
                ax_test.set_ylabel('Test Accuracy', fontsize=LABEL_SIZE)
                ax_test.get_yaxis().set_label_coords(-0.25,.6)
                ax_regret.set_ylabel('Regret', fontsize=LABEL_SIZE)
                ax_regret.get_yaxis().set_label_coords(-0.25,0.6)
          
            # Set task label at the top of each column
            ax_test.set_title(r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, pad=10)

            # Plot data for each experiment type
            regrets = []
            max_steps = 0
            for exp_type in exp_order:
                if exp_type in experiments:
                    args, data = experiments[exp_type]
                    color = exp_colors[exp_type]

                    # Compute test accuracy and regret
                    steps = sorted(int(step) for step in data['steps'].keys())
                    max_steps = max(max_steps, max(steps))
                    test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                    test_acc = [data['steps'][step].get_test_metrics()['accuracy'] for step in test_steps]

                    training_acc = [data['steps'][step].get_training_data()[4] for step in steps if data['steps'][step].training_step_processed()]
                    regret = len(training_acc) - sum(training_acc)  # Number of wrong training examples
                    regrets.append((exp_type, regret))

                    # Plot test accuracy (line plot)
                    if exp_type == "Supervised ICL":
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=4., linestyle=':')
                    else:
                        ax_test.plot(test_steps, test_acc, color=color, label=exp_type, alpha=0.9, linewidth=4.)

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
            if j !=0:
                ax_test.set_yticks([0.2, 0.4, 0.6, 0.8], ["", "", "", ""])
            else:
                ax_test.set_yticks([0.2, 0.4, 0.6, 0.8])

            # Plot regret (horizontal bar plot), omitting standard ICL
            regrets = [r for r in regrets if r[0] != "Supervised ICL"]
            if regrets:
                # Sort the regrets based on the fixed order
                sorted_regrets = sorted([r for r in regrets if r[0] != "Supervised ICL"], key=lambda x: exp_order.index(x[0]) if x[0] in exp_order else len(exp_order), reverse=True)
                
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
    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Reorder the handles and labels to match the exp_order
    handles_labels = sorted(zip(handles, labels), key=lambda x: exp_order.index(x[1]))
    handles, labels = zip(*handles_labels)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fancybox=True, fontsize=LEGEND_FONT_SIZE)

    # Adjust layout and save figure per model
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf_path = os.path.join(plot_dir, 'main_results')
    save_fig(fig, pdf_path)

def plot_p_keep_search_results(all_experiments):
    """
    Plots the results of the p_keep search.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    init_fig_settings()

    context_strategy = "random_unbiased_only_positive"
    filtered_experiments = [
        args for args in all_experiments 
        if args.context_strategy_name == context_strategy and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.task_name == "banking77" and args.max_context_examples == "None"
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

    # Ensure we have exactly two models
    if len(model_experiments) != 2:
        print(f"Expected 2 models, but found {len(model_experiments)}. Adjust the filtering if necessary.")
        return

    # Create figure and gridspec
    fig = plt.figure(figsize=(12, 3))
    gs = fig.add_gridspec(1, 5, width_ratios=[7, 8, 1, 7, 8], hspace=0.2)

    # Add a vertical label for the task name
    fig.text(0.04, 0.5, r'\textbf{' + TASK_TO_NAME['banking77'] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

    # Define a color cycle for better visualization
    color_cycle = sns.color_palette(COLOR_PALETTE, max(len(exps) for exps in model_experiments.values()))

    # Order models by name
    model_experiments = dict(sorted(model_experiments.items(), key=lambda x: x[0]))

    for i, (model_name, experiments) in enumerate(model_experiments.items()):
        ax1 = fig.add_subplot(gs[0, i*3])
        ax2 = fig.add_subplot(gs[0, i*3+1])
        
        # Add model name as title
        if i == 0:
            fig.text(0.305, 1.05, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
        else:
            fig.text(0.75, 1.05, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
        
        # Dictionary to store regret per p_keep
        regret_dict = {}
        p_keep_values = []  # To store and sort p_keep values

        ax1.set_title("Test Accuracy", fontsize=LABEL_SIZE)
        ax2.set_title("Regret", fontsize=LABEL_SIZE)
        
        for args, data in experiments:
            steps = sorted(int(step) for step in data['steps'].keys())
            if not steps:
                print(f"No steps found for {model_name} with p_keep={args.context_p_keep}")
                continue

            p_keep = args.context_p_keep
            p_keep_values.append(p_keep)  # Add p_keep to the list
            print(f"Processing {model_name} with p_keep={p_keep}")
            
            # Calculate regret: number of training examples with accuracy 0
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
                            total_regret += 1  # Assuming at least one example
            regret_dict[p_keep] = total_regret

        # Sort p_keep values
        p_keep_values.sort()

        # Plot accuracy lines with sorted colors
        for j, p_keep in enumerate(p_keep_values):
            color = color_cycle[j]
            label = f'$p_{{keep}}={p_keep}$'  # LaTeX formatting for subscript
            
            # Find the corresponding experiment data
            args, data = next((args, data) for args, data in experiments if args.context_p_keep == p_keep)
            steps = sorted(int(step) for step in data['steps'].keys())
            
            # Test Accuracy plot
            test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
            test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
            if test_acc:
                ax1.plot(
                    test_steps, test_acc, 
                    label=label, 
                    linewidth=4.,
                    color=color, alpha=0.8
                )
            else:
                print(f"No test accuracy data for {model_name} with {label}")

        
        # Finalize Accuracy Plot (ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
        ax1.set_xlim(0, len(steps) + 100)
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

        # Make ticks larger
        ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.set_xticks([1000, 5000, 9000])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

        # Set y-ticks
        ax1.set_yticks([0.2, 0.4, 0.6, 0.8])

        # Create Regret Bar Plot (ax2)
        if regret_dict:
            # Use sorted p_keep_values for consistent ordering
            regrets = [regret_dict[p] for p in p_keep_values]
            bars = ax2.bar(
                [f'$p_{{keep}}={p}$' for p in p_keep_values],
                regrets,
                color=[color_cycle[j] for j in range(len(p_keep_values))],
                edgecolor='black',
                width=0.4
            )
            
            # Hide grid lines
            ax2.grid(False)

            ax2.set_ylim(0, 12000)

            # Hide the spines
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            # Hide y-axis ticks
            ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            # Rotate x-axis labels
            # ax2.set_xticklabels([f'$p_{{keep}}={p}$' for p in p_keep_values], rotation=90, ha='center')
            # No tick labels
            ax2.set_xticks([])

            # Add text annotations above each bar
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

    # Move legend below all plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1),
               ncols=5, fancybox=True)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust bottom margin for legend

    pdf_filename = 'p_keep_search_results'
    pdf_path = os.path.join(plot_dir, pdf_filename)
    save_fig(fig, pdf_path)

    print(f"Saved plot for p_keep search as {pdf_path}")




def plot_ablations(all_experiments):
    """
    Plots the ablation results for the given experiments.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    init_fig_settings()

    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if args.context_strategy_name.startswith("random_unbiased") and args.icrl and (args.task_name == "banking77" or args.task_name == "clinic150") and args.max_context_examples == "None" and "llama" in args.model_name and (args.context_p_keep == "0.1" or args.context_p_keep == "1.0")
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, ABLATIONS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by task name
    task_experiments = {}
    for args, data in filtered_experiments:
        if args.task_name not in task_experiments:
            task_experiments[args.task_name] = []
        task_experiments[args.task_name].append((args, data))

    # Define strategies and their corresponding colors
    colors = sns.color_palette(COLOR_PALETTE, 6)
    strategies = [
        ("Explorative ICRL", colors[0]),
        ("Both pos. and neg. reward", colors[1]),
        ("Only neg. reward", colors[2]),
        ("No reward", colors[3]),
        ("Inverted pos. and neg. reward", colors[4]),
        ("Noisy pos. and neg. reward", colors[5]),
    ]

    # Create figure and gridspec
    fig = plt.figure(figsize=(12, 3))
    gs = fig.add_gridspec(1, 5, width_ratios=[7, 8, 1, 7, 8], hspace=0.2)

    # Add a vertical label for the model name
    fig.text(0.04, 0.5, r'\textbf{' + MODEL_TO_NAME['meta-llama/Meta-Llama-3.1-8B-Instruct'] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

    for i, (task_name, experiments) in enumerate(reversed(task_experiments.items())):
        ax1 = fig.add_subplot(gs[0, i*3])
        ax2 = fig.add_subplot(gs[0, i*3+1])
        
        # Add task name as title
        if i == 0:
            fig.text(0.305, 1.05, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
        else:
            fig.text(0.75, 1.05, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)    

        # Dictionary to store regret per strategy
        regret_dict = {strategy[0]: 0 for strategy in strategies}

        ax1.set_title(r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE)
        ax1.set_title("Test Accuracy", fontsize=LABEL_SIZE)
        ax2.set_title("Regret", fontsize=LABEL_SIZE)

        for args, data in experiments:
            steps = sorted(int(step) for step in data['steps'].keys())
            if not steps:
                print(f"No steps found for {args.model_name}")
                continue

            # Determine the strategy based on the conditions
            if not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.context_strategy_name == "random_unbiased_only_positive" and args.context_p_keep == "0.1":
                strategy = "Explorative ICRL"
            elif not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.context_strategy_name == "random_unbiased" and args.context_p_keep == "0.1":
                strategy = "Both pos. and neg. reward"
            elif not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.context_strategy_name == "random_unbiased_only_negative":
                strategy = "Only neg. reward"
            elif args.icrl_omit_feedback and not args.icrl_flip_feedback:
                strategy = "No reward"
            elif not args.icrl_omit_feedback and args.icrl_flip_feedback and args.icrl_flip_feedback_prob == 1.0:
                strategy = "Inverted pos. and neg. reward"
            elif not args.icrl_omit_feedback and args.icrl_flip_feedback and args.icrl_flip_feedback_prob == 0.1:
                strategy = "Noisy pos. and neg. reward"
            else:
                print(f"Unrecognized strategy configuration for {args.model_name}")
                continue

            color = next(s[1] for s in strategies if s[0] == strategy)
            
            print(f"Processing {args.model_name} with strategy {strategy}")
            
            # Calculate regret: number of training examples with accuracy 0
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
                            total_regret += 1  # Assuming at least one example
            regret_dict[strategy] = total_regret

            # Test Accuracy plot
            test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
            test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
            if test_acc:
                ax1.plot(
                    test_steps, test_acc, 
                    label=strategy, 
                    linewidth=4.,
                    color=color, alpha=0.8
                )
            else:
                print(f"No test accuracy data for {args.model_name} with {strategy}")

            # Running average of training accuracy with convolution smoothing
            training_acc = [
                data['steps'][step].get_training_data()[4] 
                for step in steps if data['steps'][step].training_step_processed()
            ]
            training_steps = [step for step in steps if data['steps'][step].training_step_processed()]

            if training_acc:
                window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
            else:
                print(f"Insufficient training accuracy data for {args.model_name} with {strategy}")

        # Finalize Accuracy Plot (ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
        ax1.set_xlim(0, len(steps) + 100)
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

        # Make ticks larger
        ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

        # Set y-ticks
        ax1.set_yticks([0.2, 0.4, 0.6, 0.8])

        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.set_xticks([1000, 5000, 9000])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

        # Create Regret Bar Plot (ax2)
        if any(regret_dict.values()):
            bars = ax2.bar(
                [strategy[0] for strategy in strategies],
                [regret_dict[strategy[0]] for strategy in strategies],
                color=[strategy[1] for strategy in strategies],
                edgecolor='black',
                width=0.4
            )
            
            # Hide grid lines
            ax2.grid(False)

            ax2.set_ylim(0, 12000)

            # Hide the spines
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            # Hide y-axis ticks
            ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            ax2.set_xticks([])

            # Add text annotations next to each bar
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

    # Move legend below both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1),
               ncols=len(strategies) // 2 if len(strategies) % 2 == 0 else len(strategies) // 2 + 1, fancybox=True)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust bottom margin for legend

    pdf_filename = 'ablations_results'
    pdf_path = os.path.join(plot_dir, pdf_filename)
    save_fig(fig, pdf_path)

    print("Saved plot for ablations")

def plot_ablations_without_regret(all_experiments):
    """
    Plots the ablation results for the given experiments without including regret data.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    init_fig_settings()

    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if args.context_strategy_name.startswith("random_unbiased") and args.icrl and (args.task_name == "banking77" or args.task_name == "clinic150") and args.max_context_examples == "None" and "llama" in args.model_name and (args.context_p_keep == "0.1" or args.context_p_keep == "1.0")
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, ABLATIONS_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by task name
    task_experiments = {}
    for args, data in filtered_experiments:
        if args.task_name not in task_experiments:
            task_experiments[args.task_name] = []
        task_experiments[args.task_name].append((args, data))

    # Define strategies and their corresponding colors
    colors = sns.color_palette(COLOR_PALETTE, 6)
    strategies = [
        ("Explorative ICRL", colors[0]),
        ("Both pos. and neg. reward", colors[1]),
        ("Only neg. reward", colors[2]),
        ("No reward", colors[3]),
        ("Inverted pos. and neg. reward", colors[4]),
        ("Noisy pos. and neg. reward", colors[5]),
    ]

    # Create figure and gridspec
    fig = plt.figure(figsize=(6, 2.5))  # Reduced width to half
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 7], hspace=0.2)  # Removed the regret plot

    # Add a vertical label for the model name
    fig.text(0.00, 0.5, r'\textbf{' + MODEL_TO_NAME['meta-llama/Meta-Llama-3.1-8B-Instruct'] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

    for i, (task_name, experiments) in enumerate(reversed(task_experiments.items())):
        ax1 = fig.add_subplot(gs[0, i])
        
        # Add task name as title
        if i == 0:
            fig.text(0.3, 0.99, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
        else:
            fig.text(0.72, 0.99, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)    

        for args, data in experiments:
            steps = sorted(int(step) for step in data['steps'].keys())
            if not steps:
                print(f"No steps found for {args.model_name}")
                continue

            # Determine the strategy based on the conditions
            if not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.context_strategy_name == "random_unbiased_only_positive" and args.context_p_keep == "0.1":
                strategy = "Explorative ICRL"
            elif not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.context_strategy_name == "random_unbiased" and args.context_p_keep == "0.1":
                strategy = "Both pos. and neg. reward"
            elif not args.icrl_omit_feedback and not args.icrl_flip_feedback and args.context_strategy_name == "random_unbiased_only_negative":
                strategy = "Only neg. reward"
            elif args.icrl_omit_feedback and not args.icrl_flip_feedback:
                strategy = "No reward"
            elif not args.icrl_omit_feedback and args.icrl_flip_feedback and args.icrl_flip_feedback_prob == 1.0:
                strategy = "Inverted pos. and neg. reward"
            elif not args.icrl_omit_feedback and args.icrl_flip_feedback and args.icrl_flip_feedback_prob == 0.1:
                strategy = "Noisy pos. and neg. reward"
            else:
                print(f"Unrecognized strategy configuration for {args.model_name}")
                continue

            color = next(s[1] for s in strategies if s[0] == strategy)
            
            print(f"Processing {args.model_name} with strategy {strategy}")

            # Test Accuracy plot
            test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
            test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
            if test_acc:
                ax1.plot(
                    test_steps, test_acc, 
                    label=strategy, 
                    linewidth=4.,
                    color=color, alpha=0.8
                )
            else:
                print(f"No test accuracy data for {args.model_name} with {strategy}")

        # Finalize Accuracy Plot (ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
        ax1.set_xlim(0, len(steps) + 100)
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

        # Make ticks larger
        ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

        # Set y-ticks
        ax1.set_yticks([0.2, 0.4, 0.6, 0.8])

        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.set_xticks([1000, 5000, 9000])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

    # Move legend below both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1),
               ncols=2, fancybox=True)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust bottom margin for legend

    pdf_filename = 'ablations_results_no_regret'
    pdf_path = os.path.join(plot_dir, pdf_filename)
    save_fig(fig, pdf_path)

    print("Saved plot for ablations without regret")


def plot_approximate_beam_sizes_results(all_experiments):
    """
    Plots the results of the approximate beam sizes experiments.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    context_strategy = "approximate_only_positive"
    tasks = ["banking77", "clinic150"]
    
    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if (context_strategy in args.context_strategy_name and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.task_name in tasks and args.approximate_context_sampling_method == "uniform") 
        or (args.task_name in tasks and args.context_strategy_name == "random_unbiased_only_positive" and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.context_p_keep == "0.1" and args.max_context_examples == "None")
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]

    print(f"Found {len(filtered_experiments)} experiments for approximate beam sizes")
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, APPROX_BEAM_SIZES_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Group experiments by task name and model name
    task_model_experiments = {}
    for args, data in filtered_experiments:
        key = (args.task_name, args.model_name)
        if key not in task_model_experiments:
            task_model_experiments[key] = []
        task_model_experiments[key].append((args, data))
    
    # Get unique task names and model names
    task_names = sorted(set(task_name for task_name, _ in task_model_experiments.keys()))
    model_names = sorted(set(model_name for _, model_name in task_model_experiments.keys()))

    # Iterate over each model and task to plot
    for i, model_name in enumerate(model_names):

        print(f"Processing model {model_name}")

        init_fig_settings()
    
        # Set up the plot style
        sns.set_palette(COLOR_PALETTE)
        
        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(12, 3))
        gs = fig.add_gridspec(1, 5, width_ratios=[7, 8, 1, 7, 8], hspace=0.2)

        # Add a vertical label for the model name
        fig.text(0.04, 0.5, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

        for j, task_name in enumerate(task_names):
            experiments = task_model_experiments.get((task_name, model_name), [])
            if not experiments:
                continue

            # Get axes for test accuracy and regret plots
            ax_test = fig.add_subplot(gs[0, j*3])
            ax_regret = fig.add_subplot(gs[0, j*3+1])

            # Add task name as title
            if j == 0:
                fig.text(0.305, 1.05, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
            else:
                fig.text(0.75, 1.05, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)

            # Set titles for the subplots
            ax_test.set_title("Test Accuracy", fontsize=LABEL_SIZE)
            ax_regret.set_title("Regret", fontsize=LABEL_SIZE)
    
            # Define a color cycle for better visualization
            beam_sizes = []
            for args, _ in experiments:
                if context_strategy in args.context_strategy_name:
                    beam_size = int(args.max_contexts)
                else:
                    beam_size = None  # Represent infinite beam size
                beam_sizes.append(beam_size)
            beam_sizes = sorted(set(beam_sizes), key=lambda x: x if x is not None else float('inf'))
            color_cycle = sns.color_palette(COLOR_PALETTE, len(beam_sizes))
            color_map = {beam_size: color for beam_size, color in zip(beam_sizes, color_cycle)}
    
            # Dictionary to store regret per beam size
            regret_dict = {}

            # Order experiments by beam size
            experiments = sorted(experiments, key=lambda x: int(x[0].max_contexts) if context_strategy in x[0].context_strategy_name else float('inf'))
    
            # Plot test accuracy lines
            for args, data in experiments:
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
                                total_regret += 1  # Assuming at least one example
                regret_dict[beam_size] = total_regret
    
                # Plot test accuracy
                test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
                if test_acc:
                    ax_test.plot(
                        test_steps, test_acc, 
                        label=label, 
                        color=color, alpha=0.8, linewidth=5.
                    )
                else:
                    print(f"No test accuracy data for {task_name} with {label}")
    
            # Finalize Test Accuracy Plot (ax_test)
            ax_test.set_ylim(0, 1)
            ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
            ax_test.set_xlim(0, len(steps) + 100)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

            # Make ticks larger
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            # Remove top and right spines
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)

            ax_test.set_xticks([1000, 5000, 9000])
            ax_test.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

            # Set y-ticks
            ax_test.set_yticks([0.2, 0.4, 0.6, 0.8])
    
            # Create Regret Bar Plot (ax_regret)
            if regret_dict:
                # Use sorted beam_sizes for consistent ordering
                regrets = [regret_dict[b] for b in beam_sizes]
                bars = ax_regret.bar(
                    [f'Beam size = {b}' if b is not None else 'Beam size = ∞' for b in beam_sizes],
                    regrets, 
                    color=[color_map[b] for b in beam_sizes],
                    edgecolor='black',
                    width=0.4
                )
                    
                # Hide grid lines
                ax_regret.grid(False)

                ax_regret.set_ylim(0, 12000)

                # Hide the spines
                ax_regret.spines['right'].set_visible(False)
                ax_regret.spines['top'].set_visible(False)
                ax_regret.spines['left'].set_visible(False)

                # Hide y-axis ticks
                ax_regret.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                # Rotate x-axis labels
                # ax2.set_xticklabels([f'$p_{{keep}}={p}$' for p in p_keep_values], rotation=90, ha='center')
                # No tick labels
                ax_regret.set_xticks([])

                # Add text annotations above each bar
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
    
        # Move legend below the figure
        handles, labels = ax_test.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1),
            ncols=7, fancybox=True)        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        safe_model_name = model_name.replace('/', '_')
        pdf_path = os.path.join(plot_dir, f'{safe_model_name}_approximate_beam_sizes_results')
        save_fig(fig, pdf_path)

def plot_approximate_beam_sizes_results_without_regret(all_experiments):
    """
    Plots the results of the approximate beam sizes experiments without including regret data.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    context_strategy = "approximate_only_positive"
    tasks = ["banking77", "clinic150"]

    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if (context_strategy in args.context_strategy_name and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.task_name in tasks and args.approximate_context_sampling_method == "uniform") 
        or (args.task_name in tasks and args.context_strategy_name == "random_unbiased_only_positive" and args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and args.context_p_keep == "0.1" and args.max_context_examples == "None")
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]

    print(f"Found {len(filtered_experiments)} experiments for approximate beam sizes")
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, APPROX_BEAM_SIZES_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Group experiments by task name and model name
    task_model_experiments = {}
    for args, data in filtered_experiments:
        key = (args.task_name, args.model_name)
        if key not in task_model_experiments:
            task_model_experiments[key] = []
        task_model_experiments[key].append((args, data))
    
    # Get unique task names and model names
    task_names = sorted(set(task_name for task_name, _ in task_model_experiments.keys()))
    model_names = sorted(set(model_name for _, model_name in task_model_experiments.keys()))

    # Iterate over each model and task to plot
    for i, model_name in enumerate(model_names):
        print(f"Processing model {model_name}")

        init_fig_settings()
    
        # Set up the plot style
        sns.set_palette(COLOR_PALETTE)
        
        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(6, 2.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.2)

        # Add a vertical label for the model name
        fig.text(0.00, 0.5, r'\textbf{' + MODEL_TO_NAME[model_name] + '}', ha='left', va='center', fontsize=TITLE_SIZE, rotation=90)

        for j, task_name in enumerate(task_names):
            experiments = task_model_experiments.get((task_name, model_name), [])
            if not experiments:
                continue

            # Get axes
            ax_test = fig.add_subplot(gs[0, j])
           
            # Add task name as title
            if j == 0:
                fig.text(0.3, 0.99, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)
            else:
                fig.text(0.72, 0.99, r'\textbf{' + TASK_TO_NAME[task_name] + '}', ha='center', va='center', fontsize=TITLE_SIZE)

            # Define a color cycle for better visualization
            beam_sizes = []
            for args, _ in experiments:
                if context_strategy in args.context_strategy_name:
                    beam_size = int(args.max_contexts)
                else:
                    beam_size = None  # Represent infinite beam size
                beam_sizes.append(beam_size)
            beam_sizes = sorted(set(beam_sizes), key=lambda x: x if x is not None else float('inf'))
            color_cycle = sns.color_palette(COLOR_PALETTE, len(beam_sizes))
            color_map = {beam_size: color for beam_size, color in zip(beam_sizes, color_cycle)}
    
            # Order experiments by beam size
            experiments = sorted(experiments, key=lambda x: int(x[0].max_contexts) if context_strategy in x[0].context_strategy_name else float('inf'))
           
            # Plot test accuracy lines
            for args, data in experiments:
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
    
                # Test Accuracy plot
                test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
                if test_acc:
                    ax_test.plot(
                        test_steps, test_acc, 
                        label=label, 
                        color=color, alpha=0.8, linewidth=5.
                    )
                else:
                    print(f"No test accuracy data for {task_name} with {label}")
    
            # Finalize Accuracy Plot (ax_test)
            ax_test.set_ylim(0, 1)
            ax_test.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
            ax_test.set_xlim(0, len(steps) + 100)
            ax_test.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
            ax_test.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

            # Make ticks larger
            ax_test.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

            # Remove top and right spines
            ax_test.spines['top'].set_visible(False)
            ax_test.spines['right'].set_visible(False)

            ax_test.set_xticks([1000, 5000, 9000])
            ax_test.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

            # Set y-ticks
            ax_test.set_yticks([0.2, 0.4, 0.6, 0.8])
    
        # Move legend below the figure
        handles, labels = ax_test.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.1),
            ncols=4, fancybox=True)  # Limited to 4 columns
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        safe_model_name = model_name.replace('/', '_')
        pdf_path = os.path.join(plot_dir, f'{safe_model_name}_approximate_beam_sizes_results_without_regret')
        save_fig(fig, pdf_path)

    print("Saved plot for approximate beam sizes without regret")


def plot_naive_plus_vs_explorative_regret_and_accuracy(all_experiments):
    """
    Plots the comparison of regret and accuracy between Naive+ and Explorative ICRL strategies.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if (args.context_strategy_name == "random_unbiased_only_positive" and args.icrl and args.max_context_examples == "None" and args.context_p_keep == "0.1") or 
              (args.context_strategy_name == "random_biased_end_only_positive" and args.icrl and args.max_context_examples == "None" and args.context_p_keep == "1.0")
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, NAIVE_PLUS_COMPARISON_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by model name and task name
    model_experiments = {}
    for args, data in filtered_experiments:
        model_name = args.model_name
        if model_name not in model_experiments:
            model_experiments[model_name] = {}
        if args.task_name not in model_experiments[model_name]:
            model_experiments[model_name][args.task_name] = []
        model_experiments[model_name][args.task_name].append((args, data))

    # Ensure exactly two strategies are present for each model and task
    assert all(len(experiments) == 2 for task_experiments in model_experiments.values() for task_name, experiments in task_experiments.items()), \
        "Expected exactly two strategies for each model and task, but found the following configurations:\n" + \
        "\n".join(str((task_name, [args.context_strategy_name for args, _ in experiments])) for task_experiments in model_experiments.values() for task_name, experiments in task_experiments.items())

    # Define strategies and their corresponding colors
    colors = sns.color_palette(COLOR_PALETTE, 2)
    strategies = [
        ("Explorative ICRL", 0.1, "random_unbiased_only_positive", colors[0]),
        ("Naive+", 1.0, "random_biased_end_only_positive", colors[1])
    ]

    for model_name, task_experiments in model_experiments.items():
        for plot_type in ['regret', 'accuracy']:
            fig, ax = plt.subplots(figsize=(20, 12))

            tasks = list(task_experiments.keys())
            x = range(len(tasks))
            width = 0.35

            all_values = []  # List to store regrets or accuracies for all strategies

            for i, (strategy_name, p_keep, strategy_context_name, color) in enumerate(strategies):
                values = []
                for task_name in tasks:
                    value = 0
                    for args, data in task_experiments[task_name]:
                        if args.context_p_keep == str(p_keep) and args.context_strategy_name == strategy_context_name:
                            steps = sorted(int(step) for step in data['steps'].keys())
                            if plot_type == 'regret':
                                for step in steps:
                                    if data['steps'][step].training_step_processed():
                                        training_data = data['steps'][step].get_training_data()
                                        accuracies = training_data[4]
                                        if isinstance(accuracies, list):
                                            value += accuracies.count(0)
                                        elif isinstance(accuracies, np.ndarray):
                                            value += np.sum(accuracies == 0)
                                        else:
                                            if accuracies == 0:
                                                value += 1
                            elif plot_type == 'accuracy':
                                test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
                                if test_steps:
                                    last_test_step = max(test_steps)
                                    value = data['steps'][last_test_step].get_test_metrics()['accuracy']
                    values.append(value)

                ax.bar([t + i * width for t in x], values, width, label=strategy_name, color=color, edgecolor='black')
                all_values.append(values)

            ylabel = 'Regret' if plot_type == 'regret' else 'Last Test Accuracy'
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
            ax.set_xticks([t + width / 2 for t in x])
            ax.set_xticklabels(tasks)
            ax.legend(fontsize=LEGEND_FONT_SIZE, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                       fancybox=True, shadow=True)

            ax.set_xlabel('Task', fontsize=LABEL_SIZE)
            ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
            ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

            # Remove top and right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add text annotations above each bar
            for i, values in enumerate(all_values):
                for j, v in enumerate([t + i * width for t in x]):
                    ax.text(v, values[j], f'{values[j]:.2f}', ha='center', va='bottom', fontsize=LEGEND_FONT_SIZE)

            plt.tight_layout()
            
            # Save the plot as a PDF
            pdf_filename = f'{model_name.replace("/", "_")}_{plot_type}_comparison'
            pdf_path = os.path.join(plot_dir, pdf_filename)
            save_fig(fig, pdf_path)

def plot_context_length_and_strategy_comparison(all_experiments):
    """
    Plots the comparison of context length and strategy for the given experiments.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    init_fig_settings()

    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and (args.task_name == "banking77" or args.task_name == "clinic150") and "llama" in args.model_name and "approximate" not in args.context_strategy_name and "only_positive" in args.context_strategy_name and args.context_p_keep == "0.1"
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, CONTEXT_LENGTH_STRATEGY_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by task
    task_experiments = {}
    for args, data in filtered_experiments:
        if args.task_name not in task_experiments:
            task_experiments[args.task_name] = []
        task_experiments[args.task_name].append((args, data))

    # Define color palette and line styles globally (applies to all tasks)
    color_palette = sns.color_palette(COLOR_PALETTE, len(set(args.max_context_examples for args, _ in filtered_experiments)))
    strategy_names = sorted(set(args.context_strategy_name for args, _ in filtered_experiments))
    line_styles = ['-', '--', ':', '-.']
    line_style_map = dict(zip(strategy_names, line_styles[:len(strategy_names)]))

    for task_name, experiments in task_experiments.items():
        # Create a figure for each task
        fig = plt.figure(figsize=(10, 3))  # Reduced height to half
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 3], height_ratios=[1, 1, 1], hspace=0.6)

        # Define color map specific to max_context_examples
        max_context_examples_values = sorted(set(args.max_context_examples for args, _ in experiments), key=lambda x: int(x) if x != "None" else float('inf'))
        color_map = dict(zip(max_context_examples_values, color_palette))

        # Initialize maximum regret for scaling bar plots
        max_regret = 0

        # Main accuracy plot
        ax1 = fig.add_subplot(gs[:, 0])

        # Dictionary to store regret data for bar plots
        regret_data = {strategy: {} for strategy in strategy_names}

        for args, data in experiments:
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

            # Test Accuracy plot
            test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
            test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
            if test_acc:
                ax1.plot(
                    test_steps, test_acc, 
                    label=f'W={args.max_context_examples}, strategy={args.context_strategy_name}', 
                    linestyle=line_style_map[args.context_strategy_name],
                    color=color_map[args.max_context_examples],
                    linewidth=4,
                    alpha=0.7
                )
            else:
                print(f"No test accuracy data for {args.model_name} with max_context={args.max_context_examples}, strategy={args.context_strategy_name}")

        # Finalize Accuracy Plot (ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
        ax1.set_xlim(0, len(steps) + 100)
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.set_ylabel('Test Accuracy', fontsize=LABEL_SIZE)

        # Make ticks larger
        ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.set_xticks([1000, 5000, 9000])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

        ax1.set_yticks([0.2, 0.4, 0.6, 0.8])

        # Text annotation for the dataset
        ax1.text(-0.2, 0.5, r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, transform=ax1.transAxes, rotation=90, ha='center', va='center')
        ax1.text(1.1, 0.5, 'Regret', ha='center', va='center', fontsize=LABEL_SIZE, transform=ax1.transAxes, rotation=90)

        # Create Horizontal Bar Plots for each strategy
        all_regret_bars = []
        for i, strategy in enumerate(strategy_names[:3]):
            ax_bar = fig.add_subplot(gs[i, 1])
            regrets = [regret_data[strategy][max_context_examples] for max_context_examples in max_context_examples_values[:2]]
            bars = ax_bar.barh(
                [f'{max_context}' for max_context in max_context_examples_values[:2]],
                regrets,
                color=[color_map[max_context] for max_context in max_context_examples_values[:2]],
                edgecolor='black',
                height=0.6
            )

            # Set x-axis limits for the regret plot
            ax_bar.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)  # Reduce y-tick label size

            # Add title as the strategy name
            ax_bar.set_title(STRATEGY_TO_NAME[strategy], fontsize=LABEL_SIZE)

            # Remove spines and ticks for regret plot
            ax_bar.grid(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['bottom'].set_visible(False)
            ax_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_bar.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            all_regret_bars.append(ax_bar)

            # Add text annotations next to each bar
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

        # Adjust x-limits of bar plots
        for regret_bar in all_regret_bars:
            regret_bar.set_xlim(0, max_regret + 100)

        # Add legend to the top with colors and line styles
        color_handles = [plt.Line2D([0], [0], color=color_palette[i], lw=4) for i in range(len(max_context_examples_values))]
        
        # Map the actual max_context_examples_values to labels "4K", "8K", "128K"
        color_labels = [f'$W={max_tokens}$' for max_tokens in ["4K", "8K", "128K"]]

        line_handles = [plt.Line2D([0], [0], color='black', linestyle=line_style_map[strategy], lw=2) for strategy in strategy_names[:3]]
        line_labels = [STRATEGY_TO_NAME[strategy] for strategy in strategy_names[:3]]

        handles = [color_handles[i // 2] if i % 2 == 0 else line_handles[i // 2] for i in range(6)]
        labels = [color_labels[i // 2] if i % 2 == 0 else line_labels[i // 2] for i in range(6)]

        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.3), fancybox=True)

        # Add text with the model name centered in the middle of the entire figure
        fig.text(0.5, 0.96, r'\textbf{' + MODEL_TO_NAME['meta-llama/Meta-Llama-3.1-8B-Instruct'] + '}', ha='center', va='center', fontsize=TITLE_SIZE)

        # Save the plot
        pdf_filename = f'context_length_and_strategy_comparison_{task_name}'
        pdf_path = os.path.join(plot_dir, pdf_filename)
        save_fig(fig, pdf_path)

        print(f"Saved plot for {task_name} as {pdf_path}")

def plot_approximate_context_sampling_method_comparison(all_experiments):
    """
    Plots the comparison of approximate context sampling methods for the given experiments.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    init_fig_settings()

    tasks = ["banking77", "clinic150"]
    
    # Filter experiments based on specific criteria
    filtered_experiments = [
        args for args in all_experiments 
        if (args.context_strategy_name == "approximate_only_positive" and
            args.icrl and not args.icrl_flip_feedback and not args.icrl_omit_feedback and
            args.task_name in tasks and
            args.approximate_context_sampling_method in ["uniform", "exact"] and
            "llama" in args.model_name and
            args.max_contexts == "8")
    ]
    filtered_experiments = [(args, experiment_storage.load_data(args)) for args in filtered_experiments]

    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, APPROXIMATE_CONTEXT_STRATEGY_SUBDIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Group experiments by task
    task_experiments = {}
    for args, data in filtered_experiments:
        if args.task_name not in task_experiments:
            task_experiments[args.task_name] = []
        task_experiments[args.task_name].append((args, data))

    # Define color palette globally (applies to all tasks)
    color_palette = sns.color_palette(COLOR_PALETTE, 2)  # Only two colors needed for uniform and exact
    sampling_methods = ["uniform", "exact"]
    color_map = dict(zip(sampling_methods, color_palette))

    for task_name, experiments in task_experiments.items():
        # Create a figure for each task
        fig = plt.figure(figsize=(10, 3))  # Reduced height to half
        gs = fig.add_gridspec(2, 2, width_ratios=[5, 3], height_ratios=[1, 1], hspace=0.6)

        # Initialize maximum regret and accuracy for scaling bar plots
        max_regret = 0
        max_accuracy = 0

        # Main accuracy plot
        ax1 = fig.add_subplot(gs[:, 0])

        # Dictionary to store regret and accuracy data for bar plots
        regret_data = {method: 0 for method in sampling_methods}
        accuracy_data = {method: 0 for method in sampling_methods}

        for args, data in experiments:
            steps = sorted(int(step) for step in data['steps'].keys())
            if not steps:
                print(f"No steps found for {args.model_name} with sampling method={args.approximate_context_sampling_method}")
                continue

            # Calculate regret (number of incorrect predictions during training)
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
            regret_data[args.approximate_context_sampling_method] = total_regret

            # Plot test accuracy
            test_steps = [step for step in steps if data['steps'][step].test_step_processed()]
            test_acc = [data['steps'][step].get_test_metrics().get('accuracy', 0) for step in test_steps]
            if test_acc:
                ax1.plot(
                    test_steps, test_acc, 
                    label=f'Method={args.approximate_context_sampling_method}', 
                    color=color_map[args.approximate_context_sampling_method],
                    linewidth=4,
                    alpha=0.7
                )
                last_accuracy = test_acc[-1]
                max_accuracy = max(max_accuracy, last_accuracy)
                accuracy_data[args.approximate_context_sampling_method] = last_accuracy
            else:
                print(f"No test accuracy data for {args.model_name} with sampling method={args.approximate_context_sampling_method}")

        # Finalize Accuracy Plot (ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step', fontsize=LEGEND_FONT_SIZE)
        ax1.set_xlim(0, len(steps) + 100)
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        ax1.set_ylabel('Test Accuracy', fontsize=LABEL_SIZE)

        # Make ticks larger
        ax1.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax1.set_xticks([1000, 5000, 9000])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

        ax1.set_yticks([0.2, 0.4, 0.6, 0.8])

        # Text annotation for the dataset
        ax1.text(-0.2, 0.5, r'\textbf{' + TASK_TO_NAME[task_name] + '}', fontsize=TITLE_SIZE, transform=ax1.transAxes, rotation=90, ha='center', va='center')

        # Create Horizontal Bar Plots for regret and accuracy
        ax_regret = fig.add_subplot(gs[0, 1])
        ax_accuracy = fig.add_subplot(gs[1, 1])

        for method in sampling_methods:
            ax_regret.barh(
                [method],
                [regret_data[method]],
                color=color_map[method],
                edgecolor='black',
                height=0.6
            )
            
            ax_accuracy.barh(
                [method],
                [accuracy_data[method]],
                color=color_map[method],
                edgecolor='black',
                height=0.6
            )

        # Configure regret and accuracy plots
        for ax in [ax_regret, ax_accuracy]:
            ax.grid(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)

        # Add text annotations next to each bar
        for ax, data in [(ax_regret, regret_data), (ax_accuracy, accuracy_data)]:
            for method in sampling_methods:
                value = data[method]
                ax.text(
                    value * 1.03,
                    ax.get_yticks()[sampling_methods.index(method)],
                    f'{int(value)}' if ax == ax_regret else f'{value:.2f}',
                    ha='left',
                    va='center',
                    fontsize=BARS_TEXT_SIZE
                )

        # Adjust x-limits of bar plots
        ax_regret.set_xlim(0, max_regret * 1.1)
        ax_accuracy.set_xlim(0, 1)

        # Add y-axis labels for the bar plots
        ax_regret.set_ylabel('Regret', fontsize=LABEL_SIZE)
        ax_accuracy.set_ylabel('Accuracy', fontsize=LABEL_SIZE)

        # Add legend to the top with colors
        color_handles = [plt.Line2D([0], [0], color=color_map[method], lw=4) for method in sampling_methods]
        color_labels = [f'{APPROXIMATE_CONTEXT_SAMPLING_METHOD_TO_NAME[method]}' for method in sampling_methods]

        fig.legend(color_handles, color_labels, loc='upper center', ncol=2, fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.2), fancybox=True)

        # Add text with the model name centered in the middle of the entire figure
        fig.text(0.5, 0.96, r'\textbf{' + MODEL_TO_NAME['meta-llama/Meta-Llama-3.1-8B-Instruct'] + '}', ha='center', va='center', fontsize=TITLE_SIZE)

        # Save the plot
        pdf_filename = f'approximate_context_sampling_method_comparison_{task_name}'
        pdf_path = os.path.join(plot_dir, pdf_filename)
        save_fig(fig, pdf_path)

        print(f"Saved plot for {task_name} as {pdf_path}")


def plot_approximate_detailed_results(all_experiments):
    """
    Plots detailed results for approximate context sampling methods.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    init_fig_settings()

    context_strategy = "approximate_only_positive"
    tasks = ["banking77", "clinic150"]
    
    # Filter experiments based on specific criteria
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

        accuracy_palette = sns.color_palette(COLOR_PALETTE, 2)
        max_context_count = len(data['steps'][list(data['steps'].keys())[0]].get_context_metrics()['approx_lengths'])
        context_palette = sns.color_palette(COLOR_PALETTE, n_colors=max_context_count)

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

        # Plot training accuracy with a windowed average
        train_steps = [step for step in steps if data['steps'][step].training_step_processed()]
        train_acc = [data['steps'][step].get_training_data()[4] for step in train_steps]
        if train_acc:
            train_acc_windowed_avg = [np.mean(train_acc[max(0, i - WINDOW_SIZE + 1):i + 1]) for i in range(len(train_acc))]
            line, = axes[0].plot(train_steps, train_acc_windowed_avg, label=f'Train Accuracy ({WINDOW_SIZE}-step avg)', color=accuracy_palette[1], linewidth=3, alpha=0.7)
            accuracy_handles.append(line)
            accuracy_labels.append(f'Train Accuracy ({WINDOW_SIZE}-step avg)')

        # Process and plot context data
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

        for i, (approx_idx, info) in enumerate(approx_index_data.items()):
            color = context_palette[i % len(context_palette)]

            axes[2].plot(info['steps'], info['context_size'], color=color, linewidth=2, alpha=0.7)
            context_handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
            context_labels.append(f'Context #{approx_idx}')

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

        # Set axis properties
        titles = ['Overall Test and Training Accuracy', 
                  f'Per-Context Train Accuracy ({WINDOW_SIZE}-step avg)',
                  'Context Sizes', 'Context Hit Rate']
        
        for i, (ax, title) in enumerate(zip(axes, titles)):
            if i in [2, 3]:  # Bottom row
                ax.set_xlabel('Time Step', fontsize=LABEL_SIZE)
            if i in [0, 2]:  # Left column
                if i == 0:  # Top left
                    ax.set_ylabel('Accuracy', fontsize=LABEL_SIZE)
                    ax.yaxis.set_label_coords(-0.25, 0.5)  # Move the label further out
                else:  # Bottom left
                    ax.set_ylabel('Context Size', fontsize=LABEL_SIZE)
                    ax.yaxis.set_label_coords(-0.25, 0.5)  # Move the label further out
            elif i == 3:  # Bottom right
                ax.set_ylabel('Hit Rate', fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        # Set same y-axis limits for accuracy plots
        max_accuracy = max(max(test_acc), max(train_acc_windowed_avg))
        axes[0].set_ylim(0, 1.0)
        axes[1].set_ylim(0, 1.0)
        # Show the same y labels as in the other plots (0.1, 0.5, 0.9)
        axes[0].set_yticks([0.2, 0.4, 0.6, 0.8])
        axes[1].set_yticks([0.2, 0.4, 0.6, 0.8])

        axes[3].set_yticks([])
        axes[3].set_yticklabels([])

        # Hide spines for accuracy plots
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        # Set major formatter
        axes[0].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        axes[1].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        axes[2].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
        axes[3].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

        # Set x limits
        axes[0].set_xlim(0, 10000)
        axes[1].set_xlim(0, 10000)
        axes[2].set_xlim(0, 10000)
        axes[3].set_xlim(0, 10000)
        axes[0].set_xticks([1000, 5000, 9000])
        axes[1].set_xticks([1000, 5000, 9000])
        axes[2].set_xticks([1000, 5000, 9000])
        axes[3].set_xticks([1000, 5000, 9000])

        # Add legend
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

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust bottom margin for legend

        pdf_filename = f'{args.model_name.replace("/","_")}_{args.task_name}_{args.approximate_context_sampling_method}'
        pdf_path = os.path.join(plot_dir, pdf_filename)

        # Set title
        fig.text(0.5, 1.05, f'\\textbf{{{MODEL_TO_NAME[args.model_name]}}} - \\textbf{{{TASK_TO_NAME[args.task_name]}}} - \\textbf{{{APPROXIMATE_CONTEXT_SAMPLING_METHOD_TO_NAME[args.approximate_context_sampling_method]}}}', 
                 fontsize=TITLE_SIZE, ha='center', va='top')

        save_fig(fig, pdf_path)

        print(f"Saved plot for approximate_context_sampling_method={args.approximate_context_sampling_method} as {pdf_path}")
def plot_compact_naive_heatmap(all_experiments):
    """
    Plots a compact heatmap comparing naive and explorative ICRL strategies.

    Args:
        all_experiments (list): List of all experiment arguments.

    Returns:
        None
    """
    init_fig_settings()

    # Filter experiments for naive ICRL
    naive_experiments = [
        args for args in all_experiments if args.icrl and args.context_p_keep == "1.0" and args.context_strategy_name == "random_biased_end" and args.task_name == "banking77" and "llama" in args.model_name
    ]
    assert (len(naive_experiments) == 1), f"There should be only one naive experiment, but found more: {naive_experiments}"
    naive_experiment = (naive_experiments[0], experiment_storage.load_data(naive_experiments[0]))
    
    # Filter experiments for explorative ICRL
    explorative_experiments = [
        args for args in all_experiments if args.icrl and args.context_p_keep == "0.1" and args.context_strategy_name == "random_unbiased_only_positive" and args.task_name == "banking77" and "llama" in args.model_name and args.max_context_examples == "None"
    ]
    assert (len(explorative_experiments) == 1), f"There should be only one explorative experiment, but found more: {explorative_experiments}"
    explorative_experiment = (explorative_experiments[0], experiment_storage.load_data(explorative_experiments[0]))
    
    # Show args for the experiments
    print("Naive Experiment Args:", naive_experiment[0].__dict__)
    print("Explorative Experiment Args:", explorative_experiment[0].__dict__)
    
    plots_dir = get_plots_dir()
    plot_dir = os.path.join(plots_dir, NAIVE_HEATMAPS_SUBIDR)
    os.makedirs(plot_dir, exist_ok=True)

    # Get steps and data for both experiments
    args_naive, data_naive = naive_experiment
    args_explorative, data_explorative = explorative_experiment

    # Get test steps
    test_steps_naive = sorted(int(step) for step in data_naive['steps'].keys() if data_naive['steps'][step].test_step_processed())
    test_steps_explorative = sorted(int(step) for step in data_explorative['steps'].keys() if data_explorative['steps'][step].test_step_processed())
    
    if len(test_steps_naive) < 1 or len(test_steps_explorative) < 1:
        print(f"Not enough test steps for {args_naive.model_name}, {args_naive.task_name}")
        return

    # Get step 0 data (same for both experiments)
    step0_data = data_naive['steps'][0]

    # Get last test steps for both experiments
    last_step_naive = test_steps_naive[-1]
    last_step_explorative = test_steps_explorative[-1]

    # Load task and get labels
    unique_labels = list(load_task(args_naive.task_name, verbose=False).get_labels())

    # Create figure with 1 row and 3 columns
    fig = plt.figure(figsize=(12, 4))  # Increased figure size
    outer_gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.1)  # Three heatmaps and one colorbar

    # Calculate the common vmax
    def get_max_value(step_data):
        """
        Get the maximum value from the confusion matrix of the given step data.

        Args:
            step_data: The step data containing test data.

        Returns:
            int: The maximum value in the confusion matrix.
        """
        labels = step_data.get_test_data()[3]
        preds = step_data.get_test_data()[1]
        if len(labels) > 0 and len(preds) > 0:
            cm = confusion_matrix(labels, preds, labels=unique_labels)
            return np.max(cm)
        return 0

    vmax = max(
        get_max_value(step0_data),
        get_max_value(data_naive['steps'][last_step_naive]),
        get_max_value(data_explorative['steps'][last_step_explorative])
    )

    heatmap_cmap = plt.cm.get_cmap('rocket')

    # Helper function to plot a single confusion matrix with marginals
    def plot_confusion_matrix_with_marginals(step_data, title, subplot_spec):
        """
        Plot a confusion matrix with marginal distributions.

        Args:
            step_data: The step data containing test data.
            title (str): The title for the plot.
            subplot_spec: The subplot specification for the plot.

        Returns:
            None
        """
        # Create the nested GridSpec
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2,  
            width_ratios=[10, 1],  
            height_ratios=[1, 10], 
            subplot_spec=subplot_spec,
            wspace=0.05,
            hspace=0.05
        )
        ax_main = fig.add_subplot(inner_gs[1, 0])

        # Get labels and predictions
        labels = step_data.get_test_data()[3]  # Ground truth
        preds = step_data.get_test_data()[1]   # Predictions

        if len(labels) > 0 and len(preds) > 0:
            cm = confusion_matrix(labels, preds, labels=unique_labels)

            # Plot heatmap on ax_main (not normalized)
            sns.heatmap(
                cm,
                annot=False,
                ax=ax_main,
                cmap=heatmap_cmap,
                cbar=False,  # No colorbar for individual plots
                linewidths=0.0,
                vmin=0,
                vmax=vmax,
                xticklabels=False,  # Remove x-axis labels
                yticklabels=False   # Remove y-axis labels
            )
            # Set background color to black
            ax_main.set_facecolor('black')
            
            # Put text title on top of heatmap + distribution
            ax_main.text(0.5, 1.15, r'\textbf{' + title + '}', horizontalalignment='center', fontsize=TITLE_SIZE, transform=ax_main.transAxes)

            # Remove all ticks and labels
            ax_main.set_xticks([])
            ax_main.set_yticks([])

            # Plot marginal distributions
            ax_top = fig.add_subplot(inner_gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(inner_gs[1, 1], sharey=ax_main)

            pred_counts = np.sum(cm, axis=0)  # Sum over true labels
            true_counts = np.sum(cm, axis=1)  # Sum over predicted labels
            
            # Normalize the counts for the bars
            pred_counts_normalized = pred_counts
            true_counts_normalized = true_counts
            
            # Compute positions shifted by +0.5 to align with heatmap squares
            positions = np.arange(len(unique_labels)) + 0.5

            # For ax_top
            ax_top.bar(positions, pred_counts_normalized, color='grey', align='center')
            ax_top.set_xlim(ax_main.get_xlim())
            ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['left'].set_visible(False)
            ax_top.tick_params(axis='y', left=False)
            ax_top.grid(False)
            ax_top.set_yticks([])

            # For ax_right
            ax_right.barh(positions, true_counts_normalized, color='grey', align='center')
            ax_right.set_ylim(ax_main.get_ylim())
            ax_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['bottom'].set_visible(False)
            ax_right.tick_params(axis='x', bottom=False)
            ax_right.set_xticks([])
            ax_right.grid(False)

        else:
            # Hide axes if no data
            ax_main.set_visible(False)
            ax_top.set_visible(False)
            ax_right.set_visible(False)

    # Plot Column 1: Step 0
    subplot_spec_0 = outer_gs[0, 0]
    plot_confusion_matrix_with_marginals(
        step_data=step0_data,
        title="Zero-Shot",
        subplot_spec=subplot_spec_0
    )

    # Plot Column 2: Last step naive
    step_data_naive = data_naive['steps'][last_step_naive]
    subplot_spec_1 = outer_gs[0, 1]
    plot_confusion_matrix_with_marginals(
        step_data=step_data_naive,
        title="Naive ICRL",
        subplot_spec=subplot_spec_1
    )

    # Plot Column 3: Last step explorative
    step_data_explorative = data_explorative['steps'][last_step_explorative]
    subplot_spec_2 = outer_gs[0, 2]
    plot_confusion_matrix_with_marginals(
        step_data=step_data_explorative,
        title="Explorative ICRL",
        subplot_spec=subplot_spec_2
    )

    # Add a single colorbar for all heatmaps
    cbar_ax = fig.add_subplot(outer_gs[0, 3])
    sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)

    # Add true label and predicted label only once
    fig.text(0.1, 0.5, 'True Labels', va='center', rotation='vertical', fontsize=LABEL_SIZE)
    fig.text(0.5, 0.05, 'Predicted Labels', ha='center', fontsize=LABEL_SIZE)

    # Adjust the layout and save the plot
    plt.tight_layout()
    
    # Save the plot
    safe_model_name = args_naive.model_name.replace('/', '_')
    pdf_filename = f'{safe_model_name}_{args_naive.task_name}_naive_vs_explorative_heatmaps'
    pdf_path = os.path.join(plot_dir, f'{pdf_filename}')
    save_fig(fig, pdf_path)

    print(f"Saved naive vs explorative heatmap plot for {args_naive.model_name}, {args_naive.task_name} as {pdf_filename}")
