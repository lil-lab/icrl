import os
from collections import defaultdict



from src.experiments.utils import get_experiment_dir, MAXIMUM_CONTEXT_LENGTH_FILENAME
from src.icl.context_generators import load_context_generator
from src.task.task import load_task
from src.model.model import VLLMModelTokenizerOnlyWrapper
from src.utils.random import set_seed_everywhere
from src.utils.logger import get_data_dir, save
from src.plot.plotting import extract_data, ExperimentStorage

from tqdm import tqdm


from src.utils.logger import read

TOTAL_TOKENS_PROCESSED_FILENAME = "total_tokens_processed.json"


def main():
    experiment_storage = ExperimentStorage()
    all_experiments_args = extract_data()

    filtered_experiments_args = []
    for args in all_experiments_args:
        if args.icrl and args.max_context_examples == "None" and args.context_strategy_name == "random_unbiased_only_positive" and float(args.context_p_keep) == 0.1 and not args.icrl_omit_feedback and not args.icrl_flip_feedback:
            exp_type = "Explorative ICRL"
        elif args.context_strategy_name == "approximate_only_positive" and int(args.max_contexts) == 8 and args.approximate_context_sampling_method == "uniform":
            exp_type = "Approximate ICRL"
        else:
            continue
        filtered_experiments_args.append(args)

    assert len(filtered_experiments_args) == 20, "Expected 20 experiments, got {}".format(len(filtered_experiments_args))

    # TODO delete this
    for args in all_experiments_args:
        if not args.icrl and "llama" in args.model_name and args.task_name == "banking77":
            filtered_experiments_args = [args]
            break
    assert len(filtered_experiments_args) == 1, "Expected 1 experiment, got {}".format(len(filtered_experiments_args))
    
    results = {}
    grouped_results = defaultdict(lambda: defaultdict(dict))

    for experiment_args in filtered_experiments_args:
        experiment_dir = get_experiment_dir(experiment_args)
        total_tokens_path = os.path.join(experiment_dir, TOTAL_TOKENS_PROCESSED_FILENAME)

        # Check if the experiment has already been processed
        if os.path.exists(total_tokens_path):
            print(f"Skipping experiment {experiment_args} because it has already been processed")
            experiment_total_tokens_processed = read(total_tokens_path)
            results[experiment_dir] = experiment_total_tokens_processed
            grouped_results[experiment_args.model_name][experiment_args.task_name]["Explorative ICRL" if experiment_args.context_strategy_name == "random_unbiased_only_positive" else "Approximate ICRL"] = experiment_total_tokens_processed
            continue

        # Extract full args
        experiment_data = experiment_storage.load_data(experiment_args)
        experiment_args = experiment_data["args"]
        
        # Nicely print directory that is being processed
        print(f"Processing experiment in {experiment_dir}")

        set_seed_everywhere(experiment_args["seed"])

        model = VLLMModelTokenizerOnlyWrapper(
            model_name=experiment_args["model_name"],
            icl=experiment_args["max_context_examples"] != 0,
            icrl=experiment_args["icrl"],
            temperature=experiment_args["temperature"],
            verbose=False
        )
        task = load_task(
            task_name=experiment_args["task_name"],
            verbose=False
        )
        model.set_task(task)

        # Define max context examples
        if experiment_args["max_context_examples"] is None:
            max_context_examples = read(os.path.join(get_data_dir(), f"{experiment_args['model_name']}", f"{experiment_args['task_name']}", MAXIMUM_CONTEXT_LENGTH_FILENAME))
        else:
            max_context_examples = experiment_args["max_context_examples"]

        context_generator = load_context_generator(
            context_generator_name=experiment_args["context_strategy_name"],
            max_examples=max_context_examples,
            p_keep=experiment_args["context_p_keep"],
            max_contexts=experiment_args["max_contexts"] if experiment_args["context_strategy_name"].startswith("approximate") else None,
            approximate_context_sampling_method=experiment_args["approximate_context_sampling_method"] if experiment_args["context_strategy_name"].startswith("approximate") else None,
            verbose=False
        )

        experiment_total_tokens_processed = 0

        # Collect training examples
        training_task_prompt_list = []
        training_model_prediction_list = []
        training_task_feedback_list = []
        training_task_answer_list = []
        training_accuracies = []

        # Iterate over steps:
        # 1. Get the context with the training data saved in the step data
        # 2. Call model.predict_labels() on the training data, which returns the number of new tokens used
        # 3. Output the number of new tokens used
        for step in tqdm(experiment_data["steps"], desc="Processing steps", leave=False):
            step_data = experiment_data["steps"][step]

            # Recompute the context using the training data saved in the step data and the same random seed
            context_generator.set_random_seed(experiment_args["seed"] + step)
            icl_task_prompt_list, icl_model_prediction_list, icl_task_feedback_list, icl_task_answer_list, icl_task_accuracies = context_generator.generate(
                training_task_prompt_list,
                training_model_prediction_list,
                training_task_feedback_list,
                training_task_answer_list,
                training_accuracies 
            )

            model.refresh_cache(icl_task_prompt_list, icl_model_prediction_list if experiment_args["icrl"] else [], icl_task_feedback_list if (experiment_args["icrl"] and not experiment_args["icrl_omit_feedback"]) else [], icl_task_answer_list if not experiment_args["icrl"] else [])
            training_data = step_data.get_training_data()
            training_task_prompt, training_model_prediction, training_task_feedback, training_task_answer, training_task_accuracy = training_data

            # Get the number of new tokens used
            new_tokens_used = model.predict_labels([training_task_prompt], generation_seed=None, force_verbose=False)[0]

            experiment_total_tokens_processed += new_tokens_used

            # Update training data
            training_task_prompt_list.append(training_task_prompt)
            training_model_prediction_list.append(training_model_prediction)
            training_task_feedback_list.append(training_task_feedback)
            training_task_answer_list.append(training_task_answer)
            training_accuracies.append(training_task_accuracy)

        # Save total tokens processed in a new json in the experiment dir
        save(experiment_total_tokens_processed, total_tokens_path)
        results[experiment_dir] = experiment_total_tokens_processed
        grouped_results[experiment_args["model_name"]][experiment_args["task_name"]][experiment_dir] = experiment_total_tokens_processed

    # Display results grouped by model and task
    print("\nResults grouped by model and task:")
    for model, tasks in grouped_results.items():
        print(f"\nModel: {model}")
        for task, experiments in tasks.items():
            print(f"  Task: {task}")
            for experiment_dir, total_tokens in experiments.items():
                print(f"    Experiment: {experiment_dir}")
                print(f"    Total Tokens Processed: {total_tokens:,}")
            # Compute ratio between highest and lowest
            highest_tokens = max(experiments.values())
            lowest_tokens = min(experiments.values())
            ratio = highest_tokens / lowest_tokens
            print(f"    Ratio between highest and lowest: {ratio:.2f}")


if __name__ == "__main__":
    main()