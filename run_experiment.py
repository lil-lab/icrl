import argparse
import os

import wandb

import numpy as np


from src.experiments.utils import ExperimentContextManager
from src.icl.context_generators import ContextGenerator, load_context_generator
from src.icl.utils import find_maximum_number_examples
from src.task.task import load_task
from src.model.model import load_model
from src.utils.random import set_seed_everywhere
from src.utils.logger import get_logger


from time import time

def init_wandb(run_name: str = None):
    wandb.init(project="iclf", name=run_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str) # "microsoft/Phi-3.5-mini-instruct" "meta-llama/Meta-Llama-3.1-8B-Instruct"
    parser.add_argument("--task_name",type=str)
    parser.add_argument("--context_strategy_name", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--context_p_keep", type=float)
    parser.add_argument("--max_context_examples", type=int, default=None)
    parser.add_argument("--icrl", action=argparse.BooleanOptionalAction)
    parser.add_argument("--icrl_omit_feedback", action=argparse.BooleanOptionalAction)
    parser.add_argument("--icrl_flip_feedback", action=argparse.BooleanOptionalAction)
    parser.add_argument("--icrl_flip_feedback_prob", type=float, default=None)
    parser.add_argument("--max_contexts", type=int, default=-1)
    parser.add_argument("--approximate_context_sampling_method", type=str, default=None)
    parser.add_argument("--train_k", type=int)
    parser.add_argument("--test_every", type=int)
    parser.add_argument("--test_k", type=int)
    parser.add_argument("--debug_k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--training_seed", type=int)
    parser.add_argument("--test_seed", type=int)
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)

    return parser.parse_args()



def generate_random_float(seed):
    rng = np.random.default_rng(seed)
    return rng.random()



def main():
    args = get_args()


    # Preliminary setup
    init_wandb()
    os.environ["HF_TOKEN"] = args.hf_token
    set_seed_everywhere(args.seed)

    # Print args
    if args.verbose:
        get_logger().info(f"Arguments: {args}")

    # Load task
    task = load_task(
        task_name=args.task_name,
        verbose=False
    )

    # Load model
    model = load_model(
        model_name=args.model_name,
        icl=args.max_context_examples != 0,
        icrl=args.icrl,
        temperature=args.temperature,
        verbose=args.verbose
    )
    model.set_task(task)

    # Load data
    training_size = args.train_k if args.max_context_examples != 0 else 0 # If we are doing zero shot, we don't need any training data
    training_data = task.get_training_data(size=training_size, seed=args.training_seed) 
    test_data = task.get_test_data(size=args.test_k, seed=args.test_seed)

    get_logger().info(f"Training data size: {len(training_data)}")
    get_logger().info(f"Test data size: {len(test_data)}")

    steps = training_size if args.max_context_examples != 0 else 1
    steps_to_test = [0] + [i for i in range(steps) if (i+1) % args.test_every == 0]


    debug_k = 0
    with ExperimentContextManager(args) as experiment_data:
        # Define max context examples
        if args.max_context_examples is None: # when using maximum possible based on model, task
            if experiment_data.data_manager.get_maximum_context_length() is None:
                max_context_examples = find_maximum_number_examples(model, task, args.verbose)
                experiment_data.data_manager.set_maximum_context_length(max_context_examples)
            else:
                max_context_examples = experiment_data.data_manager.get_maximum_context_length()
        else: # when specified 
            max_context_examples = args.max_context_examples

        # If standard ICL, we stop at the minimum test step bigger or equal to the number of maximum context examples
        if not args.icrl:
            last_step = min([step for step in steps_to_test if step >= max_context_examples])
            steps = last_step + 1

        # Load context generator
        context_generator: ContextGenerator = load_context_generator(
            context_generator_name=args.context_strategy_name,
            max_examples=max_context_examples,
            p_keep=args.context_p_keep,
            max_contexts=args.max_contexts,
            approximate_context_sampling_method=args.approximate_context_sampling_method,
            verbose=args.verbose
        )

        training_task_prompt_list, training_model_prediction_list, training_task_feedback_list, training_task_answer_list, training_accuracies = [], [], [], [], []
        
        get_logger().info("Starting experiment")

        tik = time()

        for step in range(steps):
            get_logger().info(f"Step {step}")
            step_data = experiment_data.get_step_data(step)

            if (max_context_examples > 0 and not step_data.context_processed()):
                # Set random seed for context generator
                context_generator.set_random_seed(args.seed+step)

                # Generate context
                icl_task_prompt_list, icl_model_prediction_list, icl_task_feedback_list, icl_task_answer_list, icl_task_accuracies = context_generator.generate(training_task_prompt_list, training_model_prediction_list, training_task_feedback_list, training_task_answer_list, training_accuracies)
                additional_metrics = context_generator.get_context_additional_metrics() # Get additional metrics from context generator

                # Set context for next training / test step
                step_data.set_context(icl_task_prompt_list, icl_model_prediction_list, icl_task_feedback_list, icl_task_answer_list, icl_task_accuracies, additional_metrics)
            
            if (max_context_examples > 0 and not step_data.training_step_processed()) or (not step_data.test_step_processed() and step in steps_to_test):
                # If we need to perform a training or test step and this is not zero shot, we need to refresh the model cache
                if max_context_examples > 0:
                    icl_task_prompt_list, icl_model_prediction_list, icl_task_feedback_list, icl_task_answer_list, icl_task_accuracies = step_data.get_context()
                    model.refresh_cache(icl_task_prompt_list, icl_model_prediction_list if args.icrl else [], icl_task_feedback_list if (args.icrl and not args.icrl_omit_feedback) else [], icl_task_answer_list if not args.icrl else [])

                if not step_data.training_step_processed():
                    train_example = training_data[step]

                    new_training_task_prompt = task.get_prompt(train_example)

                    if args.icrl:
                        new_training_model_prediction = model.predict_labels([new_training_task_prompt], generation_seed=args.seed+step, force_verbose=debug_k < args.debug_k)[0]
                        new_training_task_accuracy = task.get_feedback(train_example, new_training_model_prediction)
                        new_training_task_feedback = new_training_task_accuracy
                        if args.icrl_flip_feedback:
                            if generate_random_float(args.seed + step) < args.icrl_flip_feedback_prob:
                                new_training_task_feedback = 1 - new_training_task_feedback
                                if args.verbose:
                                    get_logger().info(f"Flipping feedback for example {step}")
                    else:
                        new_training_model_prediction = None
                        new_training_task_accuracy = 1.0 # We arbitrarily assign 1 for implementation reasons. This number is never used.
                        new_training_task_feedback = None

                    # Get correct answer for new data point
                    new_training_task_answer = task.get_answer(train_example)

                    # Update step training data 
                    step_data.set_training_data(new_training_task_prompt, new_training_model_prediction, new_training_task_feedback, new_training_task_answer, new_training_task_accuracy)
            
                if not step_data.test_step_processed() and step in steps_to_test:
                    test_task_prompt_list = [task.get_prompt(test_example) for test_example in test_data]
                    test_predictions = model.predict_labels(test_task_prompt_list, generation_seed=args.seed+step, force_verbose=debug_k < args.debug_k)
                    test_task_feedback_list = [task.get_feedback(test_example, test_prediction) for test_example, test_prediction in zip(test_data, test_predictions)]
                    test_task_answer_list = [task.get_answer(test_example) for test_example in test_data]
                    test_accuracies = [task.get_feedback(test_example, test_prediction) for test_example, test_prediction in zip(test_data, test_predictions)]

                    step_data.set_test_data(test_task_prompt_list, test_predictions, test_task_feedback_list, test_task_answer_list, test_accuracies)

                tok = time()
                step_data.set_time(tok-tik)           
            
                debug_k += 1

            new_training_task_prompt, new_training_model_prediction, new_training_task_feedback, new_training_task_answer, new_training_task_accuracy = step_data.get_training_data()
            
            # Update training data
            training_task_prompt_list.append(new_training_task_prompt)
            training_model_prediction_list.append(new_training_model_prediction)
            training_task_feedback_list.append(new_training_task_feedback)
            training_task_answer_list.append(new_training_task_answer)
            training_accuracies.append(new_training_task_accuracy)

            experiment_data.save_changes()


    get_logger().info("Ending experiment")

    
    
if __name__ == "__main__":
    main()