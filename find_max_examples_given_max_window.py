import argparse

from src.icl.utils import find_maximum_number_examples
from src.model.model import load_model
from src.task.task import load_task
from src.utils.logger import get_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--max_window_tokens", type=int, required=True)
    return parser.parse_args()

def find_max_examples_given_max_window(args):
    # Load the model with specified parameters
    model = load_model(
        model_name=args.model_name,
        icl=True,
        icrl=True,
        temperature=1.0,
        verbose=False
    ) 
    # Load the task with specified name
    task = load_task(args.task_name, verbose=False)
    # Set the task for the model
    model.set_task(task)
    
    # Find the maximum number of examples given the maximum window tokens
    maximum_examples = find_maximum_number_examples(model, task, verbose=True, maximum_tokens=args.max_window_tokens)

    # Log the result
    get_logger().info(f"Maximum number of examples for model {args.model_name} and task {args.task_name} given a maximum window of {args.max_window_tokens} tokens: {maximum_examples}")


if __name__ == "__main__":
    args = get_args()
    find_max_examples_given_max_window(args)