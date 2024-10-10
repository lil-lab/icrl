from src.model.model import ModelWrapper
from src.task.task import Task
from src.utils.logger import get_logger

def find_longest_label(model: ModelWrapper, prompt, all_labels):
    """
    Find the longest label based on the number of tokens.

    Args:
        model (ModelWrapper): The model wrapper.
        prompt (str): The prompt to use.
        all_labels (list): List of all possible labels.

    Returns:
        str: The longest label.
    """
    # Format messages for each label
    possible_messages = [model._format_messages([prompt, "mock input"], [label], [], []) for label in all_labels]
    # Calculate the number of tokens for each message
    lengths = [model.get_number_tokens(possible_message) for possible_message in possible_messages]
    # Find the label with the maximum length
    longest_label = all_labels[lengths.index(max(lengths))]

    # Show the longest label
    get_logger().info(f"Longest label: `{longest_label}`. All labels: {all_labels}")

    return longest_label

def find_longest_feedback(model: ModelWrapper, prompt, label):
    """
    Find the longest feedback based on the number of tokens.

    Args:
        model (ModelWrapper): The model wrapper.
        prompt (str): The prompt to use.
        label (str): The label to use.

    Returns:
        int: The longest feedback.
    """
    # Possible feedbacks
    possible_feedbacks = [0, 1]
    # Format messages for each feedback
    possible_messages = [model._format_messages([prompt, "mock input"], [label], [feedback], []) for feedback in possible_feedbacks]
    # Calculate the number of tokens for each message
    lengths = [model.get_number_tokens(possible_message) for possible_message in possible_messages]
    # Find the feedback with the maximum length
    longest_feedback = possible_feedbacks[lengths.index(max(lengths))]

    # Show the longest feedback
    get_logger().info(f"Longest feedback: `{longest_feedback}`. All feedbacks: {possible_feedbacks}")

    return longest_feedback

def sort_prompts(model: ModelWrapper, all_prompts, label, feedback):
    """
    Sort all prompts by length in descending order.

    Args:
        model (ModelWrapper): The model wrapper.
        all_prompts (list): List of all prompts.
        label (str): The label to use.
        feedback (int): The feedback to use.

    Returns:
        list: Sorted prompts by length in descending order.
    """
    # Log the sorting process
    get_logger().info("Sorting prompts by length")
    # Format messages for each prompt
    possible_messages = [model._format_messages([prompt, "mock input"], [label], [feedback], []) for prompt in all_prompts]
    get_logger().info("Possible messages formatted")
    # Calculate the number of tokens for each message
    lengths = [model.get_number_tokens(possible_message) for possible_message in possible_messages]
    get_logger().info("Lengths calculated")
    # Sort prompts by length in descending order
    sorted_prompts = [all_prompts[index] for index in sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)]

    return sorted_prompts

def find_maximum_number_examples(model: ModelWrapper, task: Task, verbose: bool, maximum_tokens: int = None):
    """
    Find the maximum number of examples that can fit within the maximum token limit.

    Args:
        model (ModelWrapper): The model wrapper.
        task (Task): The task to use.
        verbose (bool): Whether to print verbose logs.
        maximum_tokens (int, optional): The maximum number of tokens. Defaults to None.

    Returns:
        int: The maximum number of examples.
    """
    get_logger().info(f"Finding maximum number of examples for task {task.__class__.__name__} and model {model.get_name()}")
    if maximum_tokens is None:
        maximum_tokens = model.get_maximum_length()
    
    # Get all prompts
    training_data = task.get_training_data(size=-1, seed=0)
    test_data = task.get_test_data(size=-1, seed=0)
    all_prompts = [task.get_prompt(example) for example in training_data] + [task.get_prompt(example) for example in test_data]
    
    # Get all labels
    all_labels = task.get_labels()
    
    # Find the longest label
    longest_label = find_longest_label(model, all_prompts[0], all_labels)
    
    # Find the longest feedback
    longest_feedback = find_longest_feedback(model, all_prompts[0], longest_label)
    
    # Sort all prompts by length in reverse order
    sorted_prompts = sort_prompts(model, all_prompts, longest_label, longest_feedback)
    
    def check_examples(num_examples):
        """
        Check if the given number of examples fit within the maximum token limit.

        Args:
            num_examples (int): The number of examples to check.

        Returns:
            bool: True if the examples fit within the limit, False otherwise.
        """
        possible_message = model._format_messages(sorted_prompts[:num_examples], 
                                                  [longest_label] * num_examples, 
                                                  [longest_feedback] * num_examples, 
                                                  [])
        length = model.get_number_tokens(possible_message)
        return length <= maximum_tokens * 0.95

    # Binary search to find the maximum number of examples
    left, right = 0, len(sorted_prompts)
    while left < right:
        mid = (left + right + 1) // 2
        if check_examples(mid):
            left = mid
        else:
            right = mid - 1
        get_logger().info(f"Binary search: left={left}, right={right}")
    max_examples = left + 1

    get_logger().info(f"Maximum number of examples for task {task.__class__.__name__} and model {model.get_name()} is {max_examples}")
    return max_examples

