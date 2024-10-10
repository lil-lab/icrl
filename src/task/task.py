from abc import ABC, abstractmethod
from collections import Counter
from typing import List


from src.utils.logger import get_logger
from datasets import load_dataset, Dataset, DatasetDict

import random

class Task(ABC):
    """
    Abstract base class for defining a task. 
    Each task should inherit from this class and implement the abstract methods.
    """
    training_split = None
    test_split = None
    labels = None
    description = ""
    prediction_prefix = "Answer:"

    def __init__(self, verbose: bool):
        """
        Initialize the Task with verbosity option.

        Args:
            verbose (bool): Whether to enable verbose logging.
        """
        self.verbose = verbose

    def get_description(self) -> str:
        """
        Get the description of the task.

        Returns:
            str: The description of the task.
        """
        return self.description

    def get_labels(self) -> List[str]:
        """
        Get the labels for the task.

        Returns:
            List[str]: The labels for the task.
        """
        return self.labels

    @abstractmethod
    def get_prompt(self, entry) -> str:
        """
        Get the prompt for a given entry.

        Args:
            entry: The entry for which to generate the prompt.

        Returns:
            str: The generated prompt.
        """
        pass

    def get_prediction_prefix(self) -> str:
        """
        Get the prediction prefix for the task.

        Returns:
            str: The prediction prefix.
        """
        return self.prediction_prefix

    def get_feedback(self, entry, predicted_label: str):
        """
        Get feedback for a given entry and predicted label.

        Args:
            entry: The entry for which to generate feedback.
            predicted_label (str): The predicted label.

        Returns:
            int: Feedback score (1 if correct, 0 if incorrect).
        """
        answer = self.get_answer(entry)

        assert predicted_label in self.get_labels(), f"Predicted label {predicted_label} not in {self.get_labels()}"

        feedback = 1 if predicted_label == answer else 0

        if self.verbose:
            get_logger().info(f"Predicted: '{predicted_label}'. True answer: '{answer}'. Feedback: {feedback}")

        return feedback

    @abstractmethod
    def get_answer(self, entry) -> str:
        """
        Get the answer for a given entry.

        Args:
            entry: The entry for which to get the answer.

        Returns:
            str: The answer.
        """
        pass

    def get_training_data(self, size: int, seed: int):
        """
        Get the training data for the task.

        Args:
            size (int): The size of the training data.
            seed (int): The seed for random shuffling.

        Returns:
            Dataset: The training data.
        """
        return self._process_data(self.data, self.training_split, size, seed)
    
    def get_test_data(self, size: int, seed: int):
        """
        Get the test data for the task.

        Args:
            size (int): The size of the test data.
            seed (int): The seed for random shuffling.

        Returns:
            Dataset: The test data.
        """
        return self._process_data(self.data, self.test_split, size, seed)

    def _process_data(self, data: Dataset, split: str, size: int, seed: int) -> Dataset:
        """
        Process the data by shuffling and selecting a subset.

        Args:
            data (Dataset): The dataset to process.
            split (str): The split of the dataset to use.
            size (int): The size of the subset to select.
            seed (int): The seed for random shuffling.

        Returns:
            Dataset: The processed dataset.
        """
        data = data[split]
        random.seed(seed)
        
        # Randomly shuffle the entire dataset
        shuffled_data = list(data)
        random.shuffle(shuffled_data)
        
        if size == -1:
            result = shuffled_data
        else:
            result = shuffled_data[:size]
            assert len(result) == size
        
        # Count the occurrences of each label in the result
        label_counts = Counter(self.get_answer(example) for example in result)

        label_counts = dict(label_counts)
        label_counts = dict(sorted(label_counts.items(), key=lambda item: item[0], reverse=True))
        
        # Log the output information
        get_logger().info(f"Unique labels: {len(label_counts)}")
        get_logger().info(f"Dataset size: {len(result)}")
        
        return Dataset.from_list(result)

def load_task(task_name: str, verbose: bool) -> Task:
    """
    Load a task by name.

    Args:
        task_name (str): The name of the task to load.
        verbose (bool): Whether to enable verbose logging.

    Returns:
        Task: The loaded task.

    Raises:
        ValueError: If the task name is not supported.
    """
    if task_name == "trec_coarse":
        return TRECCoarseTask(verbose=verbose)
    elif task_name == "trec_fine":
        return TRECFineTask(verbose=verbose)
    elif task_name == "banking77":
        return BANKING77Task(verbose=verbose)
    elif task_name == "clinic150":
        return CLINIC150_Task(verbose=verbose)
    elif task_name == "nlu":
        return NLU(verbose=verbose)
    else:
        raise ValueError(f"Task {task_name} not supported")


class TRECTask(Task, ABC):
    """
    Base class for TREC tasks.
    """
    training_split = "train"
    test_split = "test"
    coarse_id_to_label = {
        0: "abbreviation", 
        1: "entity", 
        2: "description", 
        3: "human", 
        4: "location", 
        5: 'numeric'
    }
    fine_id_to_label = {
        0: 'abbreviation abbreviation',
        1: 'abbreviation expansion',
        2: 'entity animal',
        3: 'entity body',
        4: 'entity color',
        5: 'entity creation',
        6: 'entity currency',
        7: 'entity disease',
        8: 'entity event',
        9: 'entity food',
        10: 'entity instrument',
        11: 'entity language',
        12: 'entity letter',
        13: 'entity other',
        14: 'entity plant',
        15: 'entity product',
        16: 'entity religion',
        17: 'entity sport',
        18: 'entity substance',
        19: 'entity symbol',
        20: 'entity technique',
        21: 'entity term',
        22: 'entity vehicle',
        23: 'entity word',
        24: 'description definition',
        25: 'description description',
        26: 'description manner',
        27: 'description reason',
        28: 'human group',
        29: 'human individual',
        30: 'human title',
        31: 'human description',
        32: 'location city',
        33: 'location country',
        34: 'location mountain',
        35: 'location other',
        36: 'location state',
        37: 'numeric code',
        38: 'numeric count',
        39: 'numeric date',
        40: 'numeric distance',
        41: 'numeric money',
        42: 'numeric order',
        43: 'numeric other',
        44: 'numeric period',
        45: 'numeric percent',
        46: 'numeric speed',
        47: 'numeric temperature',
        48: 'numeric size',
        49: 'numeric weight'
    }
    prediction_prefix = "Type:"

    def __init__(self, verbose: bool):
        """
        Initialize the TRECTask with verbosity option.

        Args:
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(verbose)
        self.data = load_dataset("CogComp/trec", trust_remote_code=True)
        
    def get_prompt(self, entry) -> str:
        """
        Get the prompt for a given entry.

        Args:
            entry: The entry for which to generate the prompt.

        Returns:
            str: The generated prompt.
        """
        prompt = f"Question: {entry['text']}"
        return prompt
    
    @abstractmethod
    def get_answer(self, entry) -> str:
        """
        Get the answer for a given entry.

        Args:
            entry: The entry for which to get the answer.

        Returns:
            str: The answer.
        """
        pass

class TRECCoarseTask(TRECTask):
    """
    TREC coarse task.
    """
    labels = list(TRECTask.coarse_id_to_label.values())

    def get_answer(self, entry) -> str:
        """
        Get the answer for a given entry.

        Args:
            entry: The entry for which to get the answer.

        Returns:
            str: The answer.
        """
        return self.coarse_id_to_label[entry["coarse_label"]]

class TRECFineTask(TRECTask):
    """
    TREC fine task.
    """
    labels = list(TRECTask.fine_id_to_label.values())

    def get_answer(self, entry) -> str:
        """
        Get the answer for a given entry.

        Args:
            entry: The entry for which to get the answer.

        Returns:
            str: The answer.
        """
        return self.fine_id_to_label[entry["fine_label"]]
    

class BANKING77Task(Task):
    """
    BANKING77 task.
    """
    training_split = "train"
    test_split = "test"
    id_to_label = {
        0: 'activate my card',
        1: 'age limit',
        2: 'apple pay or google pay',
        3: 'atm support',
        4: 'automatic top up',
        5: 'balance not updated after bank transfer',
        6: 'balance not updated after cheque or cash deposit',
        7: 'beneficiary not allowed',
        8: 'cancel transfer',
        9: 'card about to expire',
        10: 'card acceptance',
        11: 'card arrival',
        12: 'card delivery estimate',
        13: 'card linking',
        14: 'card not working',
        15: 'card payment fee charged',
        16: 'card payment not recognised',
        17: 'card payment wrong exchange rate',
        18: 'card swallowed',
        19: 'cash withdrawal charge',
        20: 'cash withdrawal not recognised',
        21: 'change pin',
        22: 'compromised card',
        23: 'contactless not working',
        24: 'country support',
        25: 'declined card payment',
        26: 'declined cash withdrawal',
        27: 'declined transfer',
        28: 'direct debit payment not recognised',
        29: 'disposable card limits',
        30: 'edit personal details',
        31: 'exchange charge',
        32: 'exchange rate',
        33: 'exchange via app',
        34: 'extra charge on statement',
        35: 'failed transfer',
        36: 'fiat currency support',
        37: 'get disposable virtual card',
        38: 'get physical card',
        39: 'getting spare card',
        40: 'getting virtual card',
        41: 'lost or stolen card',
        42: 'lost or stolen phone',
        43: 'order physical card',
        44: 'passcode forgotten',
        45: 'pending card payment',
        46: 'pending cash withdrawal',
        47: 'pending top up',
        48: 'pending transfer',
        49: 'pin blocked',
        50: 'receiving money',
        51: 'refund not showing up',
        52: 'request refund',
        53: 'reverted card payment',
        54: 'supported cards and currencies',
        55: 'terminate account',
        56: 'top up by bank transfer charge',
        57: 'top up by card charge',
        58: 'top up by cash or cheque',
        59: 'top up failed',
        60: 'top up limits',
        61: 'top up reverted',
        62: 'topping up by card',
        63: 'transaction charged twice',
        64: 'transfer fee charged',
        65: 'transfer into account',
        66: 'transfer not received by recipient',
        67: 'transfer timing',
        68: 'unable to verify identity',
        69: 'verify my identity',
        70: 'verify source of funds',
        71: 'verify top up',
        72: 'virtual card not working',
        73: 'visa or mastercard',
        74: 'why verify identity',
        75: 'wrong amount of cash received',
        76: 'wrong exchange rate for cash withdrawal'
    }
    labels = list(id_to_label.values())
    prediction_prefix = "Intent:"
   
    def __init__(self, verbose: bool):
        """
        Initialize the BANKING77Task with verbosity option.

        Args:
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(verbose)
        self.data = load_dataset("PolyAI/banking77", trust_remote_code=True)

    def get_prompt(self, entry) -> str:
        """
        Get the prompt for a given entry.

        Args:
            entry: The entry for which to generate the prompt.

        Returns:
            str: The generated prompt.
        """
        prompt = f"Query: {entry['text']}"
        return prompt
   
    def get_answer(self, entry) -> str:
        """
        Get the answer for a given entry.

        Args:
            entry: The entry for which to get the answer.

        Returns:
            str: The answer.
        """
        return self.id_to_label[entry["label"]]
    
class CLINIC150_OOS_Task(Task):
    """
    CLINIC150_OOS task.
    """
    training_split = "train"
    test_split = "test"
    subset = "plus"
    prediction_prefix = "Intent:"
    id_to_label = {
        0: 'restaurant reviews',
        1: 'nutrition info',
        2: 'account blocked',
        3: 'oil change how',
        4: 'time',
        5: 'weather',
        6: 'redeem rewards',
        7: 'interest rate',
        8: 'gas type',
        9: 'accept reservations',
        10: 'smart home',
        11: 'user name',
        12: 'report lost card',
        13: 'repeat',
        14: 'whisper mode',
        15: 'what are your hobbies',
        16: 'order',
        17: 'jump start',
        18: 'schedule meeting',
        19: 'meeting schedule',
        20: 'freeze account',
        21: 'what song',
        22: 'meaning of life',
        23: 'restaurant reservation',
        24: 'traffic',
        25: 'make call',
        26: 'text',
        27: 'bill balance',
        28: 'improve credit score',
        29: 'change language',
        30: 'no',
        31: 'measurement conversion',
        32: 'timer',
        33: 'flip coin',
        34: 'do you have pets',
        35: 'balance',
        36: 'tell joke',
        37: 'last maintenance',
        38: 'exchange rate',
        39: 'uber',
        40: 'car rental',
        41: 'credit limit',
        42: 'oos',
        43: 'shopping list',
        44: 'expiration date',
        45: 'routing',
        46: 'meal suggestion',
        47: 'tire change',
        48: 'todo list',
        49: 'card declined',
        50: 'rewards balance',
        51: 'change accent',
        52: 'vaccines',
        53: 'reminder update',
        54: 'food last',
        55: 'change ai name',
        56: 'bill due',
        57: 'who do you work for',
        58: 'share location',
        59: 'international visa',
        60: 'calendar',
        61: 'translate',
        62: 'carry on',
        63: 'book flight',
        64: 'insurance change',
        65: 'todo list update',
        66: 'timezone',
        67: 'cancel reservation',
        68: 'transactions',
        69: 'credit score',
        70: 'report fraud',
        71: 'spending history',
        72: 'directions',
        73: 'spelling',
        74: 'insurance',
        75: 'what is your name',
        76: 'reminder',
        77: 'where are you from',
        78: 'distance',
        79: 'payday',
        80: 'flight status',
        81: 'find phone',
        82: 'greeting',
        83: 'alarm',
        84: 'order status',
        85: 'confirm reservation',
        86: 'cook time',
        87: 'damaged card',
        88: 'reset settings',
        89: 'pin change',
        90: 'replacement card duration',
        91: 'new card',
        92: 'roll dice',
        93: 'income',
        94: 'taxes',
        95: 'date',
        96: 'who made you',
        97: 'pto request',
        98: 'tire pressure',
        99: 'how old are you',
        100: 'rollover 401k',
        101: 'pto request status',
        102: 'how busy',
        103: 'application status',
        104: 'recipe',
        105: 'calendar update',
        106: 'play music',
        107: 'yes',
        108: 'direct deposit',
        109: 'credit limit change',
        110: 'gas',
        111: 'pay bill',
        112: 'ingredients list',
        113: 'lost luggage',
        114: 'goodbye',
        115: 'what can i ask you',
        116: 'book hotel',
        117: 'are you a bot',
        118: 'next song',
        119: 'change speed',
        120: 'plug type',
        121: 'maybe',
        122: 'w2',
        123: 'oil change when',
        124: 'thank you',
        125: 'shopping list update',
        126: 'pto balance',
        127: 'order checks',
        128: 'travel alert',
        129: 'fun fact',
        130: 'sync device',
        131: 'schedule maintenance',
        132: 'apr',
        133: 'transfer',
        134: 'ingredient substitution',
        135: 'calories',
        136: 'current location',
        137: 'international fees',
        138: 'calculator',
        139: 'definition',
        140: 'next holiday',
        141: 'update playlist',
        142: 'mpg',
        143: 'min payment',
        144: 'change user name',
        145: 'restaurant suggestion',
        146: 'travel notification',
        147: 'cancel',
        148: 'pto used',
        149: 'travel suggestion',
        150: 'change volume'
    }
    labels = list(id_to_label.values())
   
    def __init__(self, verbose: bool):
        """
        Initialize the CLINIC150_OOS_Task with verbosity option.

        Args:
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(verbose)
        self.data = load_dataset("clinc/clinc_oos", self.subset, trust_remote_code=True)

    def get_prompt(self, entry) -> str:
        """
        Get the prompt for a given entry.

        Args:
            entry: The entry for which to generate the prompt.

        Returns:
            str: The generated prompt.
        """
        prompt = f"Query: {entry['text']}"
        return prompt
   
    def get_answer(self, entry) -> str:
        """
        Get the answer for a given entry.

        Args:
            entry: The entry for which to get the answer.

        Returns:
            str: The answer.
        """
        return self.id_to_label[entry["intent"]]


class CLINIC150_Task(CLINIC150_OOS_Task):
    """
    CLINIC150 task.
    """
    def __init__(self, verbose: bool):
        """
        Initialize the CLINIC150_Task with verbosity option.

        Args:
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(verbose)
        
        self.data = self.data.filter(lambda example: self.get_answer(example) != "oos")
        self.labels = [label for label in self.labels if label != "oos"]
class NLU(Task):
    """
    NLU task.
    """
    training_split = "train"
    test_split = "test"
    prediction_prefix = "Intent:"
    id_to_label = {
        0: 'alarm query',
        1: 'alarm remove',
        2: 'alarm set',
        3: 'audio volume down',
        4: 'audio volume mute',
        5: 'audio volume other',
        6: 'audio volume up',
        7: 'calendar query',
        8: 'calendar remove',
        9: 'calendar set',
        10: 'cooking query',
        11: 'cooking recipe',
        12: 'datetime convert',
        13: 'datetime query',
        14: 'email addcontact',
        15: 'email query',
        16: 'email querycontact',
        17: 'email sendemail',
        18: 'general affirm',
        19: 'general commandstop',
        20: 'general confirm',
        21: 'general dontcare',
        22: 'general explain',
        23: 'general greet',
        24: 'general joke',
        25: 'general negate',
        26: 'general praise',
        27: 'general quirky',
        28: 'general repeat',
        29: 'iot cleaning',
        30: 'iot coffee',
        31: 'iot hue lightchange',
        32: 'iot hue lightdim',
        33: 'iot hue lightoff',
        34: 'iot hue lighton',
        35: 'iot hue lightup',
        36: 'iot wemo off',
        37: 'iot wemo on',
        38: 'lists createoradd',
        39: 'lists query',
        40: 'lists remove',
        41: 'music dislikeness',
        42: 'music likeness',
        43: 'music query',
        44: 'music settings',
        45: 'news query',
        46: 'play audiobook',
        47: 'play game',
        48: 'play music',
        49: 'play podcasts',
        50: 'play radio',
        51: 'qa currency',
        52: 'qa definition',
        53: 'qa factoid',
        54: 'qa maths',
        55: 'qa stock',
        56: 'recommendation events',
        57: 'recommendation locations',
        58: 'recommendation movies',
        59: 'social post',
        60: 'social query',
        61: 'takeaway order',
        62: 'takeaway query',
        63: 'transport query',
        64: 'transport taxi',
        65: 'transport ticket',
        66: 'transport traffic',
        67: 'weather query'
    }   
    labels = list(id_to_label.values())

    def __init__(self, verbose: bool):
        """
        Initialize the NLU task with verbosity option.

        Args:
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(verbose)
        self.data = load_dataset("xingkunliuxtracta/nlu_evaluation_data", trust_remote_code=True)

        # Split the training data
        train_test_split = self.data['train'].train_test_split(test_size=0.2, seed=0)

        # Create a new DatasetDict with both splits
        self.data = DatasetDict({
            self.training_split: train_test_split['train'],
            self.test_split: train_test_split['test']
        })

        get_logger().info(f"{self.data}")

    def get_prompt(self, entry) -> str:
        """
        Get the prompt for a given entry.

        Args:
            entry: The entry for which to generate the prompt.

        Returns:
            str: The generated prompt.
        """
        prompt = f"Utterance: {entry['text']}"
        return prompt
    
    def get_answer(self, entry) -> str:
        """
        Get the answer for a given entry.

        Args:
            entry: The entry for which to get the answer.

        Returns:
            str: The answer.
        """
        return self.id_to_label[entry["label"]]