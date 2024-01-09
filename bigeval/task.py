from abc import ABC, abstractmethod
from typing import Dict, List
from warnings import warn

from datasets import load_dataset


class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    dataset_name: str = None

    # Default stop words for the task
    stop_words: List[str] = None

    def __init__(self, *args, **kwargs):
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.dataset_name)
        except Exception as e:
            warn(
                f"Loading the dataset failed with {str(e)}. This task will use a locally downloaded dataset, not from the HF hub."
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc: Dict[str, str]) -> str:
        """Builds the prompt for the LM to generate from.
        :param doc: sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc: Dict[str, str]) -> str:
        """Builds the reference solution for the doc.
        :param doc: sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation: str, idx: int) -> str:
        """Defines the postprocessing for a LM generation.
        :param generation: code generation from LM
        :param idx: index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(
        self, generations: List[List[str]], references: List[str]
    ) -> Dict[str, float]:
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list of lists containing generations
        :param references: list of str containing refrences
        """
        pass
