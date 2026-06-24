import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import load_dataset
from src.config.constants import (
    HF_DATASET_NAME,
    HF_DATASET_QUESTION_FIELD,
    HF_DATASET_ANSWER_FIELD,
)

logger = logging.getLogger(__name__)


class Loader:
    """
    Interface for retrieving structured records from a HuggingFace dataset by loading the 
    target split and extracting the required textual fields.

    Attributes:
        dataset_name (str): Name of the HuggingFace dataset to be loaded.
        question_field (str): Name of the field containing question text.
        answer_field (str): Name of the field containing answer text.

    Methods:
        load() -> List[Dict[str, Any]]:
            Loads the dataset split, converts it into a pandas DataFrame, extracts question 
            and answer fields, and returns a list of structured records.
    """

    def __init__(self) -> None:
        self.dataset_name: Optional[str] = HF_DATASET_NAME
        self.question_field: Optional[str] = HF_DATASET_QUESTION_FIELD
        self.answer_field: Optional[str] = HF_DATASET_ANSWER_FIELD

        logger.info(
            "Loader initialization completed. "
            f"Dataset name: {self.dataset_name}. "
            f"Question field: {self.question_field}. "
            f"Answer field: {self.answer_field}."
        )

    def load(self) -> List[Dict[str, Any]]:
        if self.dataset_name is None:
            raise ValueError("HF_DATASET_NAME is not defined in the environment.")
        if self.question_field is None:
            raise ValueError("HF_DATASET_QUESTION_FIELD is not defined in the environment.")
        if self.answer_field is None:
            raise ValueError("HF_DATASET_ANSWER_FIELD is not defined in the environment.")

        logger.info(f"Loading HuggingFace dataset '{self.dataset_name}'.")

        dataset = load_dataset(self.dataset_name)["train"]

        logger.info(f"Dataset loaded with {len(dataset)} entries.")

        df: pd.DataFrame = dataset.to_pandas()

        logger.info(f"Converted dataset to pandas DataFrame with {len(df)} rows.")

        if self.question_field not in df.columns:
            raise ValueError(f"Missing required field: '{self.question_field}'")
        if self.answer_field not in df.columns:
            raise ValueError(f"Missing required field: '{self.answer_field}'")

        logger.info(
            f"Extracting fields: question='{self.question_field}', answer='{self.answer_field}'."
        )

        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            question = row.get(self.question_field)
            answer = row.get(self.answer_field)

            if isinstance(question, str) and isinstance(answer, str):
                records.append(
                    {
                        "question": question.strip(),
                        "answer": answer.strip(),
                    }
                )

        logger.info(f"Extraction completed. Total valid records: {len(records)}.")

        return records