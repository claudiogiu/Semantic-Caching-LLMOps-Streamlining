from enum import Enum


class PipelineStatus(str, Enum):
    """
    Enumeration defining the possible outcomes of a pipeline execution.
    """

    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class WarmupStatus(str, Enum):
    """
    Enumeration defining the possible outcomes of the warmup procedure.
    """

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class CacheOperation(str, Enum):
    """
    Enumeration defining the supported cache maintenance operations.
    """

    FLUSH_ALL = "FLUSH_ALL"
    DELETE_RECORD = "DELETE_RECORD"
