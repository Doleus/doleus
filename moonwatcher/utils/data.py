import operator as op
from enum import Enum

OPERATOR_DICT = {
    "<": op.lt,
    ">": op.gt,
    ">=": op.ge,
    "<=": op.le,
    "==": op.eq,
    "=": op.eq,
    "!=": op.ne,
}


class TaskType(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"


class Task(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class DataType(Enum):
    DATASET = "dataset"
    SLICE = "slice"
    MODEL = "model"
    CHECK = "check"
    CHECKSUITE = "checksuite"
    CHECK_REPORT = "check_report"
    CHECKSUITE_REPORT = "checksuite_report"
    PREDICTIONS = "predictions"
    GROUNDTRUTHS = "groundtruths"
