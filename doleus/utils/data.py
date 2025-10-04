# SPDX-FileCopyrightText: 2025 Doleus contributors
# SPDX-License-Identifier: Apache-2.0

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
    "in": lambda x, y: x in y,
    "not_in": lambda x, y: x not in y,
    "between": lambda x, y: (
        y[0] <= x <= y[1] if isinstance(y, (list, tuple)) and len(y) == 2 else False
    ),
    "not_between": lambda x, y: (
        not (y[0] <= x <= y[1]) if isinstance(y, (list, tuple)) and len(y) == 2 else False
    ),
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
    ANNOTATIONS = "annotations"
