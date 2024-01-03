from typing import Callable, Dict

from . import apps, humaneval, humanevalpack

ALL_TASKS: Dict[str, Callable] = dict()
ALL_TASKS.update(apps.TASK_CREATORS)
ALL_TASKS.update(humaneval.TASK_CREATORS)
ALL_TASKS.update(humanevalpack.TASK_CREATORS)
