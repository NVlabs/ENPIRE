from abc import ABC, abstractmethod
from typing import Any

import numpy as np

Observation = dict[str, Any]
Action = dict[str, np.ndarray]
Options = dict[str, Any]

# Policy Info dictionary
# Common (sometimes mandatory) keys:
#  - action_chunk: dict[str, np.ndarray]
Info = dict[str, Any]


class Policy(ABC):

    def reset(self) -> Info | None:
        return None

    @abstractmethod
    def get_action(self, observation: Observation) -> tuple[Action, Info]:
        pass

