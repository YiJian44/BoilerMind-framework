from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):

    name = "base_skill"
    description = ""


    @abstractmethod
    def execute(self, context: Any):
        raise NotImplementedError