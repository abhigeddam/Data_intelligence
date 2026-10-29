"""
Base Agent Interface for OmniInsight Multi-Agent System.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)


class BaseAgent(ABC):
    """
    Abstract base class providing standard lifecycle, telemetry,
    and execution protocol for intelligence agents.
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.logger = logging.getLogger(f"agent.{name.lower().replace(' ', '_')}")
        self.last_execution_duration: float = 0.0

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute core cognitive task on given workflow context.
        """
        pass
