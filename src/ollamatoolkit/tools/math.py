# ./ollamatoolkit/tools/math.py
"""
Ollama Toolkit - Math Tools
===========================
Safe mathematical evaluation with broad support for statistics and standard math.
"""

import math
import logging
import statistics
import random
from typing import List, Union

logger = logging.getLogger(__name__)


class MathTools:
    """
    Advanced mathematical evaluation tool.
    Supports:
    - Standard Math (sin, cos, sqrt, pow, log...)
    - Statistics (mean, median, mode, stdev, variance...)
    - Randomness (randint, uniform, choice...)
    - List operations (sum, min, max, len)
    """

    @staticmethod
    def calculate(expression: str) -> str:
        """
        Evaluates a mathematical expression.
        Examples:
        - "sqrt(16) * 5"
        - "mean([1, 2, 3, 4, 100])"
        - "random.randint(1, 10)"
        """
        # 1. Build a rich safe context
        allowed_context = {
            # Builtins
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "len": len,
            "sorted": sorted,
            "int": int,
            "float": float,
            "list": list,
            "set": set,
            # Math Module
            "math": math,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "ceil": math.ceil,
            "floor": math.floor,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "degrees": math.degrees,
            "radians": math.radians,
            "pi": math.pi,
            "e": math.e,
            # Statistics Module
            "statistics": statistics,
            "mean": statistics.mean,
            "median": statistics.median,
            "mode": statistics.mode,
            "stdev": statistics.stdev,
            "variance": statistics.variance,
            # Random Module (Safe subset)
            "random": random,
            "randint": random.randint,
            "choice": random.choice,
            "uniform": random.uniform,
        }

        # 2. Security Check
        # Disallow dunders, imports, lambdas, and notoriously unsafe builtins
        expression = expression.strip()
        unsafe_keywords = [
            "__",
            "import",
            "eval",
            "exec",
            "compile",
            "open",
            "file",
            "sys",
            "os",
            "subprocess",
            "exit",
            "quit",
            "help",
            "input",
        ]

        if any(bad in expression for bad in unsafe_keywords):
            return "Error: Security violation. Unsafe expression detected."

        # 3. Evaluation
        try:
            # We strictly limit __builtins__ to None to prevent access to default python globals
            # We only provide the allowed_context
            result = eval(expression, {"__builtins__": None}, allowed_context)
            return str(result)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"

    @staticmethod
    def analyze_list(numbers: List[Union[int, float]]) -> str:
        """
        Convenience method to get a statistical summary of a list.
        """
        try:
            if not numbers:
                return "Empty list."
            return (
                f"Count: {len(numbers)}\n"
                f"Sum: {sum(numbers)}\n"
                f"Mean: {statistics.mean(numbers):.4f}\n"
                f"Median: {statistics.median(numbers):.4f}\n"
                f"Min: {min(numbers)}\n"
                f"Max: {max(numbers)}\n"
                f"Stdev: {statistics.stdev(numbers) if len(numbers) > 1 else 0:.4f}"
            )
        except Exception as e:
            return f"Error analyzing list: {e}"
