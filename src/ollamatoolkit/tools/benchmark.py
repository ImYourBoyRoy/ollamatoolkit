# ./ollamatoolkit/tools/benchmark.py
"""
Ollama Toolkit - Model Benchmarking (GPU-Aware)
===============================================
Benchmarking with proper GPU resource management.

CRITICAL: Runs ONE model at a time with proper unloading to get valid results.

Usage:
    benchmark = ModelBenchmark(base_url="http://localhost:11434")

    # Run isolated benchmark (unloads model after each test)
    results = await benchmark.benchmark_vision_isolated("./image.jpeg")

    # Check GPU status
    running = benchmark.get_running_models()

Requirements:
    - Only one model should be loaded during each benchmark
    - Previous model must be unloaded before loading next
    - GPU memory must be monitored
"""

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama

from ollamatoolkit.tools.models import ModelInspector

logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================


class BenchmarkError(Exception):
    """Base benchmark error."""

    pass


class GPUBusyError(BenchmarkError):
    """GPU is occupied by other models."""

    pass


class ModelUnloadError(BenchmarkError):
    """Failed to unload model."""

    pass


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model: str
    task_type: str
    success: bool
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    output_preview: str
    gpu_memory_gb: float = 0.0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    task_type: str
    timestamp: str
    input_sample: str
    models_tested: int
    results: List[BenchmarkResult]
    fastest_model: Optional[str] = None
    most_tokens: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "input_sample": self.input_sample[:100] + "..."
            if len(self.input_sample) > 100
            else self.input_sample,
            "models_tested": self.models_tested,
            "results": [r.to_dict() for r in self.results],
            "fastest_model": self.fastest_model,
            "most_tokens": self.most_tokens,
        }


# =============================================================================
# GPU-Aware Benchmark Class
# =============================================================================


class ModelBenchmark:
    """
    GPU-aware benchmarking for Ollama models.
    Ensures only ONE model is loaded at a time for valid results.
    """

    # Wait time after unloading a model (seconds)
    UNLOAD_WAIT_TIME = 3.0
    # Maximum wait for model to unload (seconds)
    MAX_UNLOAD_WAIT = 30.0
    # Check interval for model unload (seconds)
    UNLOAD_CHECK_INTERVAL = 2.0

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize ModelBenchmark.

        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        self.async_client = ollama.AsyncClient(host=base_url)
        self.inspector = ModelInspector(base_url=base_url)
        self._reports: List[BenchmarkReport] = []
        logger.info(f"ModelBenchmark initialized for {base_url}")

    # =========================================================================
    # GPU Management
    # =========================================================================

    def get_running_models(self) -> List[Dict[str, Any]]:
        """
        Get list of currently running/loaded models.

        Returns:
            List of dicts with model name, size, context, etc.
        """
        try:
            response = self.client.ps()
            models = []
            for m in response.models if hasattr(response, "models") else []:
                size_value = getattr(m, "size", 0) or 0
                models.append(
                    {
                        "name": m.name if hasattr(m, "name") else str(m),
                        "size_gb": float(size_value) / 1e9,
                        "context": getattr(m, "context", 0),
                    }
                )
            return models
        except Exception as e:
            logger.error(f"Failed to get running models: {e}")
            return []

    def get_total_gpu_usage_gb(self) -> float:
        """Get total GPU memory usage from running models."""
        models = self.get_running_models()
        return sum(m.get("size_gb", 0) for m in models)

    def is_gpu_clear(self) -> bool:
        """Check if no models are currently loaded."""
        return len(self.get_running_models()) == 0

    async def wait_for_gpu_clear(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all models to unload from GPU.

        Args:
            timeout: Max seconds to wait (default: MAX_UNLOAD_WAIT)

        Returns:
            True if GPU is clear, False if timeout
        """
        timeout = timeout or self.MAX_UNLOAD_WAIT
        start = time.time()

        while time.time() - start < timeout:
            if self.is_gpu_clear():
                logger.info("GPU clear, ready for next model")
                return True

            running = self.get_running_models()
            logger.info(f"Waiting for GPU clear... {len(running)} models still loaded")
            await asyncio.sleep(self.UNLOAD_CHECK_INTERVAL)

        logger.warning(f"Timeout waiting for GPU to clear after {timeout}s")
        return False

    async def ensure_single_model(self, model_name: str) -> bool:
        """
        Ensure only the specified model is loaded (or GPU is clear).

        Args:
            model_name: Model that should be running

        Returns:
            True if ready for benchmark
        """
        running = self.get_running_models()

        # Check if only our model is running
        if len(running) == 1 and model_name in running[0].get("name", ""):
            return True

        # If other models are running, we need to wait
        if running:
            total_gb = self.get_total_gpu_usage_gb()
            logger.warning(
                f"GPU busy: {len(running)} models using {total_gb:.1f}GB. "
                f"Waiting for unload..."
            )

            # Wait for models to timeout and unload
            if not await self.wait_for_gpu_clear():
                raise GPUBusyError(
                    f"GPU still occupied after waiting. "
                    f"Running models: {[m['name'] for m in running]}"
                )

        return True

    # =========================================================================
    # Isolated Benchmarks (Proper GPU Management)
    # =========================================================================

    async def benchmark_model_isolated(
        self, model: str, task_type: str, test_func, wait_after: bool = True
    ) -> BenchmarkResult:
        """
        Run a single model benchmark in isolation.

        1. Check/wait for GPU to be clear
        2. Run the benchmark
        3. Wait for model to unload (if wait_after=True)

        Args:
            model: Model name
            task_type: Type of benchmark
            test_func: Async function that runs the test
            wait_after: Wait for model to unload after test

        Returns:
            BenchmarkResult
        """
        # 1. Ensure GPU is clear
        logger.info(f"[{model}] Preparing for isolated benchmark...")
        await self.ensure_single_model(model)

        # 2. Get initial state (for debugging if needed)
        _initial_models = self.get_running_models()  # noqa: F841

        # 3. Run the test
        logger.info(f"[{model}] Running {task_type} benchmark...")
        result = await test_func(model)

        # 4. Get GPU memory used
        running_after = self.get_running_models()
        for m in running_after:
            if model in m.get("name", ""):
                result.gpu_memory_gb = m.get("size_gb", 0)
                break

        # 5. Wait for unload (model will timeout naturally)
        if wait_after:
            logger.info(f"[{model}] Waiting {self.UNLOAD_WAIT_TIME}s for unload...")
            await asyncio.sleep(self.UNLOAD_WAIT_TIME)
            # Note: Ollama unloads models after idle timeout (default 5 min)
            # For benchmarking, we may need to use keep_alive=0

        return result

    # =========================================================================
    # Vision Benchmarks (Isolated)
    # =========================================================================

    async def benchmark_vision_isolated(
        self,
        image_path: str,
        prompt: str = "Describe this image briefly in 2-3 sentences.",
        models: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """
        Benchmark vision models ONE AT A TIME with GPU isolation.
        """
        # Load image
        path = Path(image_path)
        if not path.exists():
            raise BenchmarkError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        # Get vision models
        if models is None:
            models = list(self.inspector.get_vision_models().keys())

        if not models:
            raise BenchmarkError("No vision-capable models found")

        logger.info(f"Starting ISOLATED vision benchmark for {len(models)} models")
        logger.info(f"Models: {models}")

        results = []
        for i, model in enumerate(models):
            logger.info(f"\n[{i + 1}/{len(models)}] Benchmarking {model}...")

            async def vision_test(m):
                return await self._run_vision_test(m, image_b64, prompt)

            # Wait for previous model to unload except for last
            wait_after = i < len(models) - 1

            try:
                result = await self.benchmark_model_isolated(
                    model, "vision", vision_test, wait_after=wait_after
                )
                results.append(result)
                logger.info(
                    f"[{model}] {result.latency_ms:.0f}ms, "
                    f"{result.tokens_per_second:.1f} tok/s, "
                    f"{result.gpu_memory_gb:.1f}GB GPU"
                )
            except GPUBusyError as e:
                logger.error(f"[{model}] Skipped: {e}")
                results.append(
                    BenchmarkResult(
                        model=model,
                        task_type="vision",
                        success=False,
                        latency_ms=0,
                        tokens_generated=0,
                        tokens_per_second=0,
                        output_preview="",
                        error=str(e),
                    )
                )

        report = self._create_report("vision", str(path), results)
        self._reports.append(report)
        return report

    async def _run_vision_test(
        self, model: str, image_b64: str, prompt: str
    ) -> BenchmarkResult:
        """Run a single vision test with keep_alive=0 for immediate unload."""
        start_time = time.time()
        tokens = 0
        output = ""
        error = None

        try:
            # Use keep_alive="0" to unload model immediately after response
            response = await self.async_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
                options={"num_predict": 100},  # Limit tokens for benchmark
                keep_alive="0",  # Unload immediately after
            )

            output = response.message.content or ""
            tokens = int(
                response.eval_count
                if hasattr(response, "eval_count") and response.eval_count is not None
                else len(output.split())
            )

        except Exception as e:
            error = str(e)

        latency = (time.time() - start_time) * 1000
        tps = (tokens / (latency / 1000)) if latency > 0 else 0

        return BenchmarkResult(
            model=model,
            task_type="vision",
            success=error is None,
            latency_ms=latency,
            tokens_generated=tokens,
            tokens_per_second=tps,
            output_preview=output[:200] if output else "",
            error=error,
            metadata={"prompt": prompt},
        )

    # =========================================================================
    # Embedding Benchmarks (Isolated)
    # =========================================================================

    async def benchmark_embedding_isolated(
        self, text: str, models: Optional[List[str]] = None
    ) -> BenchmarkReport:
        """Benchmark embedding models ONE AT A TIME."""
        if models is None:
            models = list(self.inspector.get_embedding_models().keys())

        if not models:
            raise BenchmarkError("No embedding-capable models found")

        logger.info(f"Starting ISOLATED embedding benchmark for {len(models)} models")

        results = []
        for i, model in enumerate(models):
            logger.info(f"\n[{i + 1}/{len(models)}] Benchmarking {model}...")

            async def embed_test(m):
                return await self._run_embedding_test(m, text)

            wait_after = i < len(models) - 1

            try:
                result = await self.benchmark_model_isolated(
                    model, "embedding", embed_test, wait_after=wait_after
                )
                results.append(result)
                logger.info(f"[{model}] {result.latency_ms:.0f}ms")
            except GPUBusyError as e:
                logger.error(f"[{model}] Skipped: {e}")
                results.append(
                    BenchmarkResult(
                        model=model,
                        task_type="embedding",
                        success=False,
                        latency_ms=0,
                        tokens_generated=0,
                        tokens_per_second=0,
                        output_preview="",
                        error=str(e),
                    )
                )

        report = self._create_report("embedding", text, results)
        self._reports.append(report)
        return report

    async def _run_embedding_test(self, model: str, text: str) -> BenchmarkResult:
        """Run a single embedding test."""
        start_time = time.time()
        error = None
        dims = 0

        try:
            response = await self.async_client.embed(
                model=model,
                input=text,
                keep_alive="0",  # Unload immediately
            )
            dims = len(response.embeddings[0]) if response.embeddings else 0
        except Exception as e:
            error = str(e)

        latency = (time.time() - start_time) * 1000

        return BenchmarkResult(
            model=model,
            task_type="embedding",
            success=error is None,
            latency_ms=latency,
            tokens_generated=0,
            tokens_per_second=0,
            output_preview=f"dims={dims}",
            error=error,
            metadata={"dimensions": dims, "text_length": len(text)},
        )

    # =========================================================================
    # Completion Benchmarks (Isolated)
    # =========================================================================

    async def benchmark_completion_isolated(
        self, prompt: str, max_tokens: int = 100, models: Optional[List[str]] = None
    ) -> BenchmarkReport:
        """Benchmark completion models ONE AT A TIME."""
        if models is None:
            all_models = self.inspector.get_all_models()
            models = [
                name
                for name, data in all_models.items()
                if "embedding" not in data.get("capabilities", [])
            ]

        if not models:
            raise BenchmarkError("No completion models found")

        logger.info(f"Starting ISOLATED completion benchmark for {len(models)} models")

        results = []
        for i, model in enumerate(models):
            logger.info(f"\n[{i + 1}/{len(models)}] Benchmarking {model}...")

            async def completion_test(m):
                return await self._run_completion_test(m, prompt, max_tokens)

            wait_after = i < len(models) - 1

            try:
                result = await self.benchmark_model_isolated(
                    model, "completion", completion_test, wait_after=wait_after
                )
                results.append(result)
                logger.info(
                    f"[{model}] {result.latency_ms:.0f}ms, "
                    f"{result.tokens_per_second:.1f} tok/s"
                )
            except GPUBusyError as e:
                logger.error(f"[{model}] Skipped: {e}")
                results.append(
                    BenchmarkResult(
                        model=model,
                        task_type="completion",
                        success=False,
                        latency_ms=0,
                        tokens_generated=0,
                        tokens_per_second=0,
                        output_preview="",
                        error=str(e),
                    )
                )

        report = self._create_report("completion", prompt, results)
        self._reports.append(report)
        return report

    async def _run_completion_test(
        self, model: str, prompt: str, max_tokens: int
    ) -> BenchmarkResult:
        """Run a single completion test."""
        start_time = time.time()
        tokens = 0
        output = ""
        error = None

        try:
            response = await self.async_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens},
                keep_alive="0",  # Unload immediately
            )

            output = response.message.content or ""
            tokens = int(
                response.eval_count
                if hasattr(response, "eval_count") and response.eval_count is not None
                else len(output.split())
            )
        except Exception as e:
            error = str(e)

        latency = (time.time() - start_time) * 1000
        tps = (tokens / (latency / 1000)) if latency > 0 else 0

        return BenchmarkResult(
            model=model,
            task_type="completion",
            success=error is None,
            latency_ms=latency,
            tokens_generated=tokens,
            tokens_per_second=tps,
            output_preview=output[:200] if output else "",
            error=error,
            metadata={"max_tokens": max_tokens},
        )

    # =========================================================================
    # Report Generation & Export
    # =========================================================================

    def _create_report(
        self, task_type: str, input_sample: str, results: List[BenchmarkResult]
    ) -> BenchmarkReport:
        """Create a benchmark report from results."""
        successful = [r for r in results if r.success]

        fastest = None
        most_tokens = None

        if successful:
            fastest = min(successful, key=lambda r: r.latency_ms).model
            if any(r.tokens_generated > 0 for r in successful):
                most_tokens = max(successful, key=lambda r: r.tokens_generated).model

        return BenchmarkReport(
            task_type=task_type,
            timestamp=datetime.now().isoformat(),
            input_sample=input_sample,
            models_tested=len(results),
            results=results,
            fastest_model=fastest,
            most_tokens=most_tokens,
        )

    def export_results(
        self,
        output_dir: str,
        reports: Optional[List[BenchmarkReport]] = None,
    ) -> Dict[str, str]:
        """Export benchmark results to JSON files."""
        reports_to_export = reports if reports is not None else self._reports

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        exported = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for report in reports_to_export:
            filename = f"{report.task_type}_benchmark_{timestamp}.json"
            filepath = out_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

            exported[report.task_type] = str(filepath)

        # Summary
        summary_path = out_path / f"benchmark_summary_{timestamp}.json"
        summary = {
            "timestamp": timestamp,
            "gpu_aware": True,
            "isolated_tests": True,
            "total_reports": len(reports_to_export),
            "reports": [r.to_dict() for r in reports_to_export],
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        exported["_summary"] = str(summary_path)
        return exported

    def clear_reports(self):
        """Clear stored reports."""
        self._reports = []

    async def run_all_benchmarks_isolated(
        self,
        image_path: Optional[str] = None,
        text_sample: str = "The quick brown fox.",
        completion_prompt: str = "Write one sentence about AI.",
    ) -> Dict[str, BenchmarkReport]:
        """
        Run all benchmark types with proper GPU isolation.
        Each model is tested individually with unloading between tests.
        """
        results = {}

        # Check initial GPU state
        running = self.get_running_models()
        if running:
            logger.warning(
                f"WARNING: {len(running)} models already loaded. "
                f"Consider waiting for them to unload first."
            )
            logger.warning(f"Running: {[m['name'] for m in running]}")

        # Vision (if image provided)
        if image_path and Path(image_path).exists():
            try:
                logger.info("\n" + "=" * 60)
                logger.info("VISION BENCHMARKS")
                logger.info("=" * 60)
                results["vision"] = await self.benchmark_vision_isolated(image_path)
            except BenchmarkError as e:
                logger.warning(f"Vision benchmark skipped: {e}")

        # Embedding
        try:
            logger.info("\n" + "=" * 60)
            logger.info("EMBEDDING BENCHMARKS")
            logger.info("=" * 60)
            results["embedding"] = await self.benchmark_embedding_isolated(text_sample)
        except BenchmarkError as e:
            logger.warning(f"Embedding benchmark skipped: {e}")

        # Completion (limit to fewer models for reasonable runtime)
        try:
            logger.info("\n" + "=" * 60)
            logger.info("COMPLETION BENCHMARKS")
            logger.info("=" * 60)
            results["completion"] = await self.benchmark_completion_isolated(
                completion_prompt,
                max_tokens=50,  # Small for faster benchmarks
            )
        except BenchmarkError as e:
            logger.warning(f"Completion benchmark skipped: {e}")

        return results
