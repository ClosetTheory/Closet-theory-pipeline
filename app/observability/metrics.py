"""Metrics tracking for pipeline observability."""

from collections import defaultdict
from typing import Any, Dict, List


class PipelineMetrics:
    """Thread-safe, in-memory metrics registry for pipeline observability."""

    def __init__(self):
        self._counts: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, List[float]] = defaultdict(list)

    def record_stage_execution(
        self,
        stage: str,
        status: str,
        duration_ms: float,
        attempt: int = 1,
    ):
        self._counts[f"{stage}_total"] += 1
        self._counts[f"{stage}_{status.lower()}"] += 1
        if attempt > 1:
            self._counts[f"{stage}_retries"] += (attempt - 1)
        self._latencies[f"{stage}_latency_ms"].append(duration_ms)

    def record_event(self, event_name: str, count: int = 1):
        self._counts[event_name] += count

    def get_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {"counters": dict(self._counts), "latencies_p50": {}}
        for metric, vals in self._latencies.items():
            if vals:
                sorted_vals = sorted(vals)
                p50 = sorted_vals[len(sorted_vals) // 2]
                snapshot["latencies_p50"][metric] = round(p50, 2)
        return snapshot

    def reset(self):
        self._counts.clear()
        self._latencies.clear()


metrics = PipelineMetrics()
