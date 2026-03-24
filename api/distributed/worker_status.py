"""
Worker health tracking with exponential backoff.

Tracks worker availability and prevents repeated timeout delays
when the spot instance is clearly down.
"""

import logging
import time

logger = logging.getLogger(__name__)


class WorkerStatus:
    """
    Tracks whether the remote 14B worker is available.

    After max_consecutive_failures, backs off for backoff_duration seconds
    before trying again. This prevents wasting time on health checks
    when the spot instance is preempted or unavailable.
    """

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        backoff_duration: float = 300.0,
    ):
        self.max_consecutive_failures = max_consecutive_failures
        self.backoff_duration = backoff_duration

        self._consecutive_failures = 0
        self._last_success_time: float = 0.0
        self._backoff_until: float = 0.0
        self._total_successes = 0
        self._total_failures = 0

    @property
    def is_backing_off(self) -> bool:
        """True if we're in a backoff period (not checking worker)."""
        return time.time() < self._backoff_until

    @property
    def should_try_worker(self) -> bool:
        """True if we should attempt to use the worker."""
        return not self.is_backing_off

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_success(self):
        """Record a successful worker interaction."""
        self._consecutive_failures = 0
        self._last_success_time = time.time()
        self._backoff_until = 0.0
        self._total_successes += 1

    def record_failure(self, reason: str = ""):
        """
        Record a failed worker interaction.

        After max_consecutive_failures, enters backoff period.
        """
        self._consecutive_failures += 1
        self._total_failures += 1

        if reason:
            logger.warning(f"Worker failure #{self._consecutive_failures}: {reason}")

        if self._consecutive_failures >= self.max_consecutive_failures:
            self._backoff_until = time.time() + self.backoff_duration
            logger.warning(
                f"Worker has failed {self._consecutive_failures} times consecutively. "
                f"Backing off for {self.backoff_duration:.0f}s"
            )

    def reset(self):
        """Reset all tracking state (e.g., when a new worker is started)."""
        self._consecutive_failures = 0
        self._backoff_until = 0.0

    def get_stats(self) -> dict:
        """Get tracking statistics."""
        return {
            "consecutive_failures": self._consecutive_failures,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "is_backing_off": self.is_backing_off,
            "backoff_remaining": max(0, self._backoff_until - time.time()),
            "last_success": self._last_success_time,
        }
