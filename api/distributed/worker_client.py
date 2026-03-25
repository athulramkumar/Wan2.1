"""
HTTP client for coordinator → worker communication.

Sends segment execution requests and polls for completion.
All data transfer happens via shared volume — only control messages go over HTTP.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class WorkerClient:
    """HTTP client for communicating with the 14B worker."""

    def __init__(
        self,
        worker_url: str,
        health_timeout: float = 5.0,
        request_timeout: float = 30.0,
    ):
        self.worker_url = worker_url.rstrip("/")
        self.health_timeout = health_timeout
        self.request_timeout = request_timeout
        self._session = requests.Session()

    def health_check(self) -> bool:
        """
        Check if the worker is alive and ready.

        Returns True if healthy, False otherwise.
        """
        try:
            resp = self._session.get(
                f"{self.worker_url}/health",
                timeout=self.health_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "ok"
            return False
        except Exception as e:
            logger.debug(f"Worker health check failed: {e}")
            return False

    def is_busy(self) -> bool:
        """Check if the worker is currently running a segment."""
        try:
            resp = self._session.get(
                f"{self.worker_url}/health",
                timeout=self.health_timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("is_busy", False)
            return True  # Assume busy on error
        except Exception:
            return True

    def submit_segment(
        self,
        checkpoint_path: str,
        result_path: str,
    ) -> Optional[str]:
        """
        Submit a segment for execution on the worker.

        Args:
            checkpoint_path: Path to input checkpoint on shared volume
            result_path: Path where worker should write result

        Returns:
            task_id if submitted successfully, None on failure
        """
        try:
            resp = self._session.post(
                f"{self.worker_url}/run-segment",
                json={
                    "checkpoint_path": checkpoint_path,
                    "result_path": result_path,
                },
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                task_id = data.get("task_id")
                logger.info(f"Segment submitted to worker: task_id={task_id}")
                return task_id
            else:
                logger.warning(f"Worker rejected segment: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            logger.warning(f"Failed to submit segment to worker: {e}")
            return None

    def poll_segment(self, task_id: str) -> Optional[dict]:
        """
        Get the current status of a segment task.

        Returns status dict or None on failure.
        """
        try:
            resp = self._session.get(
                f"{self.worker_url}/segment-status/{task_id}",
                timeout=self.health_timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def run_segment_inline(
        self,
        checkpoint_data: bytes,
        timeout: float = 600.0,
    ) -> bytes:
        """
        Send checkpoint data to worker and get result back inline.

        No shared filesystem needed. Transfers checkpoint over HTTP.
        Returns result bytes or raises on failure.
        """
        try:
            resp = self._session.post(
                f"{self.worker_url}/run-segment-inline",
                data=checkpoint_data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=timeout,
            )
            if resp.status_code == 200:
                logger.info(f"Inline segment completed ({len(resp.content)/1e6:.1f}MB result)")
                return resp.content
            else:
                logger.warning(f"Worker inline segment failed: {resp.status_code} {resp.text[:200]}")
                raise RuntimeError(f"Worker returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Inline segment failed: {e}")
            raise

    def wait_for_segment(
        self,
        task_id: str,
        timeout: float,
        poll_interval: float = 5.0,
        progress_callback=None,
    ) -> Optional[dict]:
        """
        Poll until segment completes, fails, or times out.

        Args:
            task_id: The task ID from submit_segment
            timeout: Max seconds to wait
            poll_interval: Seconds between polls
            progress_callback: Optional callback(current_step, total_steps)

        Returns:
            Final status dict if completed, None if timeout/failure
        """
        start = time.time()

        while time.time() - start < timeout:
            status = self.poll_segment(task_id)

            if status is None:
                # Worker unreachable — likely preempted
                logger.warning("Worker unreachable during segment execution")
                return None

            if status.get("status") == "completed":
                logger.info(f"Worker segment completed in {status.get('elapsed_time', 0):.1f}s")
                return status

            if status.get("status") == "failed":
                logger.warning(f"Worker segment failed: {status.get('error')}")
                return status

            # Still running — report progress
            if progress_callback:
                progress_callback(
                    status.get("current_step", 0),
                    status.get("total_steps", 0),
                )

            time.sleep(poll_interval)

        logger.warning(f"Worker segment timed out after {timeout:.0f}s")
        return None
