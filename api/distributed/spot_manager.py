"""
RunPod spot instance auto-management.

Handles starting/stopping spot pods via RunPod GraphQL API.
Integrated into the coordinator to auto-start spot when a job arrives
and auto-stop after idle timeout.
"""

import logging
import os
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# RunPod GraphQL endpoint
RUNPOD_API_URL = "https://api.runpod.io/graphql"


class SpotManager:
    """
    Manages a single spot pod on RunPod.

    Auto-starts when compute is needed, auto-stops after idle timeout.
    Thread-safe for use from the coordinator's job processing loop.
    """

    def __init__(
        self,
        api_key: str,
        gpu_type: str = "NVIDIA H100 80GB HBM3",
        gpu_count: int = 1,
        volume_id: str = "",
        template_id: str = "",
        worker_port: int = 8889,
        idle_timeout: int = 600,
        max_start_wait: int = 180,
    ):
        self.api_key = api_key
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.volume_id = volume_id
        self.template_id = template_id
        self.worker_port = worker_port
        self.idle_timeout = idle_timeout
        self.max_start_wait = max_start_wait

        self._pod_id: Optional[str] = None
        self._worker_url: Optional[str] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        """Check if the spot manager has the required config."""
        return bool(self.api_key)

    @property
    def worker_url(self) -> Optional[str]:
        """Get the worker URL if a pod is running."""
        return self._worker_url

    @property
    def pod_id(self) -> Optional[str]:
        return self._pod_id

    def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query against RunPod API."""
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        resp = requests.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            raise RuntimeError(f"RunPod API error: {data['errors']}")
        return data.get("data", {})

    def get_pod_status(self, pod_id: str) -> Optional[dict]:
        """Get the current status of a pod."""
        query = """
        query getPod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
        """
        try:
            data = self._graphql(query, {"podId": pod_id})
            return data.get("pod")
        except Exception as e:
            logger.warning(f"Failed to get pod status: {e}")
            return None

    def start_spot_pod(self) -> Optional[str]:
        """
        Start a new spot pod on RunPod.

        Returns the pod ID if successful, None otherwise.
        """
        if not self.is_configured:
            logger.warning("SpotManager not configured (no API key)")
            return None

        # Build the startup command
        startup_cmd = (
            f"cd /workspace/wan2.1/Wan2.1 && "
            f"source .venv_worker/bin/activate 2>/dev/null; "
            f"python run_worker.py --port {self.worker_port}"
        )
        if self.gpu_count > 1:
            startup_cmd = (
                f"cd /workspace/wan2.1/Wan2.1 && "
                f"source .venv_worker/bin/activate 2>/dev/null; "
                f"python run_worker.py --port {self.worker_port} --auto-multi-gpu"
            )

        query = """
        mutation createSpotPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                desiredStatus
                imageName
                machineId
            }
        }
        """

        variables = {
            "input": {
                "name": "wan21-worker-spot",
                "gpuTypeId": self.gpu_type,
                "gpuCount": self.gpu_count,
                "cloudType": "SPOT",
                "dockerArgs": startup_cmd,
                "volumeInGb": 0,  # Using network volume
                "containerDiskInGb": 20,
                "minVcpuCount": 4,
                "minMemoryInGb": 32,
            }
        }

        # Add template or image
        if self.template_id:
            variables["input"]["templateId"] = self.template_id
        else:
            variables["input"]["imageName"] = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

        # Attach network volume
        if self.volume_id:
            variables["input"]["networkVolumeId"] = self.volume_id

        try:
            data = self._graphql(query, variables)
            pod_info = data.get("podFindAndDeployOnDemand", {})
            pod_id = pod_info.get("id")

            if pod_id:
                with self._lock:
                    self._pod_id = pod_id
                logger.info(f"Spot pod started: {pod_id}")
                return pod_id
            else:
                logger.warning(f"Failed to start spot pod: {data}")
                return None

        except Exception as e:
            logger.error(f"Failed to start spot pod: {e}")
            return None

    def stop_pod(self) -> bool:
        """Stop the current spot pod."""
        with self._lock:
            pod_id = self._pod_id

        if not pod_id:
            return True

        query = """
        mutation stopPod($podId: String!) {
            podStop(input: {podId: $podId}) {
                id
                desiredStatus
            }
        }
        """
        try:
            self._graphql(query, {"podId": pod_id})
            with self._lock:
                self._pod_id = None
                self._worker_url = None
            logger.info(f"Spot pod stopped: {pod_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop pod {pod_id}: {e}")
            return False

    def terminate_pod(self) -> bool:
        """Terminate (delete) the current spot pod."""
        with self._lock:
            pod_id = self._pod_id

        if not pod_id:
            return True

        query = """
        mutation terminatePod($podId: String!) {
            podTerminate(input: {podId: $podId})
        }
        """
        try:
            self._graphql(query, {"podId": pod_id})
            with self._lock:
                self._pod_id = None
                self._worker_url = None
            logger.info(f"Spot pod terminated: {pod_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate pod {pod_id}: {e}")
            return False

    def wait_for_ready(self, pod_id: Optional[str] = None) -> Optional[str]:
        """
        Wait for the spot pod to be ready and return its worker URL.

        Polls RunPod API until the pod is running or timeout.
        Returns worker URL or None on timeout.
        """
        pod_id = pod_id or self._pod_id
        if not pod_id:
            return None

        start = time.time()
        while time.time() - start < self.max_start_wait:
            status = self.get_pod_status(pod_id)
            if status and status.get("runtime"):
                # Pod is running — construct worker URL
                worker_url = f"https://{pod_id}-{self.worker_port}.proxy.runpod.net"

                # Verify worker HTTP endpoint is up
                try:
                    import requests
                    resp = requests.get(f"{worker_url}/health", timeout=5)
                    if resp.status_code == 200:
                        with self._lock:
                            self._worker_url = worker_url
                        logger.info(f"Worker ready at {worker_url}")
                        return worker_url
                except Exception:
                    pass  # Worker not ready yet, keep polling

            time.sleep(10)

        logger.warning(f"Timed out waiting for pod {pod_id} after {self.max_start_wait}s")
        return None

    def ensure_worker_available(self) -> Optional[str]:
        """
        Ensure a worker is running and return its URL.

        If no pod is running, starts one. If a pod is running, returns its URL.
        Returns None if unable to get a worker ready in time.
        """
        # Check if we already have a running worker
        with self._lock:
            if self._worker_url:
                return self._worker_url

        # Check if pod exists but worker URL is unknown
        with self._lock:
            pod_id = self._pod_id

        if pod_id:
            url = self.wait_for_ready(pod_id)
            if url:
                return url

        # Start a new pod
        logger.info("Starting spot pod for 14B worker...")
        pod_id = self.start_spot_pod()
        if not pod_id:
            return None

        return self.wait_for_ready(pod_id)

    def start_idle_timer(self):
        """Start or reset the idle shutdown timer."""
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()

            self._idle_timer = threading.Timer(
                self.idle_timeout,
                self._idle_shutdown,
            )
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def cancel_idle_timer(self):
        """Cancel the idle shutdown timer (called when a new job arrives)."""
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()
                self._idle_timer = None

    def _idle_shutdown(self):
        """Called when idle timeout expires."""
        logger.info(f"Idle timeout ({self.idle_timeout}s) reached, stopping spot pod...")
        self.terminate_pod()

    def get_status(self) -> dict:
        """Get the current spot manager status."""
        with self._lock:
            return {
                "configured": self.is_configured,
                "pod_id": self._pod_id,
                "worker_url": self._worker_url,
                "gpu_type": self.gpu_type,
                "gpu_count": self.gpu_count,
                "idle_timeout": self.idle_timeout,
            }
