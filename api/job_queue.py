"""
Simple Job Queue for video generation.

Manages job submission, status tracking, and background processing.
Single-worker queue suitable for single-GPU setups.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Optional, Dict, Any

from .schemas import GenerationRequest
from .generator import generate_video, GenerationProgress, GenerationResult
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status states."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a video generation job."""
    
    job_id: str
    request: GenerationRequest
    status: JobStatus = JobStatus.QUEUED
    
    # Progress tracking
    progress: int = 0
    current_step: int = 0
    total_steps: int = 0
    model_in_use: str = ""
    cache_hits: int = 0
    fresh_computes: int = 0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    video_path: Optional[str] = None
    generation_time: Optional[float] = None
    cache_statistics: Optional[dict] = None
    seed_used: int = -1
    
    # Error
    error: Optional[str] = None
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def estimated_remaining(self) -> Optional[float]:
        """Estimate remaining time based on progress."""
        if self.current_step == 0 or self.elapsed_time is None:
            return None
        time_per_step = self.elapsed_time / self.current_step
        remaining_steps = self.total_steps - self.current_step
        return time_per_step * remaining_steps
    
    def to_status_dict(self) -> dict:
        """Convert job to status response dict."""
        result = {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
        }
        
        if self.status == JobStatus.PROCESSING:
            result.update({
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "model_in_use": self.model_in_use,
                "elapsed_time": self.elapsed_time,
                "estimated_remaining": self.estimated_remaining,
            })
        
        if self.status == JobStatus.COMPLETED:
            result.update({
                "video_url": f"/video/{self.job_id}",
                "generation_time": self.generation_time,
                "metadata": {
                    "model": self.request.model,
                    "schedule": self.request.schedule,
                    "frame_count": self.request.frame_count,
                    "fps": self.request.fps,
                    "width": self.request.width,
                    "height": self.request.height,
                    "sampling_steps": self.request.sampling_steps,
                    "seed": self.seed_used,
                    "prompt": self.request.prompt,
                }
            })
            if self.cache_statistics:
                result["cache_statistics"] = self.cache_statistics
        
        if self.status == JobStatus.FAILED:
            result["error"] = self.error
        
        return result


class JobQueue:
    """
    Thread-safe job queue with background processing.
    
    Manages:
    - Job submission and ID generation
    - Status tracking
    - Background processing thread
    - Job history (keeps last N completed jobs)
    """
    
    def __init__(self, model_manager: ModelManager, max_history: int = 100):
        """
        Initialize the job queue.
        
        Args:
            model_manager: ModelManager with loaded models
            max_history: Maximum completed jobs to keep in history
        """
        self.model_manager = model_manager
        self.max_history = max_history
        
        # Job storage
        self._jobs: Dict[str, Job] = {}
        self._queue: Queue[str] = Queue()  # Queue of job IDs
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background worker
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()
    
    def start_worker(self):
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Worker thread already running")
            return
        
        self._shutdown_flag.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="job-queue-worker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info("Job queue worker started")
    
    def stop_worker(self):
        """Stop the background worker thread."""
        self._shutdown_flag.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            logger.info("Job queue worker stopped")
    
    def submit_job(self, request: GenerationRequest) -> Job:
        """
        Submit a new generation job.
        
        Args:
            request: Generation request
        
        Returns:
            The created Job object
        """
        job_id = str(uuid.uuid4())[:8]  # Short UUID
        
        job = Job(
            job_id=job_id,
            request=request,
            total_steps=request.sampling_steps
        )
        
        with self._lock:
            self._jobs[job_id] = job
            self._queue.put(job_id)
        
        logger.info(f"Job {job_id} submitted: {request.model}, {request.sampling_steps} steps")
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_queue_size(self) -> int:
        """Get number of jobs waiting in queue."""
        return self._queue.qsize()
    
    def _worker_loop(self):
        """Background worker loop - processes jobs one at a time."""
        logger.info("Worker loop started")
        
        while not self._shutdown_flag.is_set():
            try:
                # Wait for a job (with timeout to check shutdown flag)
                try:
                    job_id = self._queue.get(timeout=1.0)
                except:
                    continue
                
                # Get the job
                with self._lock:
                    job = self._jobs.get(job_id)
                
                if job is None:
                    logger.warning(f"Job {job_id} not found in storage")
                    continue
                
                # Process the job
                self._process_job(job)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("Worker loop stopped")
    
    def _process_job(self, job: Job):
        """Process a single job."""
        logger.info(f"Processing job {job.job_id}")
        
        # Update status
        with self._lock:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now()
        
        # Progress callback
        def on_progress(progress: GenerationProgress):
            with self._lock:
                job.progress = progress.progress_percent
                job.current_step = progress.current_step
                job.model_in_use = progress.model_in_use
                job.cache_hits = progress.cache_hits
                job.fresh_computes = progress.fresh_computes
        
        # Run generation
        request = job.request
        result = generate_video(
            model_manager=self.model_manager,
            prompt=request.prompt,
            model_type=request.model,
            schedule=request.schedule,
            width=request.width,
            height=request.height,
            frame_count=request.frame_count,
            fps=request.fps,
            sampling_steps=request.sampling_steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
            enable_caching=request.enable_caching,
            cache_start_step=request.cache_start_step,
            cache_end_step=request.cache_end_step,
            cache_interval=request.cache_interval,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
            job_id=job.job_id,
            progress_callback=on_progress,
        )
        
        # Update job with results
        with self._lock:
            job.completed_at = datetime.now()
            job.generation_time = result.generation_time
            job.seed_used = result.seed_used
            
            if result.success:
                job.status = JobStatus.COMPLETED
                job.video_path = result.video_path
                job.cache_statistics = result.cache_statistics
                job.progress = 100
                logger.info(f"Job {job.job_id} completed in {result.generation_time:.2f}s")
            else:
                job.status = JobStatus.FAILED
                job.error = result.error
                logger.error(f"Job {job.job_id} failed: {result.error}")
        
        # Cleanup old jobs
        self._cleanup_old_jobs()
    
    def _cleanup_old_jobs(self):
        """Remove old completed/failed jobs to prevent memory growth."""
        with self._lock:
            completed_jobs = [
                (job.completed_at, job_id)
                for job_id, job in self._jobs.items()
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED) and job.completed_at
            ]
            
            if len(completed_jobs) > self.max_history:
                # Sort by completion time, oldest first
                completed_jobs.sort()
                
                # Remove oldest jobs
                to_remove = len(completed_jobs) - self.max_history
                for _, job_id in completed_jobs[:to_remove]:
                    del self._jobs[job_id]
                    logger.debug(f"Cleaned up old job {job_id}")


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> Optional[JobQueue]:
    """Get the global job queue instance."""
    return _job_queue


def initialize_job_queue(model_manager: ModelManager) -> JobQueue:
    """
    Initialize the global job queue.
    
    Args:
        model_manager: ModelManager with loaded models
    
    Returns:
        The initialized JobQueue
    """
    global _job_queue
    _job_queue = JobQueue(model_manager)
    _job_queue.start_worker()
    return _job_queue


def shutdown_job_queue():
    """Shutdown the global job queue."""
    global _job_queue
    if _job_queue is not None:
        _job_queue.stop_worker()
        _job_queue = None

