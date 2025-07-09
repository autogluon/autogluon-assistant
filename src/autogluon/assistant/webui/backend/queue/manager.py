# src/autogluon/assistant/webui/backend/queue/manager.py

import logging
import threading
import time
from typing import Dict, Optional

from ..utils import start_run
from .models import TaskDatabase

logger = logging.getLogger(__name__)


class QueueManager:
    """Singleton queue manager that handles task execution"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.db = TaskDatabase()
        self._executor_thread = None
        self._stop_event = threading.Event()
        self._initialized = True
        logger.info("QueueManager initialized")

    def start(self):
        """Start the background task executor"""
        if self._executor_thread and self._executor_thread.is_alive():
            return

        self._stop_event.clear()
        self._executor_thread = threading.Thread(target=self._executor_loop, daemon=True)
        self._executor_thread.start()
        logger.info("Queue executor started")

    def stop(self):
        """Stop the background task executor"""
        self._stop_event.set()
        if self._executor_thread:
            self._executor_thread.join(timeout=5)
        logger.info("Queue executor stopped")

    def submit_task(self, task_id: str, command_data: Dict, credentials: Optional[Dict] = None) -> int:
        """
        Submit a task to the queue
        Returns: queue position (0 means will run immediately)
        """
        position = self.db.add_task(task_id, command_data, credentials)
        logger.info(f"Task {task_id} submitted, position: {position}")

        # If this is the only task, trigger executor immediately
        if position == 0:
            self._trigger_executor()

        return position

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task"""
        success = self.db.cancel_task(task_id)
        if success:
            logger.info(f"Task {task_id} cancelled")
        return success

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status"""
        return self.db.get_task_status(task_id)

    def get_queue_info(self) -> Dict:
        """Get queue information"""
        return self.db.get_queue_info()

    def complete_task_by_run_id(self, run_id: str):
        """Complete a task using run_id (called when subprocess finishes)"""
        task_id = self.db.get_task_by_run_id(run_id)
        if task_id:
            self.db.complete_task(task_id)
            logger.info(f"Task {task_id} (run_id: {run_id}) completed")
            # Trigger executor to check for next task
            self._trigger_executor()

    def _trigger_executor(self):
        """Wake up the executor thread to check for new tasks"""
        # The executor will check on its next iteration
        pass

    def _executor_loop(self):
        """Background loop that executes tasks from the queue"""
        logger.info("Executor loop started")

        while not self._stop_event.is_set():
            try:
                # Clean up any stale tasks
                self.db.cleanup_stale_tasks()

                # Get next task
                task_info = self.db.get_next_task()

                if task_info:
                    task_id, command_data, credentials = task_info
                    logger.info(f"Starting task {task_id}")

                    try:
                        # Start the task using existing infrastructure
                        run_id = start_run(
                            task_id, command_data["cmd"], credentials  # Using task_id as run_id initially
                        )

                        # Update the actual run_id in database
                        self.db.update_task_run_id(task_id, run_id)

                        # Small delay to ensure database update is visible
                        time.sleep(0.1)

                        logger.info(f"Task {task_id} started with run_id {run_id}")

                    except Exception as e:
                        logger.error(f"Failed to start task {task_id}: {str(e)}")
                        # Remove failed task from queue
                        self.db.complete_task(task_id)

            except Exception as e:
                logger.error(f"Error in executor loop: {str(e)}", exc_info=True)

            # Sleep for a short time before checking again
            time.sleep(1)

        logger.info("Executor loop stopped")


# Global instance getter
def get_queue_manager() -> QueueManager:
    """Get the singleton queue manager instance"""
    return QueueManager()
