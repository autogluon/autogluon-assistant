# src/autogluon/assistant/webui/backend/queue/models.py

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TaskDatabase:
    """SQLite database for task queue management"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default path in user's home directory
            db_dir = Path.home() / ".autogluon_assistant"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "webui_queue.db")
        
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with tasks table"""
        with self._get_connection() as conn:
            # Create table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE NOT NULL,
                    run_id TEXT,
                    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'cancelled')),
                    command_json TEXT NOT NULL,
                    credentials_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP
                )
            """)
            
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)")

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_task(self, task_id: str, command_data: Dict, credentials: Optional[Dict] = None) -> int:
        """
        Add a new task to the queue
        Returns: queue position (0 means will run immediately)
        """
        with self._get_connection() as conn:
            # Check if there are any running or queued tasks
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM tasks WHERE status IN ('queued', 'running')"
            )
            position = cursor.fetchone()['count']
            
            # Insert new task
            conn.execute("""
                INSERT INTO tasks (task_id, status, command_json, credentials_json)
                VALUES (?, 'queued', ?, ?)
            """, (
                task_id,
                json.dumps(command_data),
                json.dumps(credentials) if credentials else None
            ))
            
            return position

    def get_next_task(self) -> Optional[Tuple[str, Dict, Optional[Dict]]]:
        """
        Get the next queued task and mark it as running
        Returns: (task_id, command_data, credentials) or None
        """
        with self._get_connection() as conn:
            # Check if there's already a running task
            cursor = conn.execute("SELECT COUNT(*) as count FROM tasks WHERE status = 'running'")
            if cursor.fetchone()['count'] > 0:
                return None
            
            # Get the oldest queued task
            cursor = conn.execute("""
                SELECT id, task_id, command_json, credentials_json
                FROM tasks
                WHERE status = 'queued'
                ORDER BY id ASC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Update status to running
            conn.execute("""
                UPDATE tasks
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (row['id'],))
            
            return (
                row['task_id'],
                json.loads(row['command_json']),
                json.loads(row['credentials_json']) if row['credentials_json'] else None
            )

    def update_task_run_id(self, task_id: str, run_id: str):
        """Update the run_id after task starts"""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET run_id = ? WHERE task_id = ?",
                (run_id, task_id)
            )

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status and position"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, status, run_id, created_at, started_at
                FROM tasks
                WHERE task_id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            result = {
                'task_id': task_id,
                'status': row['status'],
                'run_id': row['run_id'],
                'created_at': row['created_at'],
                'started_at': row['started_at']
            }
            
            # Calculate position if queued
            if row['status'] == 'queued':
                cursor = conn.execute("""
                    SELECT COUNT(*) as position
                    FROM tasks
                    WHERE status IN ('queued', 'running') AND id < ?
                """, (row['id'],))
                result['position'] = cursor.fetchone()['position']
            else:
                result['position'] = 0
            
            return result

    def get_queue_info(self) -> Dict:
        """Get overall queue information"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running
                FROM tasks
            """)
            row = cursor.fetchone()
            
            return {
                'queued': row['queued'] or 0,
                'running': row['running'] or 0,
                'total_waiting': (row['queued'] or 0) + (row['running'] or 0)
            }

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task (only if queued)
        Returns: True if cancelled, False if not found or already running
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            
            if not row or row['status'] != 'queued':
                return False
            
            # Delete the task (as per requirement)
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            return True

    def complete_task(self, task_id: str):
        """Mark task as completed and remove from database"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

    def get_task_by_run_id(self, run_id: str) -> Optional[str]:
        """Get task_id by run_id"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT task_id FROM tasks WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            return row['task_id'] if row else None

    def cleanup_stale_tasks(self, timeout_seconds: int = 3600):
        """Clean up tasks that have been running too long (likely crashed)"""
        with self._get_connection() as conn:
            conn.execute("""
                DELETE FROM tasks
                WHERE status = 'running'
                AND datetime(started_at, '+' || ? || ' seconds') < datetime('now')
            """, (timeout_seconds,))