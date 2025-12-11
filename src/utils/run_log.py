"""Helpers for recording cron job executions."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger

from src.data.database import get_session, CronRunLog


class CronRunLogger:
    """Context manager that records cron job lifecycle and metrics."""

    def __init__(self, job_name: str):
        self.job_name = job_name
        self.run_id: Optional[int] = None

    def __enter__(self) -> "CronRunLogger":
        session = get_session()
        try:
            run = CronRunLog(
                job_name=self.job_name,
                started_at=datetime.now(timezone.utc),
                status='running',
                records_processed=0,
                api_calls=0,
                extra_metadata={}
            )
            session.add(run)
            session.commit()
            self.run_id = run.id
            logger.debug(f"[CronRunLogger] Started job '{self.job_name}' (id={self.run_id})")
            return self
        finally:
            session.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = 'failed' if exc_type else 'success'
        session = get_session()
        try:
            run = session.query(CronRunLog).get(self.run_id)
            if run:
                run.completed_at = datetime.now(timezone.utc)
                run.status = status
                if exc_val and not run.error_message:
                    run.error_message = str(exc_val)
                session.commit()
                logger.debug(
                    "[CronRunLogger] Completed job '%s' (id=%s) with status %s",
                    self.job_name,
                    self.run_id,
                    status
                )
        finally:
            session.close()
        # Do not suppress exceptions
        return False

    def update(self, **fields: Any):
        """Update log fields such as records_processed, api_calls, or extra metadata."""
        if self.run_id is None:
            return

        session = get_session()
        try:
            run = session.query(CronRunLog).get(self.run_id)
            if not run:
                return

            extra_updates: Dict[str, Any] = fields.pop('extra_metadata', {})
            for key, value in fields.items():
                if hasattr(run, key):
                    setattr(run, key, value)

            if extra_updates:
                current_meta = run.extra_metadata or {}
                current_meta.update(extra_updates)
                run.extra_metadata = current_meta

            session.commit()
        finally:
            session.close()

