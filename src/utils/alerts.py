"""Alert helper utilities (currently log-based)."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from loguru import logger

from src.data.database import get_session, CronRunLog


class AlertManager:
    """Simple alert manager that currently logs warnings."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.monitoring_cfg = self.config.get('monitoring', {})

    def _load_config(self, path: str) -> Dict[str, Any]:
        config_path = Path(path)
        if config_path.exists():
            try:
                with config_path.open('r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                logger.warning(f"Failed to load monitoring config: {exc}")
        return {}

    def notify(self, title: str, context: Optional[Dict[str, Any]] = None):
        """Send an alert (currently via logs)."""
        channel = self.monitoring_cfg.get('alert_channel', 'log')
        if channel == 'log':
            logger.warning(f"[ALERT] {title} | {context or {}}")
        else:
            logger.warning(f"[ALERT:{channel}] {title} | {context or {}} (channel not implemented)")

    def check_cron_failures(self, job_name: str):
        """Alert if a cron job fails consecutively past threshold."""
        cron_cfg = self.monitoring_cfg.get('cron_alerts', {})
        threshold = cron_cfg.get('max_consecutive_failures')
        if not threshold:
            return

        session = get_session()
        try:
            runs = (
                session.query(CronRunLog)
                .filter(CronRunLog.job_name == job_name)
                .order_by(CronRunLog.started_at.desc())
                .limit(threshold)
                .all()
            )
        finally:
            session.close()

        if len(runs) < threshold:
            return

        if all(run.status == 'failed' for run in runs):
            self.notify(
                f"{job_name} cron failed {threshold} times in a row",
                {
                    'job': job_name,
                    'threshold': threshold,
                }
            )

    def check_accuracy_threshold(self, accuracy: Optional[float], meta: Optional[Dict[str, Any]] = None):
        """Alert if accuracy falls below configured floor."""
        threshold = self.monitoring_cfg.get('accuracy_threshold', {}).get('minimum_accuracy')
        if threshold is None or accuracy is None:
            return
        if accuracy < threshold:
            context = {'accuracy': accuracy, 'threshold': threshold}
            if meta:
                context.update(meta)
            self.notify("Prediction accuracy below threshold", context)


_ALERT_MANAGER: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Return a singleton alert manager."""
    global _ALERT_MANAGER
    if _ALERT_MANAGER is None:
        _ALERT_MANAGER = AlertManager()
    return _ALERT_MANAGER

