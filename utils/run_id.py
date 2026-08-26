import os
import time
import uuid
import subprocess
from datetime import datetime

def get_git_info():
    """Safely retrieves git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", False

def generate_run_id(name: str) -> str:
    """Generates a unique run ID: {name}_{YYYYMMDD_HHMMSS}_{uuid[:6]}"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:6]
    return f"{name}_{timestamp}_{short_uuid}"
