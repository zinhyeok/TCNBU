# loggerhandler.py
import os
import logging

# 전역 timestamp 저장소
_GLOBAL_LOGGING_CONTEXT = {}

def setup_module_logger(name="project", log_filename=None, timestamp=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # ✅ project 하위 로거일 경우, 상위 timestamp 자동 상속
    if name.startswith("project.") and timestamp is None:
        timestamp = _GLOBAL_LOGGING_CONTEXT.get("project")

    # ✅ project 자체 로거는 timestamp 명시 필요 (최초 1회)
    if name == "project" and timestamp is not None:
        _GLOBAL_LOGGING_CONTEXT["project"] = timestamp

    if not timestamp:
        raise ValueError(f"[{name}] timestamp is not set and cannot be inferred from 'project'")

    if not logger.handlers:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(base_dir, ".."))

        log_dir = os.path.join(root_dir, f"log/log_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        fname = log_filename or name.split(".")[-1]
        log_path = os.path.join(log_dir, f"{fname}_{timestamp}.log")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(f"[{name}] %(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

    return logger
