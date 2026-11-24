"""Точка запуска HTTP-сервиса с локальным MLflow и обучением."""

import os
from pathlib import Path

from fastapi import FastAPI

from pfs.api import create_app
from pfs.processor import Processor
from pfs.registry import ModelRegistry
from pfs.workflow import WorkFlowPipeline


def create_service() -> FastAPI:
    """Собирает обучаемый сервис из путей, настроенных через окружение."""
    data_dir = Path(os.environ.get("PFS_DATA_DIR", "data/landing"))
    state_dir = Path(os.environ.get("PFS_STATE_DIR", "data/service"))
    return create_app(processor=Processor(data_dir, WorkFlowPipeline(), ModelRegistry(state_dir)))
