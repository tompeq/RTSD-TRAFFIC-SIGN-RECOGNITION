"""
Логирование процесса обучения
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict


def setup_logger(log_dir: Path, name: str = 'training') -> logging.Logger:
    """
    Настройка логгера
    
    Args:
        log_dir: директория для логов
        name: имя логгера
    
    Returns:
        настроенный логгер
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def log_metrics(logger: logging.Logger, epoch: int, metrics: Dict[str, float], prefix: str = ''):
    """
    Логирование метрик
    
    Args:
        logger: логгер
        epoch: номер эпохи
        metrics: словарь с метриками
        prefix: префикс для метрик
    """
    msg = f"Epoch {epoch}"
    if prefix:
        msg += f" [{prefix}]"
    msg += " - " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(msg)