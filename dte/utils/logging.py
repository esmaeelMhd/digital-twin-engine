"""Logging utilities for training and evaluation."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "dte",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, use_wandb: bool = False):
        """Initialize metrics logger.
        
        Args:
            use_wandb: Whether to use Weights & Biases
        """
        self.use_wandb = use_wandb
        self.metrics_history = []
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not available")
                self.use_wandb = False
    
    def log(self, metrics: dict, step: int):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        self.metrics_history.append({"step": step, **metrics})
        
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
    
    def get_history(self) -> list:
        """Get metrics history.
        
        Returns:
            List of metric dictionaries
        """
        return self.metrics_history
