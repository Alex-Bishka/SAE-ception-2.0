"""Logging utilities for SAE-ception."""

import logging
import sys


def get_logger(name: str = "sae_ception") -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name (usually __name__ from calling module)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
    
    return logger


def setup_logging(level: int = logging.INFO):
    """
    Setup root logger configuration.
    
    This is typically called once at the start of a script.
    Hydra will automatically capture logging output to files.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )