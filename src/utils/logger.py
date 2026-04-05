import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str):
    """
    Configures a structured logger with console and rotating file output.
    Uses a custom format that includes Request ID for traceability.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    
    # Professional format
    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s'
    )
    
    # 1. Console Output
    console_out = logging.StreamHandler(sys.stdout)
    console_out.setFormatter(log_format)
    logger.addHandler(console_out)
    
    # 2. Rotating File Output
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_out = RotatingFileHandler(
        os.path.join(log_dir, "cardiosense.log"),
        maxBytes=5_000_000, # 5MB
        backupCount=3
    )
    file_out.setFormatter(log_format)
    logger.addHandler(file_out)
    
    return logger
