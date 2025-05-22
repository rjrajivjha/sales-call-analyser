import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger():
    """Set up logging configuration for the application and return a logger instance."""
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger('sales_call_analyzer')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    
    # Create a rotating file handler that rotates logs when they reach 5MB
    log_file = os.path.join(logs_dir, f'sales_call_analyzer_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5
    )
    
    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Alias for backward compatibility
setup_logging = setup_logger
