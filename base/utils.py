# coding=utf-8

# Standard Library Imports
import os
import json
import logging
from logging.handlers import RotatingFileHandler


def create_dir(directory='./output'):
    """
        This function creates the specified directory if it doesn't already exist.

        Parameters:
            directory (str, optional): The path to the directory to be created. Defaults to the current directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def setup_logger(name, log_dir='./log', log_filename='dhm.log', level=logging.INFO, max_size=10485760, backup_count=5):
    """
        Set up the logger configuration and return the logger instance.

        Parameters:
            name (str): Name of the logger instance.
            log_dir (str, optional): Directory path to store log files. Defaults to "./log".
            log_filename (str, optional): Log file name. Defaults to "dhm.log".
            level (int, optional): Logging level. Defaults to logging.INFO.
            max_size (int, optional): Maximum size of the log file in bytes (default: 10 MB).
            backup_count (int, optional): Number of backups to keep (default: 5).

        Returns:
            logging.Logger: Configured logger instance for logging messages.
    """
    # Create the log directory if it doesn't exist
    create_dir(directory=log_dir)

    log_filepath = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = RotatingFileHandler(log_filepath, maxBytes=max_size, backupCount=backup_count)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers and remove them if necessary
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    return logger


def load_config(config_file="./config/config.json"):
    """
        Load configuration settings from the specified JSON config file.

        Parameters:
            config_file (str, optional): Path to the JSON config file. Defaults to "./config/config.json".

        Returns:
            dict: Dictionary containing the configuration settings.
    """
    # Get the absolute path to the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if config_file is None:
        config_file = os.path.join(base_dir, '..', 'config', 'config.json')
        
    with open(os.path.abspath(os.path.join(base_dir, '..', config_file))) as f:
        config = json.load(f)
    
    return config
