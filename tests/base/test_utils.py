import os
import json
from base.utils import create_dir, setup_logger, load_config

def test_create_dir():
    test_dir = './test_directory'
    create_dir(test_dir)
    assert os.path.exists(test_dir)
    os.rmdir(test_dir)

def test_setup_logger():
    test_log_dir = './log'
    test_log_filename = 'test.log'
    logger = setup_logger('test_logger', log_dir=test_log_dir, log_filename=test_log_filename)
    assert logger is not None
    assert os.path.exists(os.path.join(test_log_dir, test_log_filename))

    # Close logger handlers and remove the log file
    for handler in logger.handlers:
        handler.close()
    os.remove(os.path.join(test_log_dir, test_log_filename))

def test_load_config():
    test_config_file = './config/test_config.json'
    # Create a test config file with the required keys and values
    test_config = {'key': 'value'}
    with open(test_config_file, 'w') as f:
        json.dump(test_config, f)
    
    config = load_config(config_file=test_config_file)
    assert isinstance(config, dict)
    assert 'key' in config
    assert config['key'] == 'value'

    # Remove the test config file after testing
    os.remove(test_config_file)
