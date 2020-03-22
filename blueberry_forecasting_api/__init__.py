"""
The model package. This package contains the definition of the model.
"""
import os
import logging.config
import pkg_resources

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.environ.get('LOG_DIR', './logs')
LOG_PATH = os.path.join(LOG_DIR, "blueberry_forecasting_api.log")

try:
    VERSION = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    VERSION = None  # try-except used when package not yet installed

# set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.config.fileConfig(os.path.join(MODULE_DIR, "logging.ini"), defaults={'logfilename': LOG_PATH})
