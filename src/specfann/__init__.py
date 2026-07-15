import os
import importlib.metadata


__version__ = importlib.metadata.version("specfann")


# ----- silence TensorFlow logging-----------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

# optional: silence oneDNN / CPU feature spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

devnull = os.open(os.devnull, os.O_WRONLY)
old_stderr = os.dup(2)

os.dup2(devnull, 2)

try:
    import tensorflow as tf
finally:
    os.dup2(old_stderr, 2)
    os.close(devnull)
    os.close(old_stderr)

# further suppress Python-side logging
tf.get_logger().setLevel("ERROR")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


# ----- load in the modules -----------



from . import specfann
from . import pyGA
from .io_functions import open_project

# from .specfann import *

from .single_star import single_star