import sys
import os
import traceback

__package__ = 'easydl'

from .common import *
from .pytorch import *

try:
    import tensorflow
    from tf import *
except ImportError as e:
    print('[easydl] tensorflow not available!')

import warnings

warnings.filterwarnings('ignore', '.*')