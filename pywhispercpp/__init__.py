import os
import ctypes
from glob import glob

lib_dir = os.path.join(os.path.dirname(__file__), 'lib/*')

for file in glob(lib_dir):
    ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'lib', file))
