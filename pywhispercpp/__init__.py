import os
import ctypes
from glob import glob

# ggml should be loaded first
ggml_libs = glob(os.path.join(os.path.dirname(__file__), 'lib/*ggml*'))
libs = glob(os.path.join(os.path.dirname(__file__), 'lib/*'))

# Append lib dir to PATH
if os.name == 'nt':
    os.add_dll_directory(os.path.join(os.path.dirname(__file__), 'lib'))

for file in ggml_libs + libs:
    ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'lib', file))
