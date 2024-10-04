import os
import ctypes
from glob import glob

# ggml should be loaded first
ggml_libs = glob(os.path.join(os.path.dirname(__file__), '*ggml*'))
libs = glob(os.path.join(os.path.dirname(__file__), '*whisper*'))

os.add_dll_directory(os.path.dirname(__file__))

for file in ggml_libs + libs:
    ctypes.CDLL(os.path.join(os.path.dirname(__file__), file))
