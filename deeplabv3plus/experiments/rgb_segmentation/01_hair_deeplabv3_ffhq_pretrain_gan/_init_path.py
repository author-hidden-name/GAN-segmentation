import os
import sys
from pathlib import Path

EXP_PATH = Path(os.path.realpath(__file__)).parent
lib_path = EXP_PATH

while lib_path.name != 'experiments':
    lib_path = lib_path.parent
lib_path = lib_path.parent

sys.path.insert(0, str(lib_path))

