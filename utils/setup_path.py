import os
import sys

def setup_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(base_dir, 'src', 'features'))
    sys.path.append(os.path.join(base_dir, 'src', 'models'))
    sys.path.append(os.path.join(base_dir, 'ext'))
