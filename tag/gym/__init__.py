import os

# Define package-level paths
GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
GYM_ENVS_DIR = os.path.join(GYM_ROOT_DIR, "gym", "envs")

# Make these variables available for import
__all__ = ["GYM_ROOT_DIR", "GYM_ENVS_DIR"]

# Import subpackages to make them available
from . import envs, utils
