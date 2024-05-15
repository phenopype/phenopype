
import importlib.metadata

__version__ = importlib.metadata.version("phenopype")

from . import _config, utils_lowlevel

from .main import Project, Project_labelling, Pype
from .core import preprocessing, segmentation, measurement, export, visualization
from .utils import load_image, show_image, print_colours, save_image, load_template
from .tracking import motion_tracker, tracking_method

try:
    import phenopype_plugins as plugins
except:
    pass
