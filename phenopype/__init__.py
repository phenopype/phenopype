
import importlib.metadata
import warnings

__version__ = importlib.metadata.version("phenopype")

class _PluginsPlaceholder:
    def __init__(self, message="The plugins module could not be imported - is it installed?"):
        self.message = message

    def __getattr__(self, name):
        warnings.warn(self.message, UserWarning)

    def __call__(self, *args, **kwargs):
        warnings.warn(self.message, UserWarning)

from . import utils_lowlevel
from . import config, decorators
from .core import preprocessing, segmentation, measurement, export, visualization
from .main import Project, Project_labelling, Pype
from .utils import load_image, show_image, print_colours, save_image, resize_image
from .tracking import motion_tracker, tracking_method

