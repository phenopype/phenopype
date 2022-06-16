from ._version import __version__

from .main import Project, Pype

from . import core 
from . import plugins 
from . import utils 
from . import tracking

## this might be removed in the future
from .main import Project, Pype
from .core import preprocessing, segmentation, measurement, export, visualization
from .utils import load_image, show_image, print_colours, save_image, load_template
from .tracking import motion_tracker, tracking_method
