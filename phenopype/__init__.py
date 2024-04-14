from ._version import __version__

from . import core 
from . import utils 
from . import tracking
from . import _config

from .main import Project, Project_labelling, Pype
from .core import preprocessing, segmentation, measurement, export, visualization
from .utils import load_image, show_image, print_colours, save_image, load_template
from .tracking import motion_tracker, tracking_method

try:
    import phenopype_plugins as plugins
except:
    pass