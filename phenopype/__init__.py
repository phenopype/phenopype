from ._version import __version__

from .main import Project, Pype
from .core import preprocessing, segmentation, measurement, export, visualization
from .utils import load_image, show_image, save_image, load_template
from .tracking import motion_tracker, tracking_method

# from ._plugins import plugins

from importlib_metadata import entry_points
import sys

discovered_plugins = entry_points(group='phenopype.plugins')

# class plugins(object):
#     pass

# for plug in discovered_plugins:
#     setattr(plugins, plug.name, discovered_plugins[plug.name].load())

# sys.modules["phenopype"] = plugins
