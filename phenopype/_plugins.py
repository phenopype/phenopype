from importlib_metadata import entry_points
import sys

discovered_plugins = entry_points(group='phenopype.plugins')

class plugins(object):
    pass

for plug in discovered_plugins:
    setattr(plugins, 'plug.name', discovered_plugins[plug.name].load())

sys.modules["phenopype.plugins"] = plugins

# plug.name

