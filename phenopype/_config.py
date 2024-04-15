from collections import defaultdict

gui_zoom_config = None
last_print_msg = ""
models = defaultdict(dict)
pype_restart = False
template_loaded_current = None
template_path_current = None
verbose = False
verbosity_level = 0  # 0 = all, 1 = warnings, 2 = errors (non-breaking), 3 = errors (breaking)
window_close = False
window_max_dim = 1000
window_min_dim = 100
