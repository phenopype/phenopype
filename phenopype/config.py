from collections import defaultdict

gui_zoom_config = None
last_print_msg = ""
pype_restart = False

models = defaultdict(dict)
reference_templates = defaultdict(dict)

template_loaded_current = None
template_path_current = None

verbose = False
verbosity_level = 0  # 0 = all, 1 = warnings, 2 = errors (non-breaking), 3 = errors (breaking)

min_visible_px = 5
max_linewidth_px = 20
instructions_show = True
instructions_pos = (0.1,0.7)
window_close = False
window_max_dim = 1000
window_min_dim = 100
