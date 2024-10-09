#%% load modules

import cv2
from pprint import PrettyPrinter


#%% defaults

auto_line_width_factor = 0.0025
auto_point_size_factor = 0.0025
auto_text_width_factor = 0.0025
auto_text_size_factor = 0.0005

confirm_options = ["True", "true", "y", "yes", True]

_default_instructions_colour = "green"
_default_label_colour = "red"
_default_line_colour = "lime"
_default_node_colour = "red"
_default_point_colour = "red"
_default_overlay_colour_left = "lime"
_default_overlay_colour_right = "red"


default_filetypes = ["jpg", "JPG", "jpeg", "JPEG", "tif", "png", "bmp"]
default_meta_data_fields = [
    "DateTimeOriginal",
    "Model",
    "LensModel",
    "ExposureTime",
    "ISOSpeedRatings",
    "FNumber",
]

default_save_suffix = "v1"
default_window_size = 1000

pandas_max_rows = 10
pretty = PrettyPrinter(width=30)

strftime_format = "%Y-%m-%d %H:%M:%S"

#%% flags opencv

opencv_contour_flags = {
    "retrieval": {
        "ext": cv2.RETR_EXTERNAL,  ## only external
        "list": cv2.RETR_LIST,  ## all contours
        "tree": cv2.RETR_TREE,  ## fully hierarchy
        "ccomp": cv2.RETR_CCOMP,  ## outer perimeter and holes
        "flood": cv2.RETR_FLOODFILL,  ## not sure what this does
    },
    "approximation": {
        "none": cv2.CHAIN_APPROX_NONE,  ## all points (no approx)
        "simple": cv2.CHAIN_APPROX_SIMPLE,  ## minimal corners
        "L1": cv2.CHAIN_APPROX_TC89_L1,
        "KCOS": cv2.CHAIN_APPROX_TC89_KCOS,
    },
}

opencv_distance_flags = {
    "user": cv2.DIST_USER,
    "l1": cv2.DIST_L1,
    "l2": cv2.DIST_L2,
    "C": cv2.DIST_C,
    "l12": cv2.DIST_L12,
    "fair": cv2.DIST_FAIR,
    "welsch": cv2.DIST_WELSCH,
    "huber": cv2.DIST_HUBER,
}

opencv_font_flags = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "plain": cv2.FONT_HERSHEY_PLAIN,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "complex-small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "script-simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "script-complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    "italic": cv2.FONT_ITALIC, 

}

opencv_interpolation_flags = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
    "linear_exact": cv2.INTER_LINEAR_EXACT,
    "nearest_exact": cv2.INTER_NEAREST_EXACT,
    "max": cv2.INTER_MAX,
    "warp_fill_outliers": cv2.WARP_FILL_OUTLIERS,
    "warp_inverse_map": cv2.WARP_INVERSE_MAP,
}

opencv_morphology_flags = {
    "shape_list": {
        "cross": cv2.MORPH_CROSS,
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
    },
    "operation_list": {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
        "hitmiss": cv2.MORPH_HITMISS,
    },
}

# opencv_skeletonize_flags = {
#     "zhangsuen": cv2.ximgproc.THINNING_ZHANGSUEN,
#     "guohall": cv2.ximgproc.THINNING_GUOHALL,
# }

opencv_window_flags = {
    "normal": cv2.WINDOW_NORMAL,
    "auto": cv2.WINDOW_AUTOSIZE,
    "openGL": cv2.WINDOW_OPENGL,
    "full": cv2.WINDOW_FULLSCREEN,
    "free": cv2.WINDOW_FREERATIO,
    "keep": cv2.WINDOW_KEEPRATIO,
    "GUIexp": cv2.WINDOW_GUI_EXPANDED,
    "GUInorm": cv2.WINDOW_GUI_NORMAL,
}

#%% annotation definitions

## gui data
_coord_type = "points"
_coord_list_type = "polygons"
_sequence_type = "drawings"

## annotation data
_comment_type = "comment"
_contour_type = "contour"
_drawing_type = "drawing"
_landmark_type = "landmark"
_line_type = "line"
_mask_type = "mask"
_reference_type = "reference"
_shape_feature_type = "shape_features"
_texture_feature_type = "texture_features"

## USE DECORATORS
_annotation_functions = {
    ## comments
    "write_comment": _comment_type,
    "detect_QRcode": _comment_type,
    
    ## contours
    "detect_contour": _contour_type,
    "mask_to_contour": _contour_type,

    ## drawings
    "edit_contour": _drawing_type,
    
    ## landmarks
    "set_landmark": _landmark_type,
    "detect_landmark": _landmark_type,

    ## lines
    "set_polyline": _line_type,
    "detect_skeleton": _line_type,
    
    ## masks
    "contour_to_mask": _mask_type,
    "create_mask": _mask_type,
    "detect_mask": _mask_type,
    
    ## reference
    "create_reference": _reference_type,
    "detect_reference": _reference_type,
    
    ## shape_features
    "compute_shape_moments": _shape_feature_type,
    
    ## texture_features
    "compute_color_moments": _texture_feature_type,
}

_annotation_types = list(set(_annotation_functions.values()))

#%% labelling 

ascii_codes = {i: chr(i) for i in range(128)}


#%% GUI definitions

## GET RID OF THIS
_GUI_data_args = [
    'comment', 
    'contour', 
    'points', 
    'polygons', 
    'drawings'
    ]


#%% legacy 

_legacy_names = {
    "preprocessing": {
        "enter_data": "write_comment",
        "comment": "write_comment",
        "detect_shape": "detect_mask",
        },
    "segmentation": {
        
        ## plugins
        "detect_object": "predict_keras",

        ## main
        "detect_contours": "detect_contour",
        "edit_contours": "edit_contour",
    },
    "measurement": {
        
        ## plugins
        "compute_texture_features": "extract_radiomic_features",
        "texture_features": "extract_radiomic_features",
        
        ## main
        "set_landmark": "set_landmark",
        "set_landmarks": "set_landmark",
        "landmark": "set_landmark",
        "landmarks": "set_landmark",
        "shape_features": "compute_shape_moments",
        "compute_texture_moments": "compute_color_moments",

    },
    "visualization": {
        "draw_landmarks": "draw_landmark",
        "draw_contours": "draw_contour",
    },
    "export": {},
}
