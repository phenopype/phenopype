object_detection_plain = """
preprocessing:
segmentation:
- threshold:
    method: otsu 
    blocksize: 99      ## for adaptive 
    constant: 1        ## for adaptive 
    value: 127         ## for binary
    channel: gray
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 0
measurement:
visualization:
- select_canvas:
    canvas: image
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
export:
- save_contours:
    overwrite: true
"""

preset1 = object_detection_plain  # legacy

object_detection_morph = """
segmentation:
- blur:
    kernel_size: 15
- threshold:
    method: otsu 
    blocksize: 99      ## for adaptive 
    constant: 1        ## for adaptive 
    value: 127         ## for binary
    channel: gray
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 3
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 0
visualization:
- select_canvas:
    canvas: image
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
export:
- save_contours:
    overwrite: true
"""

preset3 = """
preprocessing:
- create_mask: # with this you create the boundary around the plates (gets saved after first run)
    label: mask1
    tool: polygon
- find_scale # if you have determined the scale before, it will find it in the image
segmentation:
- blur: 
    kernel_size: 15
- threshold: 
    method: otsu 
    blocksize: 99      ## for adaptive 
    constant: 1        ## for adaptive 
    value: 127         ## for binary
    channel: gray
- morphology:
    operation: close # connect pixels
    shape: ellipse
    kernel_size: 3
    iterations: 3
- draw # connect or disconnect contours (e.g. armour plates)
- find_contours:
    retrieval: ext
    min_diameter: 0
    min_area: 500
visualization:
- select_canvas:
    canvas: red
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0
"""
preset4 = """
preprocessing:
- resize_image:
    factor: 0.5
- create_mask: # with this you create the boundary around the plates (gets saved after first run)
    label: mask1
    tool: polygon
segmentation:
- blur: 
    kernel_size: 15
- threshold: 
    method: otsu 
    blocksize: 99      ## for adaptive 
    constant: 1        ## for adaptive 
    value: 127         ## for binary
    channel: gray
- morphology:
    operation: close # connect pixels
    shape: ellipse
    kernel_size: 3
    iterations: 3
- draw # connect or disconnect contours (e.g. armour plates)
- find_contours:
    retrieval: ext
    min_diameter: 0
    min_area: 500
visualization:
- select_canvas:
    canvas: red
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0
"""

landmarks_plain = """
measurement:
- landmarks:
    point_size: 25
    point_colour: green
    label_size: 3
    label_width: 5
visualization:
- draw_landmarks:
    point_size: 25
    point_colour: green
    label_size: 3
    label_width: 5
export:
- save_landmarks
"""

landmarks_scale = """
preprocessing:
- find_scale
measurement:
- landmarks:
    point_size: 25
    point_colour: green
    label_size: 3
    label_width: 5
visualization:
- draw_landmarks:
    point_size: 25
    point_colour: green
    label_size: 3
    label_width: 5
- draw_masks:
    line_colour: blue
export:
- save_landmarks
- save_masks
"""

demo1 = """
preprocessing:
- create_mask: 
    tool: polygon
segmentation:
- threshold:
    method: adaptive 
    blocksize: 199
    constant: 5
    channel: red
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 3
- find_contours:
    retrieval: ext
    min_diameter: 0
    min_area: 150
measurement:
visualization:
- select_canvas:
    canvas: image
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
- draw_masks
export:
- save_contours:
    overwrite: true
- save_canvas:
    resize: 0.5
    overwrite: true
"""

demo2 = """
preprocessing:
- create_mask
- create_scale:
    mask: true
segmentation:
- blur:
    kernel_size: 15
- threshold:
    method: adaptive
    blocksize: 49
    constant: 5
    channel: green
- morphology:
    operation: open
    shape: cross
    kernel_size: 9
    iterations: 2
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 250
visualization:
- select_canvas:
    canvas: image
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
export:
- save_contours:
    overwrite: true
"""


inverted1 = """
preprocessing:
- invert_image  
segmentation:
- blur:
    kernel_size: 5
- threshold:
    method: binary
    value: 180
    channel: blue
- morphology:
    operation: open
    shape: cross
    kernel_size: 5
    iterations: 3
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 500
visualization:
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
"""

ex1 = """
preprocessing:
- create_mask
- create_scale:
    mask: true
segmentation:
- blur:
    kernel_size: 15
- threshold:
    method: adaptive
    blocksize: 49
    constant: 5
    channel: green
- morphology:
    operation: open
    shape: cross
    kernel_size: 9
    iterations: 2
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 250
visualization:
- select_canvas:
    canvas: image
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
export:
- save_contours:
    overwrite: true
"""

ex2 = """
preprocessing: 
- find_scale
- enter_data
measurement:
- landmarks:
    point_size: 12
    point_colour: green
    label_size: 2
    label_width: 2
visualization:
- draw_masks
- draw_landmarks:
    point_size: 12
    point_colour: green
    label_size: 2
    label_width: 2
export:
- save_landmarks
- save_masks
- save_data_entry
"""

ex3 = """
# preprocessing:
# - create_mask:
    # include: false
    # overwrite: true
segmentation:
# - blur:
    # kernel_size: 9
- threshold:
    method: adaptive
    blocksize: 299
    constant: 10
# - watershed:
    # distance_cutoff: 0.01
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 3
# - draw  # to separate phytoplankton cells
- find_contours:
    retrieval: ext # needs to be ccomp for watershed
    min_diameter: 0
    min_area: 10
visualization:
- select_canvas:
    canvas: raw
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0
export:
- save_contours:
    save_coords: true
"""


ex5_1 = """
preprocessing:
- create_mask:
    tool: polygon
- find_scale
- enter_data:
    columns: ID
segmentation:
- blur:
    kernel_size: 9
- threshold:
    method: adaptive
    blocksize: 99
    constant: 3
    channel: red
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 3
- watershed:
    distance_cutoff: 0.8
# - draw:
    # line_colour: black # "black" connects, "white" separates 
    # overwrite: True
- find_contours:
    retrieval: ccomp
    min_area: 100
measurement:
visualization:
- select_canvas:
    canvas: red
- draw_contours:
    line_width: 2
    label_width: 0
    label_size: 1
    fill: 0.3
- draw_masks
export:
- save_contours:
    overwrite: true
"""


ex5_2 = """
preprocessing:
- create_mask:
    tool: polygon
- find_scale
- enter_data:
    columns: ID
segmentation:
- blur:
    kernel_size: 5
- threshold:
    method: binary
    value: 200
# - draw:
    # line_colour: black # "black" connects, "white" separates 
- find_contours:
    retrieval: ext
    min_area: 100
measurement:
visualization:
- select_canvas:
    canvas: red
- draw_contours:
    line_width: 2
    label_width: 0
    label_size: 1
    fill: 0.3
- draw_masks
export:
- save_contours:
    save_coords: false
    overwrite: true
"""


ex6 = """
preprocessing:
- create_mask
- create_scale
- enter_data
segmentation:
- blur:
    kernel_size: 3
- threshold:
    method: adaptive
    blocksize: 59
    constant: 10
    channel: gray
- watershed:
    distance_cutoff: 0.5
# - draw  # to separate snails
- find_contours:
    retrieval: ccomp # needs to be ccomp for watershed
    min_diameter: 0
    min_area: 200
    subset: child # needs to be child for watershed
measurement:
- colour_intensity:
    background: True
visualization:
- select_canvas:
    canvas: raw
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0
    watershed: true
    bounding_box: True
- draw_masks
export:
- save_contours:
    save_coords: False
- save_colours
- save_masks
"""


ex7 = """
preprocessing:
- create_mask
segmentation:
- threshold:
    method: adaptive
    blocksize: 49
    constant: 5
# - draw
- find_contours:
    retrieval: ccomp
    min_diameter: 50
    min_area: 0
measurement:
- skeletonize
# - polylines
visualization:
- select_canvas:
    canvas: image
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
- draw_masks
# - draw_polylines
export:
- save_contours:
    overwrite: true
"""


test1 = """
preprocessing:
- create_mask:
    tool: polygon
segmentation:
- threshold:
    method: adaptive
    blocksize: 149
    constant: 5
    channel: red
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 3
- find_contours:
    retrieval: ccomp
    min_area: 150
    min_nodes: 5
    min_diameter: 10
    approximation: simple
measurement:
- landmarks
- polylines
- skeletonize
- colour_intensity:
    channels: [gray, rgb]
    background: true
visualization:
- select_canvas:
    canvas: image
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0.3
    skeleton: true
    bounding_box: true
    mark_holes: True
    label: true
    label_colour: black
- draw_masks:
    line_width: 4
    line_colour: red
- draw_polylines:
    line_colour: blue
    line_width: 2
- draw_landmarks:
    point_size: 15
    point_colour: green
    label_size: 2
    label_width: 3
export:
- save_canvas:
    resize: 0.5
    save_suffix: v2
- save_colours:
    save_suffix: v2
    round_digits: 2
- save_contours:
    save_suffix: v2
    flag_subset: parent
    flag_convert_coords: true
- save_landmarks:
    save_suffix: v2
- save_masks:
    save_suffix: v2
- save_polylines:
    save_suffix: v2
"""


ex8_1 = """
preprocessing:
- create_mask: # select teeth
    label: mask1
- create_scale # manual scale selection
segmentation:
- blur: 
    kernel_size: 25
- threshold: 
    method: binary
    value: 170
    blocksize: 49      ## for adaptive 
    constant: 3        ## for adaptive 
    channel: green
- morphology:
    operation: close # connect pixels
    shape: ellipse
    kernel_size: 3
    iterations: 3
# - draw
- find_contours:
    retrieval: ext
    min_diameter: 0
    min_area: 500
measurement:
- shape_features
visualization:
- select_canvas:
    canvas: red
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0
"""


ex8_2 = """
preprocessing:
- create_scale: # manual scale selection
    mask: true
segmentation:
- blur:
    kernel_size: 5
- threshold:
    method: binary
    value: 190
    blocksize: 49      ## for adaptive 
    constant: 3        ## for adaptive 
    channel: blue
    invert: true
- morphology:
    operation: open # connect pixels
    shape: cross
    kernel_size: 3
    iterations: 3
# - draw
- find_contours:
    retrieval: ext
    min_diameter: 0
    min_area: 500
measurement:
- shape_features
visualization:
- select_canvas:
    canvas: red
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0
"""