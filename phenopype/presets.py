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

preset1 = object_detection_plain # legacy

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

preset3="""
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
preset4="""
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

ex6 = """
preprocessing:
- create_mask
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
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 200
measurement:
- colour_intensity
visualization:
- select_canvas:
    canvas: raw
- draw_contours:
    line_width: 2
    label_width: 1
    label_size: 1
    fill: 0
    watershed: true
export:
- save_contours:
    overwrite: true
"""


ex7 = """
preprocessing:
- create_mask
segmentation:
- threshold:
    method: adaptive
    blocksize: 49
    constant: 5
- draw                  # if worms touch borders
- find_contours:
    retrieval: ccomp
    min_diameter: 50
    min_area: 0
measurement:
- skeletonize
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