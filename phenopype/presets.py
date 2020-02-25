preset1 = """
segmentation:
- blur:
    kernel_size: 15
- threshold:
    method: otsu
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 10
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 0
measurement:
- colour:
    channels: [gray, rgb]
visualization:
- show_image:
    canvas: image
- show_contours:
    line_thickness: 2
    text_thickness: 1
    text_size: 1
    fill: 0.3
- show_mask:
    colour: blue
    line_thickness: 5
"""

preset2 = """
preprocessing:
- create_mask:
    label: mask1
segmentation:
- blur:
    kernel_size: 15
- threshold:
    method: otsu
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 10
- find_contours:
    retrieval: ccomp
    min_diameter: 0
    min_area: 0
measurement:
- colour:
    channels: [gray, rgb]
visualization:
- show_image:
    canvas: image
- show_contours:
    line_thickness: 2
    text_thickness: 1
    text_size: 1
    fill: 0.3
- show_mask:
    colour: blue
    line_thickness: 5
export:
- save_results:
    overwrite: true
- save_overlay:
    resize: 0.5
    overwrite: true
"""


landmark_preset = """
preprocessing:
- create_mask:
    label: mask1
- enter_data:
    ID:
measurement:
- landmarks:
    show: True
visualization:
- show_image:
    canvas: bin
- show_landmarks:
    point_col: green
    point_size: 20
    label_size: 4
    label_width: 10
- show_mask:
    colour: blue
    line_thickness: 5
export:
- save_results:
    overwrite: true
- save_overlay:
    resize: 0.5
    overwrite: true
"""