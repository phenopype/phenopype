config1 = """
preprocessing:
- measure_scale
- create_mask
segmentation:
- blur:
    kernel_size: 10 
- threshold:
    method: adaptive
    blocksize: 99
    constant: 5
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 10
- find_contours:
    retrieval: ext
- draw_contours:
    thickness: 2
    colour: black
extraction:
- colours:
    channels: [gray, rgb, hsv]
postprocessing:
- save_overlay
- save_csv
"""


config2 = """
preprocessing:
- create_mask:
    label: mask1
    
segmentation:
- blur:
    kernel_size: 10
- threshold:
    method: adaptive
    blocksize: 99
    constant: 5
- morphology:
    operation: close
    shape: ellipse
    kernel_size: 3
    iterations: 10
- find_contours:
    retrieval: ext
- draw_contours:
    thickness: 2
    colour: black
- show_mask:
    colour: green
    line_thickness: 5
    
extraction:
- colours:
    channels: [gray, rgb, hsv]
    
postprocessing:
- save_overlay
- save_csv
"""

config1_alt = """
preprocessing:
    create_mask:
segmentation:
    blur:
        kernel_size: 10
    blur:
        kernel_size: 20
    threshold:
        method: adaptive
        blocksize: 99
        constant: 5
    morphology:
        operation: close
        shape: ellipse
        kernel_size: 3
        iterations: 10
    find_contours:
        retrieval: ext
    draw_contours:
        thickness: 2
        colour: black
extraction:
postprocessing:
"""