preset1 = """
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
    
extraction:
- colour_values:
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
    
postprocessing:
- save_csv:
    overwrite: true
    
- save_overlay:
    resize: 0.5
    overwrite: true
"""

preset2 = """
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
    
extraction:
- colour_values:
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
    
postprocessing:
- save_csv:
    overwrite: true
    
- save_overlay:
    resize: 0.5
    overwrite: true
"""