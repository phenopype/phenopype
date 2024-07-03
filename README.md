<div align="center">
<img src="https://github.com/phenopype/phenopype/raw/main/assets/phenopype_logo.png" width="300">
</div><br>

| | |
| --- | --- | 
| About | [![pyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-review/issues/24)  [![Author](https://img.shields.io/badge/Author-Moritz_L%C3%BCrig-red)](https://luerig.net) [![License](https://img.shields.io/badge/License-LGPL-yellow)](https://opensource.org/licenses/LGPL-3.0)|
| Testing | [![Coverage Status](https://coveralls.io/repos/github/phenopype/phenopype/badge.svg?branch=main)](https://coveralls.io/github/phenopype/phenopype?branch=main) [![Windows](https://github.com/phenopype/phenopype/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/phenopype/phenopype/actions/workflows/ci-windows.yml) [![Ubuntu](https://github.com/phenopype/phenopype/actions/workflows/ci-ubuntu.yml/badge.svg)](https://github.com/phenopype/phenopype/actions/workflows/ci-ubuntu.yml) |
| Docs | [![Website](https://img.shields.io/badge/phenopype.org-Docs-blue)](https://phenopype.org/docs) [![Website](https://img.shields.io/badge/phenopype.org-Vignettes-blue)](https://phenopype.org/gallery) |

# phenopype

phenopype is a Python package for rapid extraction of phenotypic information from standardized images. It comes with a basic toolset for preprocessing and signal-processing-based segmentation, but also leverages state of the art AI segmentation models like [Fast Segment Anything](https://github.com/CASIA-IVA-Lab/FastSAM) through a growing [set of plugins](https://github.com/phenopype/phenopype-plugins). Additionally, phenopype includes functionality for visualization of image processing results and export into various file and training data formats (json, csv, coco, ...).

phenopype is aiming to augment, rather than replace the utility of existing computer vision libraries. Put differently, phenopype does not intend to be an exhaustive library of granular image processing functions, like OpenCV, scikit-image or ImageJ, but instead provides a set of wrappers and convenient management tools to allow users to *get their data fast* without having to fiddle with too much code. As such, phenopype may also serve as a stepping stone for ecologists and evolutionary biologists who are interested in implementing computer vision workflows. 

<div align="center">
<img src="https://github.com/phenopype/phenopype/raw/main/assets/phenopype_features.png" width="500">
</div><br>

## Quickstart

https://www.phenopype.org/docs/quickstart/

## Main features

(For a complete list [check the API reference](https://www.phenopype.org/docs/api/))

- rapid manual image labelling (~1s per image)
- image analysis workflow:
  - preprocessing (automatic reference-card and QR-code detection, colour and size correction, morphology operations)
  - segmentation (thresholding, watershed, contour-filtering, foreground-background subtraction)
  - measurement (pixel intensities, landmarks, shape features, texture features)
  - visualization (various options) and export (csv, json, ROI images, ...)
  - video analysis module for object tracking
- project management tools to organize images and data (automatic creation of project directory tree)
- customizable analysis-templates that allow anyone to reproduce all collected data with only a few lines of code (suitable for repositories like Dryad or OSF).

![](https://github.com/mluerig/phenopype/raw/master/source/phenopype_demo.gif)


## Contributions and feedback

phenopype development is ongoing and contributions towards making it more broadly applicable and user-friendly are more than  welcome. This can be done by submitting issues or feature requests in the [issue tracker](https://github.com/phenopype/phenopype/issues). You can also [get in touch with me](https://www.luerig.net) directly if you would like to contribute.

## How to cite phenopype

Lürig, M. D. (2021). phenopype : A phenotyping pipeline for Python. Methods in Ecology and Evolution. https://doi.org/10.1111/2041-210x.13771
	
	@ARTICLE{Lurig2022-pb,
	  title     = "phenopype : A phenotyping pipeline for Python",
	  author    = "L{\"u}rig, Moritz D",
	  journal   = "Methods in ecology and evolution / British Ecological Society",
	  publisher = "Wiley",
	  volume    =  13,
	  number    =  3,
	  pages     = "569--576",
	  month     =  mar,
	  year      =  2022,
	  copyright = "http://creativecommons.org/licenses/by-nc/4.0/",
	  language  = "en",
	  issn      = "2041-210X",
	  doi       = "10.1111/2041-210x.13771"
	}

