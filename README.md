
![](https://github.com/phenopype/phenopype/raw/main/assets/phenopype_logo_text.png)

| Code review | Windows | Linux | OSX | Code coverage | Code Style |
|:---:|:---:|:---:|:---:|:---:|:---:|
|[![pyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-review/issues/24) | [![Windows Server 2022](https://github.com/phenopype/phenopype/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/phenopype/phenopype/actions/workflows/ci-windows.yml) | [![CI Ubuntu 22.04](https://github.com/phenopype/phenopype/actions/workflows/ci-ubuntu.yml/badge.svg)](https://github.com/phenopype/phenopype/actions/workflows/ci-ubuntu.yml) | *none* | [![Coverage Status](https://coveralls.io/repos/github/phenopype/phenopype/badge.svg?branch=main)](https://coveralls.io/github/phenopype/phenopype?branch=main) | [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |  



**Author:** [Moritz Lürig](https://luerig.net)  
**License:** [LGPL](https://opensource.org/licenses/LGPL-3.0)  
**Homepage** [www.phenopype.org](https://www.phenopype.org)  
**Publication** https://doi.org/10.1111/2041-210x.13771

---

#### What is phenopype?

phenopype is a Python package to rapidly extract high dimensional phenotypic data from digital images. At the core, phenopype provides a project management ecosystem for large image datasets, signal processing based segmentation and data extraction routines, and the possibility to export the processed data in various formats - e.g. for direct analysis (`csv` - e.g. landmarks, coordinates, etc.), or for further processing in other machine learning pipelines (`json` - e.g. segmentation masks).  

#### Why phenopype
phenopype is aiming to augment, rather than replace the utility of existing CV low level libraries for scientists who want to extract phenotypic data from images. Put differently, phenopype does not intend to be an exhaustive library of granular image processing functions, like OpenCV, scikit-image or ImageJ, but instead, it is a set of wrappers and convenient management tools to allow biologists to *get their data fast* without having to fiddle with too much code.

#### Who uses phenopype?
phenopype is intended for ecologists and evolutionary biologists with "laboratory grade" images, which, for example, contain specimens that have been photographed against a standardized background. It is also useful for anyone interested in building a robust training dataset for deep learning models: with phenopype segmentation masks can be created semi-automatically (instead of annotating the images by hand), and, in the process, phenotypic data can already by extracted and evaluated. 

![](https://github.com/phenopype/phenopype/raw/main/assets/phenopype_features.png)

---

#### Main features

(For a complete list [check the API reference](https://www.phenopype.org/docs/api/))

- image analysis workflow:
  - preprocessing (automatic reference detection, colour and size correction, morphology operations)
  - segmentation (thresholding, watershed, contour-filtering, foreground-background subtraction)
  - measurement (pixel intensities, landmarks, shape features, texture features)
  - visualization (various options) and export (csv, json, ROI images, ...)
  - video analysis module for object tracking
- project management tools to organize images and data (automatic creation of project directory tree)
- customizable analysis-templates that allow anyone to reproduce all collected data with only a few lines of code (suitable for repositories like Dryad or OSF).

![](https://github.com/mluerig/phenopype/raw/master/source/phenopype_demo.gif)

---

#### Quickstart

https://www.phenopype.org/docs/quickstart/

#### Documentation

https://www.phenopype.org/docs/

#### Vignette gallery

https://www.phenopype.org/gallery/

#### Contributions and feedback
phenopype development is ongoing and contributions towards making it more broadly applicable and user-friendly are most welcome. This can be in the form of feature requests (e.g. more functions from the [OpenCV library](https://docs.opencv.org/master/modules.html)) or by reporting bugs via the [issue tracker](https://github.com/phenopype/phenopype/issues). You can also [get in touch with me](https://www.luerig.net) directly if you would like to contribute code - in that case, please have a look at the [API](https://www.phenopype.org/docs/api/).

#### How to cite phenopype

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

