
![](https://github.com/phenopype/phenopype/raw/main/assets/phenopype_logo.png)

---

## **phenopype**
### **A phenotyping pipeline for python**

| Project status | Windows build | Linux build | OSX build | Coverage | Style |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) | [![Build status](https://ci.appveyor.com/api/projects/status/20ncgfq137mmvbgb?svg=true)](https://ci.appveyor.com/project/phenopype/phenopype) | *soon to come* | *none* | [![Coverage Status](https://coveralls.io/repos/github/phenopype/phenopype/badge.svg?branch=main)](https://coveralls.io/github/phenopype/phenopype?branch=main) | [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |  

**Author:** [Moritz Lürig](https://luerig.net)  
**License:** [LGPL](https://opensource.org/licenses/LGPL-3.0)  
**Homepage** [www.phenopype.org](https://www.phenopype.org)  
**Publication** https://doi.org/10.1111/2041-210x.13771

---

#### What is phenopype?

phenopype is a high throughput phenotyping pipeline for Python to support ecologists and evolutionary biologists in extracting high dimensional phenotypic data from digital images. The package aims at facilitating rapid, signal processing based analysis of biological images containing phenotypic information.

#### Why phenopype
phenopype is aiming to augment, rather than replace the utility of existing CV libraries for scientists measuring phenotypes. Put differently, phenopype does not intend to be an exhaustive library of granular image processing functions, like OpenCV, scikit-image or ImageJ, but instead, it is a set of wrappers and convenient management tools to allow biologists to *get their data fast* without having to fiddle with too much code.

#### Main features

(For a complete list [check the API reference](https://mluerig.github.io/phenopype/api.html))

- image analysis workflow:
  - preprocessing (automatic reference detection, colour and size correction, morphology operations)
  - segmentation (thresholding, watershed, contour-filtering, foreground-background subtraction)
  - measurement (pixel intensities, landmarks, shape features, texture features)
  - visualization and export   
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

	@ARTICLE{Lurig2021,
	  title     = "phenopype : A phenotyping pipeline for Python",
	  author    = "L{\"u}rig, Moritz D",
	  journal   = "Methods in Ecology and Evolution",
	  publisher = "Wiley",
	  month     =  dec,
	  year      =  2021,
	  copyright = "http://creativecommons.org/licenses/by-nc/4.0/",
	  language  = "en",
	  doi       = "10.1111/2041-210x.13771"
	}
