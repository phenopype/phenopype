![](https://raw.githubusercontent.com/mluerig/phenopype/master/source/phenopype_logo.png)

---

# phenopype: a phenotyping pipeline for Python

| Project status | Windows build | Linux build | OSX build | Coverage | Style |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![Project Status: Active.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) | [![Build status](https://ci.appveyor.com/api/projects/status/4o27rpjbe8ij2kj3?svg=true)](https://ci.appveyor.com/project/mluerig/phenopype) | [![Build Status](https://travis-ci.org/mluerig/phenopype.svg?branch=master)](https://travis-ci.org/mluerig/phenopype) | *none* | [![Coverage Status](https://coveralls.io/repos/github/mluerig/phenopype/badge.svg?branch=master)](https://coveralls.io/github/mluerig/phenopype?branch=master) | [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |  

**Author:** [Moritz Lürig](https://luerig.net)  
**License:** [LGPL](https://opensource.org/licenses/LGPL-3.0)  

## What is phenopype?
phenopype is a high throughput phenotyping pipeline for Python to support ecologists and evolutionary biologists in extracting high dimensional phenotypic data from digital images. phenopype integrates computer vision functions (using [opencv-python](https://github.com/opencv/opencv-python) as the main backbone) and project management tools to facilitate rapid data collection and reproducibility.

![](source/images/phenopype_demo.gif)

#### Features

- basic computer vision functions for:
  - preprocessing (automatic reference detection, colour and size correction, morphology operations)
  - segmentation (thresholding, watershed)
  - data extraction (pixel intensities, shape features, texture features)
- basic project management routines that can automatically organize image data and create customizable analysis-templates.
- After finishing a project, users can share or archive the project structure so that anyone can reproduce all collected data with only a few lines of code (suitable for repositories like DRYAD or OSF).

#### Why phenopype
phenopype is aiming to augment, rather than replace, the utility of existing CV libraries for scientists measuring phenotypes. Put differently, phenopype is not a library of granular image processing functions, like OpenCV or scikit-image, but instead, it is a set of wrappers for these libraries, combined with project management and a focus on *collecting data fast*.


---

## Getting started
1.  [Install Phenopype](https://mluerig.github.io/phenopype/installation.html) - via the _Python Package Index_ (PYPI): `pip install phenopype`
2.  [Run the Tutorials](https://mluerig.github.io/phenopype/tutorial_0.html) - Tutorial 1 is for Python beginners, otherwise Tutorial 2 is a good starting point
3.  [Check the Examples](https://mluerig.github.io/phenopype/index.html#examples) - Example 1 delineates a typical computer vision workflow

#### Documentation, code-reference and tutorials
Detailed installation instructions, along with further resources regarding Python and the OpenCV backbone, as well as the full code reference and a (growing) number of tutorials and feature demonstrations can be found under https://mluerig.github.io/phenopype/.


---

## Contributions and feedback welcome!
Phenopype development is an ongoing process and contribution towards making it a more broadly applicable tool is most welcome. This can be in the form of feature requests (e.g. more functions from the [OpenCV library](https://docs.opencv.org/master/modules.html)) or by reporting bugs via the [issue tracker](https://github.com/mluerig/phenopype/issues). You can also [get in touch with me](https://luerig.net) directly if you have any suggestions for improvement or want to help me making Phenopype better!
