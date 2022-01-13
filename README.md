![](https://github.com/phenopype/phenopype/raw/main/assets/phenopype_logo.png)

| Project status | Windows build | Linux build | OSX build | Coverage | Style |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) | [![Build status](https://ci.appveyor.com/api/projects/status/20ncgfq137mmvbgb?svg=true)](https://ci.appveyor.com/project/phenopype/phenopype) | *soon to come* | *none* | [![Coverage Status](https://coveralls.io/repos/github/phenopype/phenopype/badge.svg?branch=main)](https://coveralls.io/github/phenopype/phenopype?branch=main) | [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |  

**Author:** [Moritz Lürig](https://luerig.net)  
**License:** [LGPL](https://opensource.org/licenses/LGPL-3.0)  

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

#### Getting started

1.  Read the [Installation Instructions](https://mluerig.github.io/phenopype/installation.html)

#### Important information

- phenopype currently **does not work on macOS**, as there are several issues with OpenCV's HighGUI module, as documented [here](https://github.com/mluerig/phenopype/issues/9) and [here](https://github.com/mluerig/phenopype/issues/5) - any help in making phenopype usable for macOS is most welcome !
- phenopype is currently undergoing [review at pyOpenSci](https://github.com/pyOpenSci/software-review/issues/24). In the process, phenopype was updated to version 2 which is **not compatible with previous versions** - read the instructions for [installing past versions](https://mluerig.github.io/phenopype/installation.html#installing-past-versions)

---

#### Documentation

The full Documentation can be found here: **https://mluerig.github.io/phenopype/**

#### Contributions and feedback
Phenopype development is an ongoing process and contributions towards making it a more broadly applicable and user-friendly tool are most welcome. This can be in the form of feature requests (e.g. more functions from the [OpenCV library](https://docs.opencv.org/master/modules.html)) or by reporting bugs via the [issue tracker](https://github.com/mluerig/phenopype/issues). You can also [get in touch with me](https://luerig.net) directly if you have any suggestions for improvement.

#### How to cite phenopype
phenopype: a phenotyping pipeline for python (v2.0.0). 2021 Lürig, M. https://github.com/mluerig/phenopype

    @misc{phenopype,
      title={{phenopype: a phenotyping pipeline for Python}},
      author={L{\"u}rig, Moritz},
      year={2021},
      url={https://github.com/mluerig/phenopype},
    }
