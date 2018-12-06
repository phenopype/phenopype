# phenopype

phenopype is a phenotyping pipeline for python. It is designed to extract phenotypic data from digital images or video material with minimal user input, and therefore aims at large datasets. It does not have a GUI, but instead is designed to be run from a python integrated development environment (IDE), like [spyder](https://www.spyder-ide.org/), [pycharm](https://www.jetbrains.com/pycharm/), or [VScode](https://code.visualstudio.com/), providing an "RStudio"-like feeling. phenopype is developed by [me](https://luerig.net) and builds on a collection of scripts ([iso_cv](https://github.com/mluerig/iso_cv), [iso_track](https://github.com/mluerig/iso_track)) that I wrote for my PhD at [Eawag](https://www.eawag.ch/en/department/eco/) and [ETH Zürich](http://www.adaptation.ethz.ch/).

phenopype is currently alpha, and still under heavy construction. A few core features like object detection or videotracking work ([see below](#features)), more are [planned](#development). If you are interested in using phenopype, check the installation guide and [contact me](https://www.eawag.ch/en/aboutus/portrait/organisation/staff/profile/moritz-luerig/show/) directly or raise an issue if you encounter problems. Some python knowledge is necessary, but not much since all the heavy lifting is done in the background. Feel free to get in touch of you have requests or ideas for further use cases (e.g. for your own study system) that you think could be solved with phenopype. 

# features

|Object detection|Object tracking| Scale detection|
|:--:|:--:|:--:|
|<img src="assets/doc/object_detection.gif" width="90%" />|<img src="assets/doc/object_tracking.gif" width="80%" />| <img src="assets/doc/scale_detection.gif" width="100%" />|
|Automatic object detection via multistep thresholding in a predefined area. Useful if your images have borders or irregular features. Accurracy can be increased with custom modules, e.g. for colour or shape|Automatic object tracking that uses foreground-background subtractor. High performance possible (shown example is close to real time with HD stream). Can be set to distinguish colour or shapes.|A scale that is once identified and measured can be found in all following pictures. Automatically corrects pixel-size ratios. Performance depends on image size| 
|<img src="assets/doc/object_detection.JPG" width="80%" />|<img src="assets/doc/object_tracking.png" width="80%" />| <img src="assets/doc/scale_detection.png" width="100%" />|
|||| 

# installation

(tested for `Python 3.7` with `opencv 3.4.4`)

## windows

1. install python3 with anaconda: https://www.anaconda.com/download/ chose python 3.x for your OS, download and install 
2. if you have not done so during the installation, [add "conda" to your PATH](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10) 
3. Install phenopype to your local python environment directly from github using `git` (dependencies will be installed on the fly):
```
conda install git
pip install git+https://github.com/mluerig/phenopype
```

## ubuntu 18.04

Under ubuntu I encountered some issues with the opencv-GUI - you can still try (need a working gtk installation).

## Mac OS

Not tested yet

# quickstart

Download the repository to use the provided examples in an IDE. Anaconda distributions come with the [spyder IDE](https://www.spyder-ide.org/), which is a great scientific python environment. Simply run `spyder` from the shell, and, after you have cloned the repository, open `example.py` from the example folder.

JUPYTER NOTEBOOK VERSION COMING SOON


# development

Planned featues include

- hdf5-implementation (original image > processed image (+ data) > image for ML-training-dataset >> hdf5)
- localized feature extraction (e.g. stickleback armour-plates, amount of pigments in leafs,...)
- shape detection (contour of objects)

If you have ideas for other functionality, let me know!

