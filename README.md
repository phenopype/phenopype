# phenopype

phenopype is a phenotyping pipeline for python. It is designed to extract phenotypic data from digital images or video material with minimal user input, and therefore aims at large datasets. It does not have a GUI, but instead is designed to be run from a python integrated development environment (IDE), like [spyder](https://www.spyder-ide.org/), [pycharm](https://www.jetbrains.com/pycharm/), or [VScode](https://code.visualstudio.com/), providing an "RStudio"-like feeling. phenopype is developed by [me](https://luerig.net) and builds on a collection of scripts ([iso_cv](https://github.com/mluerig/iso_cv), [iso_track](https://github.com/mluerig/iso_track)) that I wrote for my PhD at [Eawag](https://www.eawag.ch/en/department/eco/) and [ETH Zürich](http://www.adaptation.ethz.ch/).

phenopype is currently alpha, and still under heavy construction. A few core features like object detection or videotracking work ([see below](#features)), more are [planned](#development). If you are interested in using phenopype, check the installation guide and [contact me](https://www.eawag.ch/en/aboutus/portrait/organisation/staff/profile/moritz-luerig/show/) directly or raise an issue if you encounter problems. Some python knowledge is necessary, but not much since all the heavy lifting is done in the background. Feel free to get in touch of you have requests or ideas for further use cases (e.g. for your own study system) that you think could be solved with phenopype. 

# features

|Object detection|Object tracking| Scale detection|
|:--:|:--:|:--:|
|<img src="assets/object_detection.gif" width="90%" />|<img src="assets/object_tracking.gif" width="80%" />| <img src="assets/scale_detection.gif" width="100%" />|
|Automatic object detection via multistep thresholding in a predefined area. Useful if your images have borders or irregular features. Accurracy can be increased with custom modules, e.g. for colour or shape|Automatic object tracking that uses foreground-background subtractor. High performance possible (shown example is close to real time with HD stream). Can be set to distinguish colour or shapes.|A scale that is once identified and measured can be found in all following pictures. Automatically corrects pixel-size ratios. Performance depends on image size| 
|<img src="assets/object_detection.JPG" width="80%" />|<img src="assets/object_tracking.png" width="80%" />| <img src="assets/scale_detection.png" width="100%" />|
|||| 

# installation

Currently, only windows 10 is tested and running stable:

## windows

1. install python3 (anaconda is highly recommended: https://www.anaconda.com/download/ chose python 3.x for your OS, download and install 
2. if you have not done so during the installation, [add "conda" to your PATH](https://docs.anaconda.com/anaconda/faq/#should-i-add-anaconda-to-the-windows-path) 
3. open cmd/powershell and enter:
```
conda update conda
conda config --add channels conda-forge 
conda install exifread
conda install -c conda-forge opencv 
conda install -c soft-matter trackpy
```
(if you have trouble installing opencv check [this](https://stackoverflow.com/questions/23119413/how-do-i-install-python-opencv-through-conda))
4. Install phenopype to your local python environment directly from github using `git` and `pip`:
```
conda install git pip
pip install git+https://github.com/mluerig/phenopype
```

## ubuntu 18.04

Under ubuntu I encountered some issues with the opencv-GUI - you can still try yourself (need a working gtk installation and a python 3.6 environment):

1. install python3 (anaconda is highly recommended: https://www.anaconda.com/download/ chose python 3.x for your OS, install, 
2. if you have not done so during the installation add "conda" to your PATH:
```
export PATH="~/anaconda3/bin:$PATH"
```
3. create a python 3.6 environment: 
```
conda create -n p36 python=3.6`
conda activate p36`
```
4. install dependencies:
```
conda update conda
conda install exifread
conda install -c menpo opencv3
conda install -c soft-matter trackpy
```
5. Install phenopype to your local python environment directly from github using `git` and `pip`:
```
conda install git pip
pip install git+https://github.com/mluerig/phenopype
```

# quickstart

Download the repository to use the provided examples in an IDE. Anaconda distributions come with the [spyder IDE](https://www.spyder-ide.org/), which is a great scientific python environment. Simply run `spyder` from the shell, and, after you have cloned the repository, open `example.py` from the example folder.


# development

Planned featues include

- hdf5-implementation (original image > processed image (+ data) > image for ML-training-dataset >> hdf5)
- localized feature extraction (e.g. stickleback armour-plates, amount of pigments in leafs,...)
- shape detection (contour of objects)
- landmark tool (analagours to tpsDig, but faster)

If you have ideas for other functionality, let me know!

