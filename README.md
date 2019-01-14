<p align="center">
<img src="assets/phenopype_header.png" width="100%" />
</p>

phenopype is a phenotyping pipeline for python. It is designed to extract phenotypic data from digital images or video material with minimal user input. Like other scientific python packages it is designed to be run from a python integrated development environment (IDE), like [spyder](https://www.spyder-ide.org/) or [pycharm](https://www.jetbrains.com/pycharm/). phenopype is developed by [me](https://luerig.net) and builds on a collection of scripts ([iso_cv](https://github.com/mluerig/iso_cv), [iso_track](https://github.com/mluerig/iso_track)) that I wrote for my PhD at [Eawag](https://www.eawag.ch/en/department/eco/) and [ETH Zürich](http://www.adaptation.ethz.ch/).

phenopype is currently alpha, and still under heavy construction. A few core features like blob-counting, object detection or videotracking work ([see below](#features)), more are planned. If you are interested in using phenopype, check the installation guide and quickstart, or [contact me](https://www.eawag.ch/en/aboutus/portrait/organisation/staff/profile/moritz-luerig/show/) directly. Some python knowledge is necessary, but most of the heavy lifting is done in the background. Feel free to get in touch of you have requests or ideas for further use cases (e.g. for your own study system) that you think could be solved with phenopype. 


# features

| | |
|:---:|:---:|
|<img src="assets/object_detection.gif" width="150%" />|Automatic **object detection** via multistep thresholding in a predefined area. Useful if your images have borders or irregular features. Accurracy can be increased with custom modules, e.g. for colour or shape|
|<img src="assets/object_tracking.gif" width="150%" />|Automatic **object tracking** that uses foreground-background subtractor. High performance possible (shown example is close to real time with HD stream). Can be set to distinguish colour or shapes.|
| <img src="assets/scale_detection.gif" width="150%" />|**Automatic scale detection** and pixel-size ratios adjustments. Performance depends on image size| 
| <img src="assets/landmarks.gif" width="150%" />|Basic **landmarking** functionality - high throughput.| 
| <img src="assets/local_features.gif" width="150%" />|Extract **local features** like stickleback body armour or organelles| 



# installation

(tested for windows with `Python 3.7` and `opencv 3.4.4`)

1. install python3 with anaconda: https://www.anaconda.com/download/ chose python 3.x for your OS, download and install 
2. if you have not done so during the installation, [add "conda" to your PATH](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10) 
3. Install phenopype to your local python environment directly from github using `git` (dependencies will be installed on the fly):
```
conda install git
pip install git+https://github.com/mluerig/phenopype
```
If you have difficulties installing opencv, try it manually:

```
pip install opencv-contrib-python
```

# quickstart

Download this repository, open a command line /bash terminal, and cd to the example folder inside the repo. Assuming you have phenopype, it's dependencies and jupyter notebook installed (comes with scientific python distributions like Anaconda, see [above](#installation)), type `jupyter notebook` and open one of the tutorials:  

* [0_python_intro.ipynb](examples/0_python_intro.ipynb) This tutorial is meant to provide a very short overview of the python code needed for basic phenopype workflow. This is useful if you have never used python before, but would like to be able to explore phenopype functionality on your own.

* [1_basic_functions.ipynb](examples/1_basic_functions.ipynb) This tutorial is meant to provide a very short overview of the python code needed for basic phenopype workflow. This is useful if you have never used python before, but would like to be able to explore phenopype functionality on your own.

* [2_object_detection.ipynb](examples/2_object_detection.ipynb) This tutorial is meant to provide a very short overview of the python code needed for basic phenopype workflow. This is useful if you have never used python before, but would like to be able to explore phenopype functionality on your own.

* [3_landmarks_and_local_features.ipynb](examples/3_landmarks_and_local_features.ipynb) This tutorial is meant to provide a very short overview of the python code needed for basic phenopype workflow. This is useful if you have never used python before, but would like to be able to explore phenopype functionality on your own.


# development

Planned featues include

- hdf5-implementation (original image > processed image (+ data) > image for ML-training-dataset >> hdf5)
- build your own training data for deep learning algorithms using hdf5 framework
- add Mask R-CNN deep learning algorithm using the opencv implementation (https://github.com/opencv/opencv/tree/master/samples/dnn) 

If you have ideas for other functionality, let me know!

