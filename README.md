<p align="center">
<img src="https://raw.githubusercontent.com/mluerig/phenopype/master/assets/phenopype_header.png" />
</p>

**phenopype is a phenotyping pipeline for python**. It is designed to extract phenotypic data from digital images or video material with minimal user input in a semi, or fully automated fashion. At the moment it is set up to be run from a python integrated development environment (IDE), like [spyder](https://www.spyder-ide.org/). Some python knowledge is necessary, but most of the heavy lifting is done in the background. If you are interested in using phenopype, [install](#installation) it from the Python Package Index using `pip install phenopype`. You also may want to clone this repository so you can use the [tutorials](#tutorials) to get started.  


***
**DISCLAIMER: ONGOING DEVELOPMENT**

The program is still in alpha stage and development progresses slow - this is [me](https://luerig.net) trying to write a program, while learning to code properly in the first place, next to my everyday work. A few core features like blob-counting, object detection or videotracking are working ([see below](#features)), other modules like landmarking or local object-extraction are not fully implemented yet. More detailed documentation is in the making, but please do get in touch if things are not working as expected and I will try my best to help. 

***


# features

| | |
|:---:|:---:|
|<img src="https://raw.githubusercontent.com/mluerig/phenopype/master/assets/object_detection.gif" width="150%" />|Automatic **object detection** via multistep thresholding in a predefined area. Useful if your images have borders or irregular features. Accurracy can be increased with custom modules, e.g. for colour or shape|
|<img src="https://raw.githubusercontent.com/mluerig/phenopype/master/assets/object_tracking.gif" width="150%" />|Automatic **object tracking** that uses foreground-background subtractor. High performance possible (shown example is close to real time with HD stream). Can be set to distinguish colour or shapes. An example with stickleback and isopods can be found here: https://vimeo.com/283075068|
| <img src="https://raw.githubusercontent.com/mluerig/phenopype/master/assets/scale_detection.gif" width="150%" />|**Automatic scale detection** and pixel-size ratios adjustments. Performance depends on image size| 
| <img src="https://raw.githubusercontent.com/mluerig/phenopype/master/assets/landmarks.gif" width="150%" />|Basic **landmarking** functionality - high throughput.| 
| <img src="https://raw.githubusercontent.com/mluerig/phenopype/master/assets/local_features.gif" width="150%" />|Extract **local objects** like stickleback body armour| 


# installation

1. install python3 with anaconda: go to https://www.anaconda.com/download/, chose python 3.x for your OS, download install it 

2. if you have not done so during the installation, [add "conda" to your PATH](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10) 

3. Install phenopype using `pip` in your terminal or command line:
```
pip install phenopype
```
4. Run the [tutorials](tutorials) with `jupyter notebook`:
```
pip install jupyter notebook
jupyter notebook
```

If you are having difficulties refer to these tutorials:

   - https://conda.io/docs/user-guide/install/windows.html
   - https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444
   - https://datatofish.com/install-package-python-using-pip/

In windows, run everything with administrator privileges! 

# tutorials

Download and unpack this repository, open a command line /bash terminal, and cd to the example folder inside the repo. Assuming you have phenopype, it's dependencies and jupyter notebook installed (comes with scientific python distributions like Anaconda, see [above](#installation)), type `jupyter notebook` and open one of the [tutorials](tutorials):  

* [0_python_intro.ipynb](tutorials/0_python_intro.ipynb) This tutorial is meant to provide a very short overview of the python code needed for basic phenopype workflow. This is useful if you have never used python before, but would like to be able to explore phenopype functionality on your own.

* [1_basic_functions.ipynb](tutorials/1_basic_workflow.ipynb) This tutorial demonstrates basic workflow with phenopype: the creation of a project, directories and how to use the functions alone and within a programmed loop.

* [2_object_detection.ipynb](tutorials/2_object_detection.ipynb) This tutorial demonstrates how single or multiple objects can be detected and phenotyped in images. 

* [3_landmarks_and_local_features.ipynb](tutorials/3_landmarks_and_local_features.ipynb)


# development

Planned featues include

- hdf5-implementation (original image > processed image (+ data) > image for ML-training-dataset >> hdf5)
- build your own training data for deep learning algorithms using hdf5 framework
- add Mask R-CNN deep learning algorithm using the opencv implementation (https://github.com/opencv/opencv/tree/master/samples/dnn) 

If you have ideas for other functionality, let me know!

