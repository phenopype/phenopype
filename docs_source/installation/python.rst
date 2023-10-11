Installing and managing Python 
##############################

Python can be installed in many ways or may even already be installed on your machine (e.g., on some Unix distros). However, to avoid conflicts between package dependencies, and for a cleaner and more reproducible workflow, phenopype should *always* be installed inside a Python virtual environment that you create first. To do so, use a package and environment management system like `conda <https://docs.conda.io/en/latest/>`_. 

Install conda/mamba through miniforge
=====================================

Download and install miniforge to create virtual environments using mamba (like conda but much faster), which is a scientific Python distribution that comes with some packages already built in. `Download the miniforge3 installer <https://github.com/conda-forge/miniforge#miniforge3>`_ for your os and follow installation instructions. Then, open the Miniforge prompt and type:

.. code-block:: bash

	mamba init

If you get an error, refer to the references below for troubleshooting. 

.. admonition:: Troubleshooting references
	:class: note
	
	Consult these references if you have trouble installing Miniconda (they are discussing Anaconda, but the same applies for Miniconda):

	- https://docs.anaconda.com/anaconda/install/
	- https://docs.anaconda.com/anaconda/user-guide/troubleshooting/
	- https://stackoverflow.com/questions/28612500/why-anaconda-does-not-recognize-conda-command
	- https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10
	- https://askubuntu.com/questions/908827/variable-path-issue-conda-command-not-found


Create a virtual environment with mamba
=======================================

Use mamba to create a new Python virtual environment (needs to be Python 3.7 for phenopype):

.. code-block:: bash

	mamba create -n <NAME> python=3.7  # <NAME> == chosen name, e.g. "pp-env"	
	mamba activate <NAME>  			   # activates the new environment 


After successful installation and activation, you should see the name of the environment in the console - e.g.:

.. code-block:: bash

	(pp-env) D:\projects>

Now all libraries installed into this environment will be isolated from those installed in other virtual environments. You can now move on to install phenopype.

