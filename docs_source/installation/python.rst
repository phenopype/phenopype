Installing Python
#################

Python can be installed in many ways or may even already be installed on your machine (e.g., on some Unix distros). However, to avoid conflicts between package dependencies, and for a cleaner and more reproducible workflow, phenopype should *always* be installed inside a Python virtual environment that you create first. To do so, use a package and environment management system like "mamba" (see below), "pipenv", or "conda". 

Install mamba through miniforge
=====================================

Download and install miniforge to create virtual environments using mamba (it's like conda but *a lot* faster), which is a scientific Python distribution that comes with some packages already built in. `Download the Miniforge3 installer <https://github.com/conda-forge/miniforge#miniforge3>`_ (under "Latest installers with Python 3.10 in the base environment") for your OS and follow the installation instructions. Then, open the Miniforge prompt or terminal, and type:

.. code-block:: bash

	mamba init

If you get an error, refer to the references below for troubleshooting. 

.. admonition:: Troubleshooting references
	:class: note
	
	Consult these references for troubleshooting:

	- https://mamba.readthedocs.io/en/latest/mamba-installation.html
	- https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html


Create a virtual environment with mamba
=======================================

Use mamba to create a new Python virtual environment (needs to be Python 3.9 for phenopype):

.. code-block:: bash

	mamba create -n <NAME> python=3.9  # <NAME> == chosen name, e.g. "pp-env"	
	mamba activate <NAME>  			   # activates the new environment 


After successful installation and activation, you should see the name of the environment in the console - e.g.:

.. code-block:: bash

	(pp-env) D:\projects>

Now all libraries installed into this environment will be isolated from those installed in other virtual environments. You can now move on to install phenopype.

