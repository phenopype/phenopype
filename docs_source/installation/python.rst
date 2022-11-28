Installing Python with Anaconda
###############################

Python can be installed in many ways or may even already be installed on your machine (e.g., on some Unix distros). However, since phenopype is a scientific Python package, only the installation procedure using Anaconda, a scientific Python distribution, is described here. Anaconda, more specifically, its terminal interface :code:`conda`, is both a Python package and environment manager. To avoid conflicts between package dependencies, and for a cleaner and more reproducible workflow, phenopype should *always* be installed inside a Python virtual environment that you create first `(read about virtual envs here) <https://docs.python.org/3/tutorial/venv.html>`_. This procedure is explained here. 

Installing :code:`conda`
========================

Download and install `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ to create virtual environments using the :code:`conda` manager. Miniconda is a scientific Python distribution that comes with some packages already built in. Follow the  `OS-specific installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. WINDOWS USERS: install miniconda directly to the top level of your drive so you don't run into privilege issues - e.g. :code:`C:\miniconda3`. Test if conda was successfully installed:

.. code-block:: bash

	conda --version

If this doesn't show the current conda version, please refer to the references below for troubleshooting. 

.. admonition:: Troubleshooting references
	:class: note
	
	Consult these references if you have trouble installing Miniconda (they are discussing Anaconda, but the same applies for Miniconda):

	- https://docs.anaconda.com/anaconda/install/
	- https://docs.anaconda.com/anaconda/user-guide/troubleshooting/
	- https://stackoverflow.com/questions/28612500/why-anaconda-does-not-recognize-conda-command
	- https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10
	- https://askubuntu.com/questions/908827/variable-path-issue-conda-command-not-found

Installing :code:`mamba` (optional)
===================================

This step is optional, but highly recommended! The `mamba <https://github.com/mamba-org/mamba>`_ package manager replaces :code:`conda` and is `much` faster. For detailed installation instructions and user guide refer to the `documentation <https://mamba.readthedocs.io/en/latest/#>`_. In short, do the following:

.. code-block:: bash

	conda install -c conda-forge mamba
	
Test the installation with 

.. code-block:: bash

	mamba --version
	

Creating a virtual environment with :code:`conda` (or :code:`mamba`)
====================================================================

.. note:: If you have installed mamba, use :code:`mamba` instead of :code:`conda` here (except when activating and environment: there you still need to use :code:`conda activate`).

Use :code:`conda` to create a new Python virtual environment (needs to be Python 3.7 for phenopype):

.. code-block:: bash

	conda create -n <NAME> python=3.7  # <NAME> == chosen name, e.g. "pp-env"	
	conda activate <NAME>  			   # activates the new environment 


After successful installation and activation, you should see the name of the environment in the console - e.g.:

.. code-block:: bash

	(pp-env) D:\projects>

Now all libraries installed into this environment will be isolated from those installed in other virtual environments. 
