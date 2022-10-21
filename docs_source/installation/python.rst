Installing Python
-----------------

Installing miniconda
~~~~~~~~~~~~~~~~~~~~

Download and install `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ to create virtual environments using the :code:`conda` manager. A virtual environment is a Python environment such that the Python kernel, all libraries and scripts installed into it are isolated from those installed in other virtual environments. Miniconda is a scientific python distribution which comes with the some scientific Python packages already built in. Follow the  `OS-specific installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. Test if conda was successfully installed with the following

.. code-block:: bash

	conda --version
	
If this doesn't show the conda version, please refer to the references for troubleshooting below. 
	

Installing mamba
~~~~~~~~~~~~~~~~

Next, install the `mamba <https://github.com/mamba-org/mamba>`_ package manager, which replaces :code:`conda` and is `much` faster. For detailed installation instructions and user guide refer to the `documentation <https://mamba.readthedocs.io/en/latest/#>`_. In short, do the following:

.. code-block:: bash

	conda install -c conda-forge mamba
	
and test the installation with 

.. code-block:: bash

	mamba --version
	
For instance, to create a new Python 3.7 environment and install spyder:

.. code-block:: bash

	mamba create -n new_env python=3.7
	conda activate new_env  				## still need 'conda' to activate
	mamba install -c conda-forge spyder


Troubleshooting references
~~~~~~~~~~~~~~~~~~~~~~~~~~

Consult these references if you have trouble installing Miniconda (they are discussing Anaconda, but the same applies for Miniconda):

- https://docs.anaconda.com/anaconda/install/
- https://docs.anaconda.com/anaconda/user-guide/troubleshooting/
- https://stackoverflow.com/questions/28612500/why-anaconda-does-not-recognize-conda-command
- https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10
- https://askubuntu.com/questions/908827/variable-path-issue-conda-command-not-found


 
