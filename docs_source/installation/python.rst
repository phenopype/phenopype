Installing Python
-----------------

phenopype needs Python 3, and I recommend using `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ to install and maintain Python 3 environments. Miniconda, a minimal version of `Anaconda <https://www.anaconda.com/>`_, is a scientific python distribution which comes with the some scientific Python packages already built in. Follow the OS-specific instructions below:

For Windows
~~~~~~~~~~~

Open the Miniconda installer file you downloaded. Depending on the installation location of your choice you may need admin rights (right click on the file and select `Run as Administrator`), but DON'T install Miniconda to `Program files`, as this may cause problems later. A good idea is typically the root directory of your hard drive (e.g. `C:\\Miniconda3`). Then test if the installation was successful by opening a terminal and typing:

.. code-block:: bash

	conda --version

If :code:`conda` is not recognized you need add the path to your Miniconda installation directory to your environmental variables (if you have not done so during the installation). To do so, go to `Control Panel\\System` and `Security\\System\\Advanced System Settings` and look for `Environment Variables`. Then click `new` and add the path to the Miniconda folder (i.e., the path you selected during installation - e.g. `C:\\Miniconda3`) and the subfolder `scripts` (e.g. `C:\\Miniconda3\\Scripts`.

An alternative to manipulating the environment variables is to use the Anaconda prompt that can be launched from a shortcut in the `Start` menu (should get added during the installation), or through the `Anaconda Navigator <https://docs.anaconda.com/anaconda/user-guide/getting-started/>`_).


For Linux
~~~~~~~~~

Run the Miniconda installer script you downloaded, e.g. :code:`bash ~/Downloads/Miniconda3-2020.02-Linux-x86_64.sh`, and follow the instructions. When the installer prompts “Do you wish the installer to initialize Miniconda3 by running conda init?”, type `yes`. Then test if the installation was successful by opening a terminal and typing:

.. code-block:: bash

	conda --version

If :code:`conda` is not recognized you need to add the path to your Miniconda installation directory to your `.bashrc` file. To do so, type :code:`echo 'export PATH=/path/to/Miniconda3/bin:$PATH' >> ~/.bashrc`.


.. important::

	In some shells, you may have to use :code:`conda init` before the command is recognized.


Using mamba
~~~~~~~~~~~

I recommend using the `mamba <https://github.com/mamba-org/mamba>`_ cross-platform package manager, which is a blazing fast reimplementation of the conda package manager in C++. For detailed installation instructions and user guide refer to the `documentation <https://mamba.readthedocs.io/en/latest/#>`_. In short, do the following:

.. code-block:: bash

	conda install -c conda-forge mamba
	
and test the installation with 

.. code-block:: bash

	mamba --version
	
Note that you STILL have to use :code:`conda activate` to activate an environment you created with :code:`mamba create` - `all other commands remain the same <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html?highlight=activate#mamba-vs-conda-clis>`_:

.. code-block:: bash

	mamba install ...
	mamba create -n ... -c ... ...
	mamba list


Troubleshooting references
~~~~~~~~~~~~~~~~~~~~~~~~~~

Consult these references if you have trouble installing Miniconda (they are discussing Anaconda, but the same applies for Miniconda):

- https://docs.anaconda.com/anaconda/install/
- https://docs.anaconda.com/anaconda/user-guide/troubleshooting/
- https://stackoverflow.com/questions/28612500/why-anaconda-does-not-recognize-conda-command
- https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10
- https://askubuntu.com/questions/908827/variable-path-issue-conda-command-not-found


 