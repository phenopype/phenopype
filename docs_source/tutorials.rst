Tutorials
=========

The tutorials are written as jupyter `notebooks` - browser based Python kernels to run, document, and visualize code (`https://jupyter.org/ <https://jupyter.org/>`_). If you installed Python using Anaconda, it is possible that jupyter is already installed (check with :code:`jupyter --version` - you should see something like :code:`notebook : 6.4.11` in the list that appears). If no list appears, your need to install jupyter notebook first.

Instructions
------------

1. Install phenopype` (see `installation instructions <installation.html>`_) and :code:`jupyter notebook` (:code:`pip install jupyter notebook`).
2. `Download <https://github.com/phenopype/phenopype-tutorials/archive/refs/heads/main.zip>`_ and unpack the github archive containing the data and code. 
3. Open a terminal in the unpacked folder - don't forget to activate your conda environment (see warning below).
4. Start the notebooks with :code:`jupyter notebook` and click on one of the tutorial files (your browser might give you a security warning - you can ignore it).
5. Run the code cell by cell inside the browser window (Shift + Enter to run cell and advance).

.. warning::

	Make sure you install jupyter notebook to your specific environment (i.e. activate it first using :code:`conda activate pp`. If not installed in a specific environment, running :code:`jupyter notebook` will fall back on the conda base environment where phenopype may not be installed (this is a common source of confusion).

Tutorials (read-only)
---------------------

Below are the read-only html versions of the code contained in the notebooks - to run them yourself, follow the above instructions. If you want to use the notebooks as a blueprint for your own project, you can also save them as a Python script from a running jupyter notebook using :code:`File > Download as > Python (.py)`.

|

.. grid:: 2
	:gutter: 2

	.. grid-item-card::  Tutorial 1
			:link: tutorial_1/

			A (very) brief python introduction

			- Python modules
			- Paths and directories
			- Images in Python


	.. grid-item-card::   Tutorial 2
			:link: tutorial_2/

			Interacting with images in phenopype

			- Window control
			- Opening images
			- Creating masks


	.. grid-item-card::  Tutorial 3
			:link: tutorial_3/

			Image analysis workflow

			- Overview
			- Low throughput
			- High throughput


	.. grid-item-card:: Tutorial 4
			:link: tutorial_4/

			The Pype class

			- Operation
			- Configuratiob templates
			- YAML-syntax


	.. grid-item-card:: Tutorial 5
			:link: tutorial_5/

			Setting up and managing projects

			- Project directories
			- Adding images and configs
			- Collecting results


	.. grid-item-card:: Tutorial 6
			:link: tutorial_6/

			Creating and detecting a reference

			- Setting project wide size references
			- Detecting size references


	.. grid-item-card:: Tutorial 7
			:link: tutorial_7/

			Video analysis

			- Motion tracker class
			- Tracking methods


.. toctree::
	:hidden:

	tutorials/tutorial_1
	tutorials/tutorial_2
	tutorials/tutorial_3
	tutorials/tutorial_4
	tutorials/tutorial_5
	tutorials/tutorial_6
	tutorials/tutorial_7
