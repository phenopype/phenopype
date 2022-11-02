Installation
############

For Python beginners
====================

Follow these instructions step by step:

.. toctree::
	:numbered:
	:maxdepth: 1

	python
	spyder
	phenopype
	text

For advanced users
==================

Create a new Python (3.7) environment with :code:`conda` or :code:`mamba`:

.. code-block:: bash

	mamba create -n <NAME> python=3.7  # <NAME> == chosen name, e.g. "pp-env"	


Install phenopype with :code:`pip install phenopype`. 

.. code-block:: bash

	conda activate <NAME>  			   
	pip install phenopype

To work interactively, I strongly recommend to use `Spyder <https://docs.spyder-ide.org/current/index.html>`_ (best installed via conda-forge): 

.. code-block:: bash

	mamba install spyder -c conda-forge

