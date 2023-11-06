Installing phenopype
####################

Initial installation
====================

Activate your virtual environment:

.. code-block:: bash

	mamba activate <NAME>  			   	# <NAME> == chosen name, e.g. "pp-env"	


Install phenopype to your environment using :code:`pip` (`pip` is the standard package manager for Python):

.. code-block:: bash

	pip install phenopype

That's it - happy `phenopyping`! You can now use phenopype, e.g. by typing:

.. code-block:: bash

	python
	import phenopype as pp
	

You can also use phenopype interactively (recommended), e.g. using a code editor like Spyder or vscode, or from a `jupyter notebook` - see the tutorials-section for more details. 

.. warning:: 

	Always remember to activate your environment before running phenopype from Python, Spyder or a jupyter notebook.


Installing updates
==================

For regular major and minor releases, use the :code:`-U` flag with :code:`pip`:

.. code-block:: bash

	pip install phenopype -U
	
	
Installing from dev branch
==========================

You can install phenopype directly from the latest commit developmental branch to test experimental features and new implementations. 

.. warning::
   This is generally not recommended and should only be done if you know what you're doing, or if you have been contacted by the phenopype developers.

.. code-block:: bash

	pip install https://github.com/phenopype/phenopype/archive/dev.zip
	

Installing past versions
========================

Major releases are not backwards compatible, so if you have existing phenopype projects that were created with a previous version you need to download that specific version. You can tell `pip` to do so, for example, for version `1.0.0`:

.. code-block:: bash

		pip install "phenopype==1.0.0"

Or, for the latest phenopype version that is still 1.x.x:

.. code-block:: bash

		pip install "phenopype < 2"
		
		
		
		
		


