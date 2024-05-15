API reference
=============

The phenopype API
-----------------
As most Python packages, phenopype has a Application Programming Interface (API) with a "public" and a "private" part (see https://lwn.net/Articles/795019/). The public part is documented below and includes all Python functions, classes and methods that are available for high throughput image analysis, project management, and image import and export. The private part works in the background and includes helper functions and classes that support the public API. They are not intended to be called directly by users. 

.. image:: /_assets/images/luerig_2021_figure1.jpg
   :align: center
   :alt: Schematic of the API for phenopype (3.0.0)

The API reference
-----------------
The API reference presented here is auto-generated from the `docsstrings <https://www.python.org/dev/peps/pep-0257/>`_ in each Python function, class or method, using `Sphinx <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`_. In Python, you can access the content of each docsstring using :code:`help()`. Most IDEs have a shortcut to access the docsstring, e.g. :code:`Ctrl+i` in Spyder or :code:`Ctrl+Q` in PyCharm. 

The :code:`Project` class for comprehensive image annotation, pattern recognition and feature extraction workflows:

.. toctree::

	project
	
The :code:`Pype`-class for high-throughput analysis:

.. toctree::

	pype

The :code:`Project_labelling` class for minimalistic labelling workflows:

.. toctree::

	project_labelling

Utility functions, e.g. for loading, saving	and viewing images:

.. toctree::

	utility
	
All image processing functions:
	
.. toctree::

	core

The video tracking tools:
	
.. toctree::

	video
	













