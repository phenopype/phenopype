Utility functions
-----------------

Utility functions, e.g. to load and display images ,e.g. :func:`pp.utils.load_image` and :func:`pp.utils.show_image`. Any of these functions can be called directly after loading  phenopype without the need to add the :code:`utils` submodule:

.. code-block:: python

	import phenopype as pp

	image_path = "image_dir/my_image.jpg"

	image = pp.load_image(image_path) 	# instead of pp.utils.load_image
	pp.show_image(image) 			# instead of pp.utils.show_image
	dir(pp) 				# shows available classes and functions

.. automodule:: phenopype.utils
	:members:
	:undoc-members:
	:show-inheritance:
	:exclude-members: custom_formatwarning