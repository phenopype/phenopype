Core modules
------------

Core image processing functions. Any function can be executed in prototyping, low throughput, and high throughput workflow. In general, the typical order in which the different steps are performed is:

**preprocessing > segmentation > measurement > visualization > export**

However, since phenopype version 2 **you can arrange them however you like**. In general, any function can either take an array or a phenopype container. If an array is passed, additional input arguments may be required (e.g. to draw contours onto an image, but an array and a DataFrame containing the contours must be supplied, whereas a container already includes both).


Preprocessing
"""""""""""""

.. automodule:: phenopype.core.preprocessing
	:members:
	:undoc-members:
	:show-inheritance:
	


Segmentation
""""""""""""

.. automodule:: phenopype.core.segmentation
	:members:
	:undoc-members:
	:show-inheritance:



Measurement
"""""""""""

.. automodule:: phenopype.core.measurement
	:members:
	:undoc-members:
	:show-inheritance:



Visualization
"""""""""""""

.. automodule:: phenopype.core.visualization
	:members:
	:undoc-members:
	:show-inheritance:



Export
""""""

.. automodule:: phenopype.core.export
	:members:
	:undoc-members:
	:show-inheritance:
