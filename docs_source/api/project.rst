Project class
-------------

Phenopype projects are composed of a directory tree in which each folder contains the copy or a link to a single raw image file. Each raw should have only one folder, so that result or intermediate output from different iterations of processing and analysis are stored side by side. The project root folder should be separate from the raw data, e.g. as a folder inside of your project folder:

.. code-block:: python

	import phenopype as pp

	myproj = pp.Project(root_dir="\my_project\phenopype")
	myproj.add_files(image_dir="\my_project\data_raw")
	myproj.add_config(name = "v1")
	pp.project.save(myproj)

	## after closing Python, the project can be accessed by simply using Project class again:
	myproj = pp.Project(root_dir="\my_project\phenopype")

Above code creates a folder structure as follows:

.. code-block:: bash

	my_project\
	 data_raw
	 phenopype\		# create phenopype project here with "pp.Project"
	  data\ 		# created automatically
	   file1\		# add files to project using "pp.Project.add_files"
	    raw.jpg			# created by "pp.Project.add_files"
	    attributes.yaml	# created by "pp.Project.add_files"
	    pype_config_v1.yaml	# added with "pp.Project.add_config"
	    results_v1.csv
	   file2\
	    ...
	   ...
	 data
	 scripts
	 manuscript
	 figures

.. important::

	Currently the only useful information contained in the project object (:code:`myproj`) is a list of all directories inside the project's directory tree. It is important to save `both` the project AND all results in the data directories using the appropriate functions in (:ref:`Export`). Future release will expand the functionality of the project class and its associated methods.

.. autoclass:: phenopype.main.Project
	:members:
	:undoc-members:
	:show-inheritance: