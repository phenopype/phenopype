Project class
-------------

Phenopype projects are composed of a directory tree in which each folder contains the copy or a link to a single raw image file. Each raw should have only one folder, so that result or intermediate output from different iterations of processing and analysis are stored side by side. The project root folder should be separate from the raw data, e.g. as a folder inside of your project folder:

.. code-block:: python

	import phenopype as pp

	myproj = pp.Project(root_dir="my_project")
	myproj.add_files(image_dir="my-data")
	myproj.add_config(tag = "v1", template_path="templates\template1.yaml")
	...

	## after closing Python, the project can be accessed by simply using Project class again:
	myproj = pp.Project(root_dir="\my_project\phenopype")

Above code creates a folder structure as follows:

.. code-block:: bash

	my_project\
		data\ 							# created automatically
			file1\						# add files to project using "pp.Project.add_files"
				raw.jpg					# created by "pp.Project.add_files"
				attributes.yaml			# created by "pp.Project.add_files"
				pype_config_v1.yaml		# added with "pp.Project.add_config"
				results_v1.csv
			file2\
			...
		...
	my-data
	scripts
	templates
	template1.yaml
	manuscript
	figures
	...

.. important::

	Currently the only useful information contained in the project object (:code:`myproj`) is a list of all directories inside the project's directory tree. Always save your progressing using the appropriate functions in (:ref:`Export`). 

.. autoclass:: phenopype.main.Project
	:members:
	:undoc-members:
	:show-inheritance: