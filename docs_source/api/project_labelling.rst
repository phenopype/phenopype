Project_labelling class
-----------------------

Phenopype labelling projects are compact versions of regular phenopype projects, suitable for rapid labelling of images using a key-map. After creating a labelling project, the data folder contains images, labels, and backup-files; the export folder contains the exported csv-files.

.. code-block:: python

	import phenopype as pp

	proj = pp.Project_labelling(root_dir=r"my_labelling_project")
	proj.add_files(r"my-data", recursive=True)
	proj.run(tag="v1", config_path=r"templates\template_labelling1.yaml", label_position=(0.1,0.1), skip=False)
	proj.export(tag="v1", overwrite=True, category_column=True)

	## after closing Python, the project can be accessed by simply using Project class again:
	proj = pp.Project_labelling(root_dir=r"labelling_test")

Above code creates a folder structure as follows:

.. code-block:: bash

	my_labelling_project\
		data\ 
			images.json
			v1_labels.json
		export\
			v1_labels.csv
	my-data
	scripts
	templates\
		template_labelling1.yaml
	manuscript
	figures
	...

.. autoclass:: phenopype.main.Project_labelling
	:members:
	:undoc-members:
	:show-inheritance: