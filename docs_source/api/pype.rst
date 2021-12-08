Pype class
----------

.. raw:: html

	Method used to process image datasets with high throughput and reproducibility. Should be used in conjunction with image organized in a Phenopype directory structure. 	The ``pype`` function has specific implicit behaviour that aims at supporting speed and robustness when working in "production" (i.e. when performing the actual analysis of large image datasets compared to	prototyping and low throughput workflow). Here I list some important aspects of that behaviour.

.. important::

	**Enhanced Window control**

	In addition to regular window control functions documented in `(Tutorial 2) <https://mluerig.github.io/phenopype/tutorial_2_phenopype_images.html#Window-control>`__:

	-  Editing and saving the opened configuration file in the text editor will trigger another iteration, i.e. close the image window and run the config file again.
	-  Closing the image window manually (with the X button in the upper right), also runs triggers another run.
	-  ``Esc`` will close all windows and interrupt the pype routine (triggers ``sys.exit()``, which will also end a Python session if run from the command line), as well as any loops.
	-  Each step that requires user interaction (e.g. ``create_mask`` or ``landmarks``) needs to be confirmed with ``Enter`` until the next function in the sequence is executed.
	-  At the end of the analysis, when the final steps (visualization and export functions) have run, use  ``Ctrl+Enter`` to finish and close the window.

	**Function execution**

	Most important things to keep in mind during a ``pype`` iteration:

	-  The ``pype`` function will automatically load the image and execute all functions in sequence, but it will not overwrite overwrite data from past iterations on disk unless specified.
	-  To overwrite interactive user input, set the argument ``overwrite: true`` at the specific function in the configuration file. **Remember to remove it after the next run.**.
	-  If you forget to remove an overwrite argument and are prompted to overwrite previous input, don't do anything, remove the ``overwrite: true`` argument, and save the config file.
	-  If a ``pype`` is initialized on a project directory it will attempt to load input data (e.g. masks) that contain the provided ``name`` argument. For example,``pp.pype(image, name="run1", dirpath="path\to\directory)`` will attempt to load any saved files in ``directory`` that contains the suffix ``"run1"`` (e.g. ``"masks_run1.csv"``).

	**Visualizing the results**

	Aspects of visual feedback during a ``pype`` run (can be completely suppressed by setting ``feedback=False`` in the ``pype`` arguments):

	-  Visual feedback (i.e. output from ``landmarks``, ``find_contours`` or ``create_mask``) is can be visualized on top of a "canvas" (a copy of the original image).
	-  Use ``- select_canvas`` to draw the results either on the raw image, a  binary image, or a single colour channel (gray, red, green or blue).
	-  If ``- select_canvas`` is not explicitly specified, it is called automatically and defaults to the raw image as canvas.
	-  Output from all functions, **needs to be specified manually**. For example, after using ``- landmarks``, ``- draw_landmarks`` should be called in the ``visualization`` module.
	-  Visual parameters of interactive tools (e.g. ``point_size`` or ``line_thickness``) are specified separately in the respective function, *and* in the ``visualization`` module.

	**Exporting the results**

	Saving results and canvas:

	-  All results are saved automatically, even if the respective functions in ``export`` are not specified, with the ``name`` argument in ``pype`` as suffix.
	-  If a file already exist in the directory, and the respective function is *not* listed under ``export:``, then it *will not* be overwritten.
	-  If an export function *is* specified under ``export:``, it *will also not overwrite* any existing file, unless specified using ``overwrite: true``.
	-  The canvas is an exception: it will always be saved and always be overwritten to show the output from the last iteration. However, users can modify the canvas name with ``name`` in the arguments to save different output side by side. For example, ``name: binary`` under ``- save_canvas:`` save the canvas as 	``canvas_binary.jpg``\


.. autoclass:: phenopype.main.Pype
	:members:
	:undoc-members:
	:show-inheritance: