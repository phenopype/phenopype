Pype class
----------

phenopype's core method to process image datasets with high throughput and reproducibility. Should be used in conjunction with image organized in a Phenopype directory structure. The ``Pype`` class has specific implicit behaviour that aims at supporting speed and robustness when working in "production" (i.e. when performing the actual analysis of large image datasets compared to	prototyping and low throughput workflow). These are the most important things to keep in mind during a ``Pype`` iteration

.. important::

	**Enhanced Window control**

	In addition to regular window control functions documented in `Tutorial
	2 <tutorial_2.ipynb#Window-control>`__:

	-  Editing and saving the opened configuration file in the text editor
	   will trigger another iteration, i.e. close the image window and run
	   the config file again.
	-  Closing the image window manually (with the X button in the upper
	   right), also runs triggers another run.
	-  ``Esc`` will close all windows and interrupt the pype routine
	   (triggers ``sys.exit()``, which will also end a Python session if run
	   from the command line), as well as any loops.
	-  Each step that requires user interaction (e.g. ``create_mask`` or
	   ``landmarks``) needs to be confirmed with ``Enter`` until the next
	   function in the sequence is executed.
	-  At the end of the analysis, when the final steps (visualization and
	   export functions) have run, use ``Ctrl+Enter`` to finish and close
	   the window.

	**Function execution**

	-  ``Pype`` will automatically load the image and execute all functions
	   in sequence, but it will not overwrite overwrite data from past
	   iterations on disk unless specified.
	-  To overwrite interactive user input, set the argument ``edit: true``
	   or ``edit: overwrite`` in the function’s annotation control
	   arguments.
	-  If you forget to remove an overwrite argument and are prompted to
	   overwrite previous input, immediately change to ``edit: false``
	   argument, and save the config file.
	-  If a ``Pype`` is initialized on a project directory it will attempt
	   to load input data (e.g. masks) that contain the provided ``tag``
	   argument. For example,\ ``pp.Pype(path, tag="v1"`` will attempt to
	   load any files in the directory that contain the suffix ``"v1"``
	   (e.g. ``"annoations_v1.json"``).

	**Visualizing the results**

	Aspects of visual feedback during a ``pype`` run (can be completely
	suppressed by setting ``visualize=False``:

	-  Visual feedback (i.e. output from ``landmarks``, ``detect_contours``
	   or ``create_mask``) are drawn onto a “canvas” (a copy of the original
	   image).
	-  Use ``select_canvas`` to draw the results either on the raw image, a
	   binary image, or a single colour channel (gray, red, green or blue).
	-  If ``select_canvas`` is not explicitly specified, it is called
	   automatically and defaults to the raw image as canvas.
	-  Output from all functions, **needs to be specified manually**. For
	   example, after using ``- landmarks``, ``- draw_landmarks`` should be
	   called in the ``visualization`` module.
	-  Visual parameters of interactive tools (e.g. ``point_size`` or
	   ``line_thickness``) are specified separately in the respective
	   function, *and* in the ``visualization`` module.

	**Exporting the results**

	Saving annotations, canvas and other results:

	-  All results are saved automatically, even if the respective functions
	   in ``export`` are not specified, with the ``tag`` argument in
	   ``Pype`` as suffix.
	-  If a file already exist in the directory, and the respective function
	   is *not* listed under ``export``, then it *will not* be overwritten.
	-  If an export function *is* specified under ``export:``, it *will also
	   not overwrite* any existing file, unless specified using
	   ``overwrite: true``.
	-  The canvas is an exception: it will always be saved and always be
	   overwritten (unless specified with ``overwrite: False`` to show the
	   output from the last iteration. However, users can modify the canvas
	   name with ``file_name`` in the arguments to save different output
	   side by side. For example, ``file_name: binary`` under
	   ``- save_canvas:`` save the canvas as ``canvas_binary.jpg``


.. autoclass:: phenopype.main.Pype
	:members:
	:undoc-members:
	:show-inheritance: