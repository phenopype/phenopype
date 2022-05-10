Installing phenopype
-----------------------

(These instructions are valid across operating systems, and use :code:`mamba` instead of :code:`conda`).


Initial installation
~~~~~~~~~~~~~~~~~~~~~~~~

Open a terminal. Then create a virtual environment with `mamba`. Using such environments will give you full control over which Python packages are installed, and reduces the change of package related issues. Note that phenopype requires Python v3.7, which needs to be explicitly specified. For example, for an environment named "pp", type:


.. code-block:: bash

	mamba create -n "pp" python=3.7


You can now activate your environment. **This needs to be done every time you are using phenopype**:

.. code-block:: bash

	conda activate pp


Now install phenopype to the environment using :code:`pip` (`pip` is the package installer for Python):

.. code-block:: bash

	pip install phenopype


.. tip::

	If you prefer an "Rstudio-like" environment, you can use Phenopype from a Python Integrated Development Environment (IDE), such as `Spyder <https://www.spyder-ide.org/>`_. `Spyder` needs to be installed with `mamba` directly to the environment you created before. Using the example from above:


	.. code-block:: bash

		conda activate pp
		mamba install spyder -c conda-forge


	Once installed, you can run `Spyder` by typing :code:`spyder`


That's it - happy `phenopyping`! You can now use phenopype by after loading :code:`python` or :code:`spyder` from the terminal. You can also use phenopype from a `jupyter notebook` - for more details, give the `tutorials <tutorial_0.html>`_ a try. **Always remember to activate your environment.**


Installing updates
~~~~~~~~~~~~~~~~~~~~~~

For regular major and minor releases, use the :code:`-U` flag with :code:`pip`:

.. code-block:: bash

	pip install phenopype -U

Installing past versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Major releases are not backwards compatible, so if you have existing phenopype projects that were created with a previous version you need to download that specific version. You can tell `pip` to do so, for example, for version `1.0.0`:

.. code-block:: bash

		pip install "phenopype==1.0.0"

Or, for the latest phenopype version that is still 1.x.x:

.. code-block:: bash

		pip install "phenopype < 2"
