Installing Spyder (optional)
############################

phenopype behaves like most Python packages, and thus can be run from any Python shell. However, if you plan on using the tool in an interactive fashion, i.e., run single lines of code from your script rather than running an entire scripts, like R in RStudio, I highly recommend the installation of `Spyder <https://docs.spyder-ide.org/current/index.html>`_, a scientific programming environment for Python. Spyder is free, open-source, and maintained by an awesome community!

.. image:: /_assets/images/spyder.png
   :align: center
   :alt: Spyder IDE
   
Install the latest version of Spyder with mamba:

.. code-block:: bash

	mamba activate <NAME>  			# <NAME> == chosen name, e.g. "pp-env"	
	mamba install spyder
	
After successful installation, start Spyder to run your scripts:

.. code-block:: python

	spyder
	

	
