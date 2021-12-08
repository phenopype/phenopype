Installing Python
-----------------

Download and install `Anaconda3 <https://www.anaconda.com/distribution/>`_ (Python 3). Anaconda is a scientific python distribution which comes with the most common scientific Python packages already built in. Follow the OS-specific instructions below:

For Windows
~~~~~~~~~~~

Open the Anaconda installer file you downloaded. Depending on the installation location of your choice you may need admin rights (right click on the file and select `Run as Administrator`), but DON'T install Anaconda to `Program files`, as this may cause problems later. A good idea is typically the root directory of your hard drive (e.g. `C:\\Anaconda3`). Then test if the installation was successful by opening a terminal and typing:

.. code-block:: bash

	conda --version

If :code:`conda` is not recognized you need add the path to your Anaconda installation directory to your environmental variables (if you have not done so during the installation). To do so, go to `Control Panel\\System` and `Security\\System\\Advanced System Settings` and look for `Environment Variables`. Then click `new` and add the path to the Anaconda folder (i.e., the path you selected during installation - e.g. `C:\\Anaconda3`) and the subfolder `scripts` (e.g. `C:\\Anaconda3\\Scripts`.

An alternative to manipulating the environment variables is to use the anaconda prompt that can be launched from a shortcut in the `Start` menu (should get added during the installation), or through the `Anaconda Navigator <https://docs.anaconda.com/anaconda/user-guide/getting-started/>`_).


For Linux
~~~~~~~~~

Run the Anaconda installer script you downloaded, e.g. :code:`bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh`, and follow the instructions. When the installer prompts “Do you wish the installer to initialize Anaconda3 by running conda init?”, type `yes`. Then test if the installation was successful by opening a terminal and typing:

.. code-block:: bash

	conda --version

If :code:`conda` is not recognized you need to add the path to your Anaconda installation directory to your `.bashrc` file. To do so, type :code:`echo 'export PATH=/path/to/anaconda3/bin:$PATH' >> ~/.bashrc`.


Troubleshooting references
~~~~~~~~~~~~~~~~~~~~~~~~~~

Consult these references if you have trouble installing anaconda 

- https://docs.anaconda.com/anaconda/install/
- https://docs.anaconda.com/anaconda/user-guide/troubleshooting/
- https://stackoverflow.com/questions/28612500/why-anaconda-does-not-recognize-conda-command
- https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10
- https://askubuntu.com/questions/908827/variable-path-issue-conda-command-not-found


 