.. _gtc-install:

==============
Installing GTC
==============

.. _from-pypi:

From PyPI
---------

**GTC** is available as a `PyPI package`_. It can be installed using pip

.. code-block:: console

   pip install gtc

This is the recommended way to install the package.

From the Source Code
--------------------

**GTC** is actively developed on GitHub, where the `source code`_ is available.

The easiest way to install **GTC** with the latest features and updates is to run

.. code-block:: console

   pip install https://github.com/MSLNZ/GTC/archive/master.zip

Alternatively, you can either clone the public repository

.. code-block:: console

   git clone git://github.com/MSLNZ/GTC.git

or download the tarball_ (Unix) or zipball_ (Windows) and then extract it.

Once you have a copy of the source code, you can install it by running

.. code-block:: console

   cd GTC
   pip install .

The recommended way to install **GTC** is :ref:`from PyPI <from-pypi>` because it is
a stable release.

Dependencies
------------
* Python 2.7, 3.4+
* scipy_

.. _PyPI package: https://pypi.org/project/GTC/
.. _source code: https://github.com/MSLNZ/GTC/
.. _tarball: https://github.com/MSLNZ/GTC/archive/master.tar.gz
.. _zipball: https://github.com/MSLNZ/GTC/archive/master.zip
.. _scipy: https://www.scipy.org/
