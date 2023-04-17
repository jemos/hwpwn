.. hwpwn documentation master file, created by
   sphinx-quickstart on Mon Apr 10 22:57:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hwpwn
=====

This is a Python-based hardware security analysis tool/library designed to streamline basic operations on
signals captured during hardware security experiments, to provide consistent plotting capabilities,
among others. The motivation behind creating this tool was to share the code with others after completing
the experiments, realizing that it could be of interest to the broader community.

Although existing Python libraries, such as numpy and scipy, already offer functionalities for signal
processing, hwpwn leverages these well-established libraries to specifically cater to the needs of
hardware security attack analysis. This allows for more efficient and accessible handling of electrical
signals in the context of hardware security.

This project is currently in its early stages of development, and as such, it may not be as comprehensive or
polished as desired. However, it is a work in progress, and future enhancements are expected to improve
its functionality and usability in hardware security analysis. Users are encouraged to provide feedback
and contribute to the project's growth and refinement.

Instalation
-----------

To install simply use pip command as installing a normal python package.

.. code-block::

   $ pip install hwpwn
   ...

To learn more about how to use this tool, refer to the Quick Start section.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   configuration
   reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. sidebar-links::
   :github:
   :pypi: hwpwn