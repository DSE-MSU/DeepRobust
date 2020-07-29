.. Deep documentation master file, created by
   sphinx-quickstart on Fri Jul  3 12:19:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Starting to build your robust models with DeepRobust!
================================
.. comments original size: 626*238

.. image:: ./DeepRobust.png
   :width: 313px
   :height: 119px

DeepRobust is a pytorch adversarial learning library, which contains most popular attack and defense algorithms in image domain and graph domain.

Installation
============
#. Activate virtual environment
#. Install package
   .. code-block:: none
    $ git clone https://github.com/DSE-MSU/DeepRobust.git
    $ cd DeepRobust
    $ python setup.py install

Package API
===========
.. toctree::
   :maxdepth: 1
   :caption: Image Package
   
   source/deeprobust.image.attack
   source/deeprobust.image.defense
   source/deeprobust.image.netmodels


.. toctree::
   :maxdepth: 1
   :caption: Graph Package
   
   source/deeprobust.graph.global_attack
   source/deeprobust.graph.targeted_attack
   soure/deeprobust.graph.defense
   source/deeprobust.graph.data

Indices and tables
==================

* :ref:`modindex`
* :ref:`search`
