.. DenseTorch documentation master file, created by
   sphinx-quickstart on Wed May  1 19:30:09 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/drsleep/densetorch

.. mdinclude:: ../../README.md

.. mdinclude:: ../../examples/multitask/README.md


.. mdinclude:: ../../examples/singletask/README.md



Code Documentation
===================

densetorch.nn
------------

The `nn` module implements a range of well-established encoders and decoders.

.. automodule:: densetorch.nn.decoders
    :members:
.. automodule:: densetorch.nn.mobilenetv2
    :members:
.. automodule:: densetorch.nn.xception
    :members:


densetorch.engine
------------

The `engine` module contains metrics and losses typically used for tasks of semantic segmenation and depth estimation.
Also contains training and validation functions.

.. automodule:: densetorch.engine.losses
    :members:
.. automodule:: densetorch.engine.metrics
    :members:
.. automodule:: densetorch.engine.trainval
    :members:

densetorch.data
------------

The `data` module implements datasets and relevant utilities used for data pre-processing. It supports multi-modal data.

.. automodule:: densetorch.data.datasets
    :members:
.. automodule:: densetorch.data.utils
    :members:



densetorch.misc
-----------

The `misc` module has various useful utilities.

.. automodule:: densetorch.misc.utils
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
