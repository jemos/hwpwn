Configuration
=============

The hwpwn has common configuration parameters that are shared across the modules. The configuration
parameters are:

* The `ts` which stands for sample period.
* The `scale` which is the default time scale.

The `scale` is used for example in the plots. When the plot module is used to show a plot, it will
use this parameter to adjust the x-axis scale. In the modules commands, whenever you see a reference
to :math:`\mathrm{cfg_\_scale}`, it refers to this scale configuration.

Now that the configuration parameters are understood, how do we change them?

Changing Configuration
----------------------

To set specific configuration parameters, the `hwpwn` options in the command line can be used, as
shown below.

.. code-block::

   $ hwpwn --ts <TS_VALUE> --scale <SCALE> ...

In addition, the `flow` module supports loading a specific flow configuration file. This configuration
file may contain `hwpwn` configuraiton parameters. Refer to the module documentation on the flow file
for more information.