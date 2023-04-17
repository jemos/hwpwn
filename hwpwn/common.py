r"""
This module contains common code used across the multiple modules such as the data, flow, and plot. For example, the
configuration options parsing is placed in this module. The data variable shared across modules is placed in this
module also (with name `data_aux`). The format of the `data_aux` is described below.

Attributes:
    **data_aux** (dict): A dictionary that has the current data information.
        - x_axis (list[float]): The x-axis column values (shared for all rows).
        - signals (list[dict]): The list of signals.
        - triggers (list[dict]): The list of triggers.
        - ts (float): The sample period for the x_axis.

Each signal and trigger has the same data structure of a `signal`.

Attributes:
    **signal** (dict): A signal informaiton
        - vector (list[float]): The values of the signal.
        - name (str): The name of the signal.

Example of data structure:

.. code-block:: python

    data_aux = {
      'x_axis': [1, 2, 3],
      'signals': [
        { 'name': 'test', 'vector': [5, 6, 1] }
      ],
      'triggers': [],
      'ts': 1
    }

Another action done by the common module is to handle the piping of data information across multiple commands (as
examplified in the quick start).
"""

import logging
from pprint import pprint

import typer
import os
import yaml
import json
import sys

cfg = {'scale': 1, 'ts': 1}
data_aux = None
f_verbose = False


def config_load(filepath: str):
    filename, fileext = os.path.splitext(filepath)
    if fileext == '.yaml':
        with open(filepath, 'r') as f:
            return yaml.safe_load(f.read())
    if fileext == '.json':
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    logging.error(f"Unsupported or invalid config file extesion ({fileext})!")
    sys.exit(-1)


def config_from_data(config: dict):
    global cfg
    if 'ts' in config:
        cfg['ts'] = float(config['ts'])
    if 'scale' in config:
        cfg['scale'] = float(config['scale'])
    return


def config_get(name: str, default: any = None, fail_on_missing: bool = True):
    global cfg
    if name in cfg:
        return cfg[name]
    else:
        if fail_on_missing:
            return error(f"trying to use a configuration parameter ({name}) which was not defined,"
                         f" refer to help to know how to define configuration parameters.")
        return default


def default_typer_callback(verbose: bool = typer.Option(False, "--verbose", "-v"),
                           logfile: str = typer.Option(None, help="Log to file instead of stdout."),
                           scale: float = typer.Option(None, help="Plot scale for the time axis [s]."),
                           ts: float = typer.Option(None, help="Default sample period of the data [s].")):
    global cfg, data_aux, f_verbose
    log_format = "%(levelname)s: %(message)s"
    if verbose:
        logging.basicConfig(level=logging.INFO, format=log_format)
        f_verbose = True
    else:
        logging.basicConfig(level=logging.ERROR, format=log_format)
        f_verbose = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    # Override scale if it's provided in the options.
    if scale is not None:
        cfg['scale'] = scale

    # Override sample period if it's provided in the options.
    if ts is not None:
        cfg['ts'] = ts

    if sys.stdin.isatty():
        return

    stdin = sys.stdin.read().lstrip()
    if not len(stdin):
        return
    logging.info("loading data from stdin...")
    try:
        data_aux = json.loads(stdin)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON input: {e}")
    if not validate_data(data_aux):
        logging.error(f"Input data did not pass validation tests!")


def validate_data(datatest: dict):
    return True


def error(*args):
    r"""
    Outputs an error message and exits the application.
    """
    if len(args) == 1:
        logging.error(args[0])
    else:
        logging.error(*args)
    sys.exit(-1)


def warning(*args):
    r"""
    Outputs a warning message.
    """
    if len(args) == 1:
        logging.warning(args[0])
    else:
        logging.warning(*args)
    return


def info(*args):
    r"""
    Outputs an information message.
    """
    if len(args) == 1:
        logging.info(args[0])
    else:
        logging.info(*args)
    return


def finish(data: dict):
    r"""
    This function should be called whenever a command has completed. It will store the data passed by :paramref:`data`
    in the local `data_aux` variable to persist across sequential command calls in the same process.
    """
    global data_aux, f_verbose, cfg
    data_aux = data
    if sys.stdout.isatty():
        logging.info("Note: use a pipe if you want to see the output of the command.")
        return

    if 'multicommand' in cfg and cfg['multicommand'] is True:
        if f_verbose:
            print(json.dumps(data))
        else:
            return
