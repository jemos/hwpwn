import logging
from pprint import pprint

import typer
import os
import yaml
import json
import sys

cfg = {'scale': 1, 'ts': 1}
data_aux = None


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


def config_get(config: dict, name: str, default: any = None):
    if name in config:
        return config[name]
    else:
        return default


def default_typer_callback(verbose: bool = typer.Option(False, "--verbose", "-v"),
                           config: str = typer.Option(None, help="File path that contains default options values."),
                           scale: float = typer.Option(None, help="Plot scale for the time axis [s]."),
                           ts: float = typer.Option(None, help="Default sample period of the data [s].")):
    global cfg, data_aux
    lvl = logging.ERROR
    fmt = "%(levelname)s: %(message)s"
    if verbose:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format=fmt)

    if config:
        cfg = config_load(filepath=config)

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
    if len(args) == 1:
        logging.error(args[0])
    else:
        logging.error(*args)
    sys.exit(-1)


def warning(*args):
    if len(args) == 1:
        logging.warning(args[0])
    else:
        logging.warning(*args)
    return


def info(*args):
    if len(args) == 1:
        logging.info(args[0])
    else:
        logging.info(*args)
    return


def finish(data: dict):
    if sys.stdout.isatty():
        print("Note: use a pipe if you want to see the output of the command.")
        return
    print(json.dumps(data))