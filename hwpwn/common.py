import logging
import typer
import os
import yaml
import json
import sys

global cfg


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
                           config: str = None):
    global cfg
    lvl = logging.INFO
    fmt = "%(levelname)s: %(message)s"
    if verbose:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format=fmt)

    if config:
        cfg = config_load(filepath=config)