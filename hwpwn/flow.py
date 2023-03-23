from pprint import pprint

import numpy as np
import typer
import yaml
from inspect import signature

from . import common

app = typer.Typer(callback=common.default_typer_callback)


@app.command()
def run(filepath: str):

    from . import data
    from . import plot

    with open(filepath, 'r') as f:
        flowraw = yaml.safe_load(f)

    opts = flowraw['options'] if 'options' in flowraw else {}
    steps = flowraw['operations']
    # Iterate over the steps and execute them in sequence
    for step in steps:
        # Get the command and arguments from the step
        command, args = list(step.items())[0]

        # Split the command into parts
        app_name, cmd_name = command.split(".")

        # Get the Typer application object
        if app_name == 'data':
            cmd_app = data
        elif app_name == 'plot':
            cmd_app = plot
        else:
            return common.error(f'Unsupported or invalid app name {app_name}!')

        # Get the command object
        cmd = getattr(cmd_app, cmd_name)

        if isinstance(args, dict):
            cmd(**args)
        elif isinstance(args, list):
            cmd_args = []
            cmd_args.extend(str(arg_value) for arg_value in args)
            cmd(*cmd_args)
