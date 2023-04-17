r"""
This module allows to execute hwpwn commands that are written in a YAML file. This is different from executing
multiple hwpwn commands in chain in the console.

The flow file structure is the following:

.. code-block::

    ---
    options:
      scale: <number, i.e., 1e-6>
      ts: <number, i.e., 4e-9>
      description: |
        <A multiline description of this flow.
        Second line comes here.>
    operations:
      - data.load:
          filepath: my_data.csv.gz
      - plot.time:
      - flow.exit:

The options has three possible attributes: the scale, the ts, and the description. The operations is the list of
steps to be executed. These are run in sequence. They are specified with the notation `<module_name>.<function_name>`.

This essentially allows to execute any command from the modules, from the flow file.

.. HINT::
    Refer to the quickstart section for more examples of usage of this command.

"""
import json
import sys
import typer
from ruamel.yaml import YAML

from . import common

app = typer.Typer(callback=common.default_typer_callback)


def float_constructor(loader, node):
    return float(loader.construct_scalar(node))


@app.command(help="Runs a specific flow stored in the file specified by filepath parameter.")
def run(filepath: str):
    r"""
    Runs a specific flow stored in the file specified by :paramref:`filepath` parameter.

    :param filepath: Path to the flow to be executed. This file is expected to be in YAML format.
    """
    from . import data
    from . import plot

    yaml = YAML(typ='safe')
    with open(filepath, 'r') as f:
        flowraw = yaml.load(f)

    opts = flowraw['options'] if 'options' in flowraw else {}
    steps = flowraw['operations']

    # Set multicommand flag to inform hwpwn that multiple commands will be executed by this process.
    opts['multicommand'] = True

    # Load configuration from the YAML file
    common.config_from_data(opts)

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
        elif app_name == 'flow':
            cmd_app = sys.modules[__name__]
        else:
            return common.error(f'Unsupported or invalid app name {app_name}!')

        # Get the command object
        cmd = getattr(cmd_app, cmd_name)
        if isinstance(args, dict):
            cmd(**args)
        elif isinstance(args, list):
            cmd(args)
        else:
            cmd()

    if sys.stdout.isatty():
        common.info("Note: use a pipe if you want to see the output of the command.")
        return
    print(json.dumps(common.data_aux))


@app.command()
def exit():
    r"""
    Terminates the execution of the flow.
    """
    sys.exit()
