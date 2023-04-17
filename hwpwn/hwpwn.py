import yaml
import typer
import colorama

from . import common
from . import data
from . import plot
from . import flow

colorama.init()


app = typer.Typer(callback=common.default_typer_callback)
app.add_typer(data.app, name="data")
app.add_typer(plot.app, name="plot")
app.add_typer(flow.app, name="flow")


def export_config(config: dict, filename: str):
    with open(filename, "w") as f:
        f.write(yaml.dump(config))


if __name__ == "__main__":
    app()
