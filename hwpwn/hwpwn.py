import yaml
import typer

from . import common
from . import data
from . import plot

app = typer.Typer(callback=common.default_typer_callback)
app.add_typer(data.app, name="data")
app.add_typer(plot.app, name="plot")


def export_config(config: dict, filename: str):
    with open(filename, "w") as f:
        f.write(yaml.dump(config))


if __name__ == "__main__":
    app()
