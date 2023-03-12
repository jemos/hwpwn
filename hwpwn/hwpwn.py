import gzip
import json
import logging
import math
import os
import pickle
from datetime import datetime
from pprint import pprint
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pandas
import yaml
from matplotlib.pyplot import cm
import csv
from scipy import signal
import sys
import seaborn as sns
from scipy.spatial.distance import euclidean
import typer

from . import data
from . import common

app = typer.Typer()
app.add_typer(data.app, name="data")


def export_config(config: dict, filename: str):
    with open(filename, "w") as f:
        f.write(yaml.dump(config))


if __name__ == "__main__":
    app()
