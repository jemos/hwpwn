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


def import_npz(npzfile: str, ts: float = 4e-9):
    npzdata = np.load(npzfile, allow_pickle=True)
    raw_data = npzdata['data']
    header = npzdata['column_names']

    logging.info(f"loaded {len(raw_data)} datapoints from {npzfile}.")
    x_axis = [raw_data[i][0] for i in range(0, len(raw_data))]

    triggers = []
    signals = []
    for i in range(1, len(raw_data[0])):
        # This is a trigger signal
        if 'T' == header[i][-1:] or '_HT' in header[i]:
            logging.info(f"found trigger signal {header[i]}")
            tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
            triggers.append({'name': header[i], 'vector': tv})
            continue

        # This is a normal signal
        tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
        signals.append({'name': header[i], 'vector': tv})

    return {'x_axis': x_axis, 'signals': signals, 'triggers': triggers, 'ts': ts*1e-6}


def import_csv(csvfile: str, scale: float = 1e-6, ts: float = 4e-9):
    """
    Loads a CSV file into a data structure that is easier to use for signal processing and plotting. This function
    expects a CSV with the time in the first column and signal voltages in the following columns. The first line
    must have the signal labels. If the label starts with character "T", it's considered to be a trigger signal.
    There can be more than one trigger signal in the file.
    """
    cfg_scale = scale
    cfg_ts = ts
    with open(csvfile, "r") as f:
        cr = csv.reader(f)
        header = next(cr)
        raw_data = list(cr)

    logging.info(f"loaded {len(raw_data)} datapoints from {csvfile}.")
    x_axis = [float(raw_data[i][0]) for i in range(0, len(raw_data))]
    new_ts = cfg_ts
    if new_ts is None:
        new_ts = float(raw_data[1][0]) - float(raw_data[0][0]) * 1.0e-6/cfg_scale
        if not math.isclose(abs(min(x_axis) - max(x_axis)), 0.0):
            logging.warning("the time axis seems to have different intervals between some points, please verify.")
        logging.info("inferred sampling period from data (%0.3f)." % new_ts)
        logging.info("if this is wrong, please use/correct --sample-period option.")
    else:
        new_ts = cfg_ts * 1e6
        logging.info("using sampling period of %0.3f ps." % new_ts)

    triggers = []
    signals = []
    for i in range(1, len(raw_data[0])):
        # This is a trigger signal
        if 'T' == header[i][-1:] or '_HT' in header[i]:
            logging.info(f"found trigger signal {header[i]}")
            tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
            triggers.append({'name': header[i], 'vector': tv})
            continue

        # This is a normal signal
        tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
        signals.append({'name': header[i], 'vector': tv})

    return {'x_axis': x_axis, 'signals': signals, 'triggers': triggers, 'ts': new_ts*1e-6}


def export_config(config: dict, filename: str):
    with open(filename, "w") as f:
        f.write(yaml.dump(config))


if __name__ == "__main__":
    app()
