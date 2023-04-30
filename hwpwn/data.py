"""
This module provides a set of functions for data manipulation. For instance, there are functions to:

* Load data from a file.
* Save data to a file.
* Perform basic operations such as subtraction and factor multiplication.
* Cropping the signal within a x-axis interval.
* Normalizing and DC-removal.
* Phase correcting the signals using cross-correlation.
* Phase correcting based on moving window.

Example usage:

.. code-block:: python

    >>> import hwpwn.data as d
    >>> d.load('sample.csv')
    >>> d.signals(['a1', 'a2', 'b1', 'b2'])
    >>> d.triggers(['a_T', 'b_T'])
    >>> d.hp(cutoff=10e3, order=3)
    >>> d.save(filepath="output.csv.gz")

Between the commands the data is stored in `common` module, in a variable with name `data_aux`. The data structure (a
dictionary) is shared between the commands and has the following attributes:

* `x_axis` is the x-axis column, a list of numbers.
* `signals` a list of signals.
* `triggers` a list of triggers.
* `ts` the sample period.

For more information refer to the :doc:`common module documentation</common>`.
"""

import gzip
import io
import math
import os.path
from pprint import pprint

import numpy as np
import csv
from scipy import signal
import sys
import typer
import pywt

from . import common

app = typer.Typer(callback=common.default_typer_callback)


def gaussian_periodic_penalty(lag: int, period: int, sigma: float):
    """
    :meta private:
    """
    abs_lag = np.abs(lag)
    min_distance = np.minimum(((abs_lag - period) % period),
                              (period - ((abs_lag - period) % period)))
    penalty = np.exp(-(min_distance**2) / (2 * sigma**2))
    penalty = np.where(abs_lag < period / 2, 0, penalty)
    return penalty


def correlate_diff_penalized(x_axis: list[float], vector1: list[float], vector2: list[float], capidx: int = 10,
                             period: int = 6, sigma: float = 0.25, weight: float = 5):
    """
    :meta private:
    """
    common.info("calculating signal correlation through sum of absolute errors with penality element:")
    csums = []
    idxrange = list(range(-capidx, capidx, 1))
    for i in idxrange:
        v2aux = np.roll(np.array(vector2), i).tolist()
        # Calculate Sum( | vector1 - v2aux | ) + w*D(lag, period, sigma) and add it to csums
        cae = np.sum(np.abs(np.subtract(vector1, v2aux)))
        penality = weight * gaussian_periodic_penalty(i, period, sigma)
        common.info(f"  .. index {i}: cae={cae:.2f} penality={penality:.2f}")
        csums.append({'lag': i, 'score': cae + penality})

    best_lag = min(csums, key=lambda x: x['score'])
    common.info(f"  .. best correction lag for window {min(x_axis):.2f} to {max(x_axis):.2f} is: lag={best_lag['lag']}, score={best_lag['score']:.2f}")
    return best_lag['lag']


def correlate_diff_min_sae(vector1: list[float], vector2: list[float], capidx: int = 10):
    """
    :meta private:
    """
    common.info("calculating best signal lag using minimum sum of absolute errors:")
    csums = []
    idxrange = list(range(-capidx, capidx, 1))
    for i in idxrange:
        v2aux = np.roll(np.array(vector2), i).tolist()
        # Calculate Sum( | vector1 - v2aux | ) and add it to csums
        csums.append(np.sum(np.abs(np.subtract(vector1, v2aux))))

    min_diff = np.min(csums)
    min_idxs = [i for i, x in enumerate(csums) if x == min_diff]
    min_idx = np.abs(min_idxs).min()
    common.info(f"  .. best lag found was: {min_idx}")
    return min_idx


def correlate_xcorr(vector1: list[float], vector2: list[float]):
    """
    :meta private:
    """
    common.info("calculating signal correlation through cross-correlation:")
    corr = np.correlate(vector1, vector2, mode='full')
    best_lag = np.argmax(corr) - len(vector1)
    common.info(f"  .. best lag found was: {best_lag}")
    return best_lag


@app.command(help="This command utilizes a parametrizable cross-correlation technique with a moving window to correct "
                  "for signal lag. Rather than applying the correction to the entire signal at once, it is applied "
                  "within a moving window")
def mwxcorr(winsize: str = '3.0e-6', mode: str = 'individual', function: str = 'xcorr', period: int = 6,
            sigma: float = 0.25, weight: float = 10):
    """
    This command utilizes a parametrizable cross-correlation technique with a moving window to correct for signal lag.
    Rather than applying the correction to the entire signal at once, it is applied within a moving window, as
    illustrated in the following image.

    .. image:: images/mwxcorr.png
      :width: 300
      :align: center

    For each window, a function can be selected to calculate the best corrective lag. This is given by the
    :paramref:`function` argument.

    - The **diff**, will calculate the sum of absolute error (SAE) and return the lag with the minimum SAE.
    - The **diffp** is a period-penalizing version of diff, where periodic lags become less likely.
    - The **xcorr** will use the well known cross-correlation function.

    Now, having multiple lag values, one for each window, this command can use these lags differently,
    depending on the :paramref:`mode` argument.

    - The **individual** mode, each window lag is used individually.
    - The **average** mode, the average lag is employed in all windows.
    - The **consensus** mode, where the most frequent lag is employed in all windows.

    :param winsize: The window size in x-axis units.
    :type winsize: str
    :param mode: The operation done in the lags of each window. This argument can accept multiple values.
                 Possible values are: "individual", "average", "consensus".
    :type mode: str
    :param function: The function to calculate the lag in each window. This argument can accept multiple values.
                     Possible values are: "diff", "diffp", "xcorr".
    :type function: str
    :param period: Is only used with `diffp` function and is the lag period in vector index units. Lags of this value
                   will be avoided with `diffp` function.
    :param sigma: Is only used with `diffp` function and corresponds to the square root of the variance of the
                  Guass curve.
    :param weight: Is only used with `weight` function and is a scaling factor for the penalizing term. The higher
                   this value, the more effect the penalizing term has in the lag calculation.

    """
    common.info("moving window cross-correlation started...")
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': [], 'ts': data['ts']}

    pprint(float(winsize))
    pprint(data['ts'])
    winsizen = int(round(float(winsize)/data['ts']))
    common.info(f"window size is {winsizen} points")
    x_axis = np.array(data['x_axis'])
    common.info(f"time axis has {len(x_axis)} points")
    x_axis = x_axis[0:int(len(x_axis)/winsizen)*winsizen]
    common.info(f"gap corrected time axis has {len(x_axis)} points")

    # save corrected x_axis
    new_data['x_axis'] = x_axis

    for s_idx in range(0, len(data['signals'])):
        sig = data['signals'][s_idx]
        common.info(f"processing signal {sig['name']} ...")

        # remove DC component from signal
        # mean = np.mean(signal['vector'])
        # orig = np.subtract(signal['vector'], mean)

        # high-pass filter the signal at 7MHz
        # orig = butter_highpass_filter(orig, 4.0e6, 1.0 / data['ts'])
        orig = sig['vector']

        vector = []
        corr_values = []
        if s_idx == 0:
            vector = orig[:len(x_axis)]
            ref_vector = orig[:len(x_axis)]
        else:
            for widx in range(0, int(len(ref_vector)/winsizen)):
                i_start = widx*winsizen
                i_end = (widx+1)*winsizen
                # common.info(f"at window {widx+1}, i_start={i_start}, i_end={i_end}")
                sub_ref = ref_vector[i_start:i_end]
                sub_orig = orig[i_start:i_end]
                sub_x = x_axis[i_start:i_end]

                # calculate lag
                # common.info("performing cross-correction to identify lag...")
                if function == 'xcorr':
                    corr_value = correlate_xcorr(sub_ref, sub_orig)
                elif function == 'diff':
                    corr_value = correlate_diff_min_sae(sub_ref, sub_orig)
                elif function == 'diffp':
                    corr_value = correlate_diff_penalized(sub_x, sub_ref, sub_orig, period=period, sigma=sigma,
                                                          weight=weight)
                else:
                    return common.error(f"Invalid or unsupported function provided: {function}")

                corr_values.append(corr_value)

            if mode == 'consensus':
                common.info("employing consensus lag based correction")
                best_lag = max(set(corr_values), key=corr_values.count)
                common.info(f"lagging signal {sig['name']} by {best_lag} points (most common)")
                vector = np.roll(orig[:len(x_axis)], best_lag)
            elif mode == 'individual':
                common.info("employing individual lag based corrections")
                for widx in range(0, int(round(len(ref_vector) / winsizen))):
                    corr_value = corr_values[widx]
                    i_start = widx * winsizen
                    i_end = (widx + 1) * winsizen
                    sub_orig = orig[i_start:i_end]
                    common.info(f"lagging signal {sig['name']} by {corr_value} points")
                    aux = np.roll(sub_orig, corr_value)
                    vector = np.concatenate((vector, aux), axis=0)
                common.info(f"average correlation lag value: %0.3f" % np.mean(corr_values))
            elif mode == 'average':
                common.info("employing average lag based correction")
                best_lag = np.mean(corr_values)
                common.info(f"lagging signal {sig['name']} by {best_lag} points (most common)")
                vector = np.roll(orig[:len(x_axis)], best_lag)
            else:
                return common.error(f"invalid/unsupported mode provided ({mode})")

        # save generated signal
        common.info(f"new vector size {len(vector)}")
        new_signal = {'name': sig['name'], 'vector': vector}
        new_data['signals'].append(new_signal)

    for s_idx in range(0, len(data['triggers'])):
        sig = data['triggers'][s_idx]
        common.info(f"adjusting trigger signal length {sig['name']} ...")
        orig = sig['vector']
        vector = orig[:len(x_axis)]
        common.info(f"new vector length {len(vector)}")
        new_signal = {'name': sig['name'], 'vector': vector}
        new_data['triggers'].append(new_signal)

    common.finish(new_data)


@app.command(help="Filter specific triggers to retain only the ones specified by the names argument.")
def triggers(names: list[str]):
    """
    From the list of triggers, include only the ones specified by the :paramref:`names` argument.
    For example, if there is trigger `x1_T`, `x2_T`, `x3_T` and one uses the triggers command with names
    set to :code:`["x2_T"]`, the output data will only contain `x2_T` trigger as well as the data signals.

    :param names: the trigger names to include in the output data.
    """
    if names is None:
        return common.finish(common.data_aux)
    common.info("filtering triggers to include only {:s}".format(", ".join(names)))
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': [], 'ts': data['ts']}
    for t in data['triggers']:
        if t['name'] not in names:
            continue
        new_trigger = {'name': t['name'], 'vector': t['vector']}
        new_data['triggers'].append(new_trigger)
    common.info(f"resulting trigger count {len(new_data['triggers'])}")
    common.finish(new_data)


@app.command(help="This command tries to identify the minimum and maximum points of the signals.")
def min_max(max_win: int = 5, min_win: int = 5):
    r"""
    This function tries to identify the minimum and maximum points of the signals.

    .. WARNING::
       This function is experimental. It returns a different data structure compared to the other functions. In
       addition of signal `vector`, the returning signal will have also `markers` and `markers_x`.
       The reason is that the x-axis will be different, it will not be temporal based, but index based.

    .. image:: images/minmax.png
      :width: 400
      :align: center

    As shown in the plots above, the minimum and maximum values are calculated and returned as a list.

    .. NOTE::
       It's assumed that the signals are oscilating and the lookup will alternate between minimum and maximum values.
       Thus, this function doesn't support looking for groups of minimum points or maximum points in sequence.

    :param max_win: Number of points to wait for holding the last maximum value. If the maximum value doesn't change
                    within this number of points, the maximum is added to the list. By default this is 5.
    :param min_win: Number of points to wait for holding the last minimum value. If the minimum value doesn't change
                    within this number of points, the minimum is added to the list. By default this is 5.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        common.info("calculating min, max, points of signal {:s} ...".format(s['name']))
        max_v = None
        max_idx = None
        max_cnt = 0
        max_found = False
        min_v = None
        min_idx = None
        min_cnt = 0
        min_found = False
        markers = []
        markers_x = []
        for i, v in enumerate(s['vector']):
            if not max_v:
                max_v = v
                max_idx = i
                max_cnt = 0
                max_found = False
            elif not max_found and v > max_v:
                max_v = v
                max_idx = i
                max_cnt = 0
            elif not max_found and v < max_v and max_cnt < max_win:
                max_cnt += 1
            elif not max_found and v < max_v and max_cnt >= max_win:
                max_found = True
                min_found = False
                min_v = None
                min_idx = None
                min_cnt = 0
                common.info(f"max_idx={max_idx} max_v={max_v}")
                markers_x.append(data['x_axis'][max_idx])
                markers.append(max_v)
            elif max_found and not min_v:
                min_v = v
                min_idx = i
                min_cnt = 0
                min_found = False
            elif max_found and not min_found and v < min_v:
                min_v = v
                min_idx = i
                min_cnt = 0
            elif max_found and not min_found and v > min_v and min_cnt < min_win:
                min_cnt += 1
            elif max_found and not min_found and v > min_v and min_cnt >= min_win:
                min_found = True
                max_found = False
                max_v = None
                max_idx = None
                max_cnt = 0
                markers_x.append(data['x_axis'][min_idx])
                markers.append(min_v)
            else:
                common.error("Unexpected state.")
                sys.exit(-1)
            # new_vector.append(0.0)

        new_signal = {'name': s['name'], 'vector': s['vector'], 'markers': markers, 'markers_x': markers_x}
        new_data['signals'].append(new_signal)

    common.finish(new_data)


@app.command(help="Converts markers to signals so that they can be plotted directly using plot functions.")
def markers2signals():
    """
    Converts markers to signals so that they can be plotted directly using plot functions.
    """
    data = common.data_aux
    new_data = {'x_axis': None, 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}

    lens = []
    for s in data['signals']:
        lens.append(len(s['markers']))
    max_idx = min(lens)

    for s in data['signals']:
        common.info("converting markers to signals of signal {:s} ...".format(s['name']))
        new_signal = {'name': s['name'], 'vector': s['markers'][:max_idx]}
        new_data['signals'].append(new_signal)

    new_data['x_axis'] = range(0, max_idx)
    common.finish(new_data)


def data_get_signal(data: dict, name: str):
    """
    :meta private:
    """
    for s in data['signals']:
        if s['name'] == name:
            return s
    return None


@app.command(help="Corrects signal lag by employing cross-correlation.")
def xcorr(capidx: int = None, refname: str = None):
    """
    Corrects signal lag by employing cross-correlation.

    :param capidx: The maximum lag index value.
    :param refname: The reference signal name to use as comparison.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}

    if refname is None:
        ref_vector = data['signals'][0]['vector']
        ref_name = data['signals'][0]['name']
    else:
        ref_signal = data_get_signal(data, refname)
        ref_name = refname
        if ref_signal is None:
            common.error(f"unable to find signal with name {refname}!")
        ref_vector = ref_signal['vector']

    if capidx is not None:
        ref_vector = ref_vector[:capidx]

    new_data['signals'].append(data['signals'][0])
    for s in data['signals']:
        if s['name'] == ref_name:
            continue
        aux = s['vector']
        if capidx is not None:
            aux = aux[:capidx]
        corr = np.correlate(ref_vector, aux, mode='full')
        max_idx = np.argmax(corr) - len(ref_vector)
        common.info(f"correcting signal {s['name']} lag by {max_idx} points...")
        aux = np.roll(s['vector'], max_idx)
        new_signal = {'name': s['name'], 'vector': aux}
        new_data['signals'].append(new_signal)
    common.finish(new_data)


def data_dict_to_list(data: dict):
    """
    :meta private:
    """
    rows = []
    headers = ['time']
    [headers.append(s['name']) for s in data['signals']]
    [headers.append(s['name']) for s in data['triggers']]
    rows.append(headers)
    x_axis = data['x_axis']
    for i in range(len(x_axis)):
        row = [x_axis[i]]
        [row.append(s['vector'][i]) for s in data['signals']]
        [row.append(s['vector'][i]) for s in data['triggers']]
        rows.append(row)
    return rows


def process_raw_table_ts(raw_data: list[list], xscale: float = 1e-6):
    cfg_scale = common.cfg['scale']
    cfg_ts = common.cfg['ts']
    new_ts = cfg_ts
    x_axis = [float(raw_data[i][0]) for i in range(0, len(raw_data))]
    if new_ts is None:
        new_ts = float(raw_data[1][0]) - float(raw_data[0][0]) * xscale / cfg_scale
        if not math.isclose(abs(min(x_axis) - max(x_axis)), 0.0):
            common.warning("the time axis seems to have different intervals between some points, please verify.")
        common.info("inferred sampling period from data (%0.3f)." % new_ts)
        common.info("if this is wrong, please use/correct --ts option.")
    else:
        new_ts = cfg_ts
        common.info(f"using sampling period of {new_ts:0.1e} [s].")
    return new_ts


def process_raw_table_signals(header: list[str], raw_data: list[list]):
    """
    Internal function to process table data from header list and raw data information. The :paramref:`header` list is
    just a list of strings that correspond to the column names. The :paramref:`raw_data` is a list of rows. Each row
    is a list of values which are converted to float.

    Example usage:

    .. code-block:: python

       >>> cols = ['t', 'a', 'b']
       >>> rows = [[1, 10, 11], [2, 11, 13], [3, 9, 15]]
       >>> s, t = process_raw_table_signals(header=cols, raw_data=rows)

    :param header: List of strings corresponding to the column names.
    :type header: list[str]
    :param raw_data: List of rows, where each row is another list of strings or numbers. The items in the row will be
                     converted to float.
    :type raw_data: list[list]
    """
    trigs = []
    sigs = []
    for i in range(1, len(raw_data[0])):
        # This is a trigger signal
        if '_T' == header[i][-2:] or '_HT' in header[i]:
            common.info(f"found trigger signal {header[i]}")
            tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
            trigs.append({'name': header[i], 'vector': tv})
            continue

        # This is a normal signal
        common.info(f"found signal {header[i]}")
        tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
        sigs.append({'name': header[i], 'vector': tv})

    return sigs, trigs


@app.command(help="Loads a CSV file into a data structure that is easier to use for signal processing and plotting. "
                  "This function expects a CSV with the time in the first column and signal voltages in the following "
                  "columns. The first line must have the signal labels. If the label ends with characters \"_T\" "
                  "or characters \"_HT\", it's considered a trigger signal. There can be more than one trigger "
                  "signal in the file.")
def load_csv(filepath: str, xscale: float = 1e-6):
    r"""
    Loads a CSV file into a data structure that is easier to use for signal processing and plotting. This function
    expects a CSV with the time in the first column and signal voltages in the following columns. The first line
    must have the signal labels. If the label starts with character `T`, it's considered to be a trigger signal.
    There can be more than one trigger signal in the file.

    .. NOTE::
       The `load` command can accept multiple file types and call the correct load function. No need to call this
       command for CSV files if the file has a `.csv` extension.

    :param filepath: The file to load.
    :param xscale: The x-axis scale to seconds. Each x-axis value will be multipled by `xscale` from
                   and divided by `scale` parameter of configuration as follows:

    .. math::
       x_i' = x_i * \mathrm{xscale}/\mathrm{cfg\_scale}

    So for example, if `cfg_scale` is 1e-6 (meaning that we want to use common unit of x-axis in microseconds),
    and `xscale` is 1e-9 (meaning that the x-axis values units are in nanoseconds), and the CSV has the x-axis
    (230.0, 250.0, 410.0) the data stored in memory will have the x-axis values (0.230, 0.250, 0.410).
    """
    with open(filepath, "r") as f:
        cr = csv.reader(f)
        header = next(cr)
        raw_data = list(cr)

    common.info(f"loaded {len(raw_data)} datapoints from {filepath}.")
    sigs, trigs = process_raw_table_signals(header=header, raw_data=raw_data)
    new_ts = process_raw_table_ts(raw_data=raw_data, xscale=xscale)
    x_axis = np.multiply([float(raw_data[i][0]) for i in range(0, len(raw_data))], new_ts)

    return {'x_axis': x_axis, 'signals': sigs, 'triggers': trigs, 'ts': new_ts}


@app.command(help="This command is very similar to the load_csv command, except that the CSV file is compressed using "
                  "gzip. The first column is loaded as the x-axis, and signal voltages (including the triggers) are "
                  "loaded from the following columns.")
def load_csvz(filepath: str, xscale: float = 1e-6):
    r"""
    This command is very similar to the :func:`load_csv` command, except that the CSV file is compressed using gzip.
    The first column is loaded as the x-axis, and signal voltages (including the triggers) are loaded from the
    following columns.

    To distinguish between trigger signals and normal signals, a suffix should be used. If a column name ends with the
    characters `_T` or `_HT`, it is considered to be a trigger signal. There may be more than one trigger signal in the
    file.

    .. NOTE::
       The `load` command can accept multiple file types and call the correct load function. No need to call this
       command for `CSVZ` files if the file has a `.csvz` extension.

    :param filepath: The file to load.
    :param xscale: The x-axis scale to seconds. Each x-axis value will be multipled by `xscale` from
                   and divided by `scale` parameter of configuration as follows:

    .. math::
       x_i' = x_i * \mathrm{xscale}/\mathrm{cfg\_scale}

    So for example, if `cfg_scale` is 1e-6 (meaning that we want to use common unit of x-axis in microseconds),
    and `xscale` is 1e-9 (meaning that the x-axis values units are in nanoseconds), and the CSV has the x-axis
    (230.0, 250.0, 410.0) the data stored in memory will have the x-axis values (0.230, 0.250, 0.410).
    """
    with gzip.open(filepath, "r") as f:
        cr = csv.reader(io.TextIOWrapper(f, newline=""))
        header = next(cr)
        raw_data = list(cr)

    common.info(f"loaded {len(raw_data)} datapoints from {filepath}.")
    sigs, trigs = process_raw_table_signals(header=header, raw_data=raw_data)
    new_ts = process_raw_table_ts(raw_data=raw_data, xscale=xscale)
    x_axis = np.multiply([float(raw_data[i][0]) for i in range(0, len(raw_data))], new_ts)

    return {'x_axis': x_axis, 'signals': sigs, 'triggers': trigs, 'ts': new_ts}


def load_pklz(filepath: str, xscale: float = 1e-6):
    raise NotImplementedError


@app.command(help="Loads data from a local file. The file format can be: compressed CSV (.csv.gz) or CSV (.csv). "
                  "The format is automatically determined from the filename extension. To override the format "
                  "argument can be provided.")
def load(filepath: str, format: str = 'auto', xscale: float = 1e-6, append: bool = False):
    r"""
    Loads data from a local file. The file format can be: compressed CSV (.csv.gz) or CSV (.csv). The format is
    automatically determined from the filename extension. To override the format argument can be provided.

    Each x-axis value will be multipled by `xscale` from and divided by `scale` parameter of configuration as follows:

    .. math::
       x_i' = x_i * \mathrm{xscale}/\mathrm{cfg\_scale}

    So for example, if `cfg_scale` is 1e-6 (meaning that we want to use common unit of x-axis in microseconds),
    and `xscale` is 1e-9 (meaning that the x-axis values units are in nanoseconds), and the CSV has the x-axis
    (230.0, 250.0, 410.0) the data stored in memory will have the x-axis values (0.230, 0.250, 0.410).
    """
    common.info(f"trying to load file {filepath} ...")
    if append is True:
        raise NotImplementedError("append is not implemented yet in load()")
    if format != 'auto':
        if not (format in ('gz', 'csv')):
            return common.error(f"Invalid or unsupported format provided ({format})!")
        filetype = format
    else:
        _, filetype = os.path.splitext(filepath)
        if not (filetype in ('.gz', '.csv')):
            return common.error(f"Invalid or unsupported file extension ({filetype})!")

    if filetype == '.csv':
        data_aux = load_csv(filepath=filepath, xscale=xscale)
    elif filetype == '.gz':
        data_aux = load_csvz(filepath=filepath, xscale=xscale)
    else:
        raise ValueError(f"Invalid or unsupported file extension ({filetype})!")

    return common.finish(data_aux)


@app.command(help="Saves the current data to a file. By default the format is compressed CSV file (csv.gz).")
def save(filepath: str, format: str = 'csv.gz'):
    """
    Saves the current data to a file. By default the format is compressed CSV file (`.csv.gz`).

    :param filepath: Where to save the current data in memory.
    :param format: The format of the file. Supported values are: `csv`, or `csv.gz`.
    """
    if format == 'csv':
        common.info(f"Saving data to {filepath} in CSV format...")
        with open(filepath, 'w') as f:
            csvw = csv.writer(f)
            rows = data_dict_to_list(common.data_aux)
            csvw.writerows(rows)
    elif format == 'csv.gz':
        common.info(f"Saving data to {filepath} in GZIPed CSV format...")
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            csvw = csv.writer(f)
            rows = data_dict_to_list(common.data_aux)
            csvw.writerows(rows)
    else:
        return common.error(f"Invalid or unsupported format provided ({format})!")

    common.finish(common.data_aux)


@app.command(help="Filters the input signals to retain only the signals whose names are specified in the names "
                  "argument.")
def signals(names: list[str]):
    """
    Filters the input signals to retain only the signals whose names are specified in the :paramref:`names` argument.

    :param names: List of signal names to retain.
    :type names: list[str]
    """
    common.info("filtering signals to include only {:s}".format(", ".join(names)))

    new_data = {'x_axis': common.data_aux['x_axis'], 'signals': [], 'triggers': [], 'ts': common.data_aux['ts']}
    for s in common.data_aux['signals']:
        if s['name'] not in names:
            continue
        new_signal = {'name': s['name'], 'vector': s['vector']}
        new_data['signals'].append(new_signal)

    for s in common.data_aux['triggers']:
        if s['name'] not in names:
            continue
        new_trigger = {'name': s['name'], 'vector': s['vector']}
        new_data['triggers'].append(new_trigger)

    common.info(f"resulting signals {len(new_data['signals'])} and triggers {len(new_data['triggers'])}")
    common.finish(new_data)


@app.command(help="Filter data from a specific x-axis interval, so triggers, time axis and signal will be copied and "
                  "only points between xstart and xend will remain. If xstart or xend is None it will consider the "
                  "minimum or maximum value respectively.")
def xzoom(xstart: float = None, xend: float = None):
    """
    Filter data from a specific x-axis interval, so triggers, time axis and signal will be copied
    and only points between xstart and xend will remain. If :paramref:`xstart` is `None` or :paramref:`xend` is `None`,
    it will default to the minimum or maximum value accordingly.

    :param xstart: Start x-axis value of the interval to retain. If `None` defaults to minimum x-axis value.
    :type xstart: float
    :param xend: End x-axis value of the interval to retain. If `None` defaults to maximum x-axis value.
    :type xend: float

    Example usage:

    .. code-block:: python

       >>> xzoom(xstart=10, xend=20)

    If the configuration scale is set to 1e-6, this means it will crop the signals between 10us and 20us.
    """
    if xstart is None and xend is None:
        return common.finish(common.data_aux)

    # Fill default values for tstart and tend, if these are not provided
    xstart = min(common.data_aux['x_axis']) if xstart is None else xstart
    xend = max(common.data_aux['x_axis']) if xend is None else xend

    common.info("extracting signals points within the x interval between {:0.3f} and {:0.3f}".format(xstart, xend))
    x_axis = np.array(common.data_aux['x_axis'])
    idx_list = np.where(np.logical_and(x_axis <= xend, x_axis >= xstart))
    x_axis = x_axis[idx_list].tolist()

    new_data = {'x_axis': x_axis, 'signals': [], 'triggers': [], 'ts': common.data_aux['ts']}
    for s in common.data_aux['signals']:
        new_vector = np.array(s['vector'])[idx_list].tolist()
        new_signal = {'name': s['name'], 'vector': new_vector}
        new_data['signals'].append(new_signal)

    for t in common.data_aux['triggers']:
        new_vector = np.array(t['vector'])[idx_list].tolist()
        new_trigger = {'name': t['name'], 'vector': new_vector}
        new_data['triggers'].append(new_trigger)

    common.finish(new_data)


@app.command(help="Subtract two signals and save the result in a new signal name. It supports appending to existing "
                  "signals and also the absolute value calculation (turning this into an Absolute Error calculation "
                  "command)")
def subtract(pos: str, neg: str, dest: str, abs: bool = False, append: bool = False):
    """
    Subtract two signals and save the result in a new signal name. It supports appending to existing signals and also
    the absolute value calculation (turning this into an Absolute Error calculation command).

    :param pos: The positive term of the difference, a signal name.
    :param neg: The negative term of the difference, a signal name.
    :param dest: The signal name to store the result of the subtraction.
    :param append: If set `True`, the original signals are retained.
    :param abs: If set `True`, calculates the absolute of the subtraction before saving the value.
    """
    data = common.data_aux
    common.info(f"calculating signal subtract {dest} = {pos} - {neg} (append={append})")
    if append:
        new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': data['triggers'],
                    'ts': data['ts']}
    else:
        new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    pos_signal = data_get_signal(data, pos)
    neg_signal = data_get_signal(data, neg)
    diff = np.subtract(pos_signal['vector'], neg_signal['vector'])
    if abs:
        diff = np.abs(diff)
    new_data['signals'].append({
        'name': dest,
        'vector': diff.tolist()
    })
    common.finish(new_data)


@app.command(help="Multiply two signals with multiplier and save the result in a new signal. If append is not set,"
                  "the previous data structure is cleared and only the new signal will persist.")
def multiply(source: str, multiplier: float, dest: str, append: bool = False):
    """
    Multiply two signals with multiplier and save the result in a new signal. If append is not set,
    the previous data structure is cleared and only the new signal will persist.

    :param source: Source signal to multiply with the :paramref:`multiplier` value.
    :param multiplier: Value to multiply :paramref:`source` signal values with.
    :param dest: Destination signal name.
    :param append: If set `True`, the original signals are retained.
    """
    data = common.data_aux
    common.info(f"multiplying signal {source} with {multiplier} (append={append})")
    if append:
        new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': data['triggers'],
                    'ts': data['ts']}
    else:
        new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    sig = data_get_signal(data, source)
    new_data['signals'].append({
        'name': dest,
        'vector': np.multiply(sig['vector'], multiplier).tolist()
    })
    common.finish(new_data)


@app.command(help="Generates a new signal that is the average of the existing signals.")
def average(name: str = 'avg_signal', append: bool = False):
    r"""
    Generates a new signal that is the average of the existing signals. More specifically, this function does:

    .. math::
       \bar{x}[n] = \frac{1}{M} \sum_{i=1}^{M} x_i[n]

    The resulting signal, :math:`\bar{x}[n]`, is the average of the other signals given by :math:`x_i[n]`.

    :param name: The output signal name. By default, the new signal name is 'avg_signal'.
    :param append: If set `True`, the original signals are retained.
    """
    data = common.data_aux
    common.info(f"averaging signals")
    if append:
        new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': data['triggers'],
                    'ts': data['ts']}
    else:
        new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    vectors = [s['vector'] for s in data['signals']]
    new_data['signals'].append({'name': name, 'vector': np.mean(vectors, axis=0)})
    common.finish(new_data)


@app.command(help="Remove the signal mean. For each signal, the mean is calculated and then, each point of the signal "
                  "is subtracted by this mean value.")
def remove_mean():
    r"""
    Remove the signal mean. For each signal, the mean is calculated and then, each point of the signal is subtracted
    by this mean value.

    .. math::
       x_{new}[n] = x[n] - \frac{1}{N} \sum_{n=0}^{N-1} x[n]

    The new signal :math:`x_{new}[n]` is the orignal signal :math:`x[n]` subtracting the mean value.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        mean = np.mean(s['vector'])
        vector = np.subtract(s['vector'], mean).tolist()
        common.info("removing mean ({:0.3f}) of signal {:s} ...".format(mean, s['name']))
        new_signal = {'name': s['name'], 'vector': vector}
        new_data['signals'].append(new_signal)
    common.finish(new_data)


@app.command(help="Normalize the signals based on minimum and maximum values.")
def normalize():
    r"""
    Normalize the signals based on minimum and maximum values.

    .. math::
       x_{norm}[n] = \frac{x[n] - \min_{n}(x[n])}{ \max_{n} (x[n]) -  \min_{n} (x[n])}

    The normalization will make the resulting signal have a maximum value of 1.0 and a minimum value of 0.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        v = np.array(s['vector'])
        v = (v - v.min(initial=None)) / (v.max(initial=None) - v.min(initial=None))
        new_signal = {'name': s['name'], 'vector': v.tolist()}
        new_data['signals'].append(new_signal)
    common.finish(new_data)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    return sos


@app.command(help="Apply linear high-pass filter based on Butterworth polynomial to all the signals (except triggers).")
def hp(cutoff: float, order: int = 5, append: bool = False, append_prefix: str = 'fhp'):
    """
    Apply linear high-pass filter based on Butterworth polynomial to all the signals (except triggers).

    :param cutoff: The cutoff frequency (in Hertz) of the high-pass filter.
    :param order: The order of the Butterworth polynomial, which has direct impact in the steepness of the transition
                  between the passband and the stopband. By default, an order of 5 is used.
    :param append: If set `True`, the original signals are retained.
    :param append_prefix: If :paramref:`append` is `True`, this is the prefix used for the new signal names.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    sos = butter_highpass(cutoff=cutoff, fs=1.0/data['ts'], order=order)
    for s in data['signals']:
        common.info("filtering signal {:s} using HP filter with cutoff frequency of {:0.3f} and order {:d}".format(
            s['name'], cutoff, order))
        if append:
            new_name = '%s_%s' % (append_prefix, s['name'])
            new_data['signals'].append({'name': s['name'], 'vector': s['vector']})
            common.info(f"appending signal to {new_name} ...")
        else:
            new_name = s['name']
        new_signal = {'name': new_name, 'vector': signal.sosfiltfilt(sos, s['vector'])}
        new_data['signals'].append(new_signal)
    common.finish(new_data)


@app.command(help="Apply Wavelet-based denoising of the signals. Warning: this command is experimental.")
def denoise(wavelet: str = 'db8'):
    """
    Apply Wavelet-based denoising of the signals.

    :param wavelet: The name of the wavelet to use for the denoising.

    .. WARNING::
       This function is experimental.
    """
    common.info("denoise called.")
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        common.info("cleaning signal {:s} noise using wavelet denoising".format(s['name']))
        vector = s['vector']
        coeffs = pywt.wavedec(vector, wavelet)
        threshold = np.sqrt(2 * np.log(len(vector))) * (np.median(np.abs(coeffs[-1])) / 0.6745)
        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
        new_data['signals'].append({'vector': denoised_signal[1:], 'name': s['name']})
    common.finish(new_data)
