import gzip
import math
import pickle
import numpy as np
import csv
from scipy import signal
import sys
import typer

from . import common

app = typer.Typer(callback=common.default_typer_callback)


@app.command()
def mwxcorr(winsize: float = 3.0):
    """
    Moving window, cross-correlation phase correction. By default, uses window of 3.0 time units.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': [], 'ts': data['ts']}

    winsizen = int(winsize*1e-6/data['ts'])
    common.info(f"window size is {winsizen} points")
    x_axis = np.array(data['x_axis'])
    common.info(f"time axis has {len(x_axis)} points")
    x_axis = x_axis[0:int(len(x_axis)/winsizen)*winsizen]
    common.info(f"gap corrected time axis has {len(x_axis)} points")

    # save corrected x_axis
    new_data['x_axis'] = x_axis

    for s_idx in range(0, len(data['signals'])):
        signal = data['signals'][s_idx]
        common.info(f"processing signal {signal['name']} ...")

        # remove DC component from signal
        # mean = np.mean(signal['vector'])
        # orig = np.subtract(signal['vector'], mean)

        # high-pass filter the signal at 7MHz
        # orig = butter_highpass_filter(orig, 4.0e6, 1.0 / data['ts'])
        orig = signal['vector']

        vector = []
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

                # calculate lag
                # common.info("performing cross-correction to identify lag...")
                corr = np.correlate(sub_ref, sub_orig, mode='full')
                max_idx = np.argmax(corr) - len(sub_ref)

                common.info(f"lagging signal by {max_idx} samples.")
                aux = np.roll(sub_orig, max_idx)

                # new_t = new_t + x_axis
                vector = np.concatenate((vector, aux), axis=0)

        # save generated signal
        common.info(f"new vector length {len(vector)}")
        new_signal = {'name': signal['name'], 'vector': vector}
        new_data['signals'].append(new_signal)

    for s_idx in range(0, len(data['triggers'])):
        signal = data['triggers'][s_idx]
        common.info(f"adjusting trigger signal length {signal['name']} ...")
        orig = signal['vector']
        vector = orig[:len(x_axis)]
        common.info(f"new vector length {len(vector)}")
        new_signal = {'name': signal['name'], 'vector': vector}
        new_data['triggers'].append(new_signal)

    common.finish(new_data)


@app.command()
def triggers(names: list[str]):
    """
    Filter data to include only the triggers specified in the arguments.
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


@app.command()
def min_max():
    """
    This function tries to identify the minimum and maximum points of the signals.
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
            elif not max_found and v < max_v and max_cnt < 5:
                max_cnt += 1
            elif not max_found and v < max_v and max_cnt >= 5:
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
            elif max_found and not min_found and v > min_v and min_cnt < 5:
                min_cnt += 1
            elif max_found and not min_found and v > min_v and min_cnt >= 5:
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


@app.command()
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


def data_catch22(data: dict):
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
    return new_data


def data_get_signal(data: dict, name: str):
    for s in data['signals']:
        if s['name'] == name:
            return s
    return None


@app.command()
def xcorr(capidx: int = None, refname: str = None):
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


def load_npz(npzfile: str, ts: float = 4e-9):
    npzdata = np.load(npzfile, allow_pickle=True)
    raw_data = npzdata['data']
    header = npzdata['column_names']

    common.info(f"loaded {len(raw_data)} datapoints from {npzfile}.")
    x_axis = [raw_data[i][0] for i in range(0, len(raw_data))]

    new_triggers = []
    new_signals = []
    for i in range(1, len(raw_data[0])):
        # This is a trigger signal
        if 'T' == header[i][-1:] or '_HT' in header[i]:
            common.info(f"found trigger signal {header[i]}")
            tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
            new_triggers.append({'name': header[i], 'vector': tv})
            continue

        # This is a normal signal
        tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
        new_signals.append({'name': header[i], 'vector': tv})

    return {'x_axis': x_axis, 'signals': new_signals, 'triggers': new_triggers, 'ts': ts*1e-6}


def load_csv(filepath: str, xscale: float = 1e-6):
    """
    Loads a CSV file into a data structure that is easier to use for signal processing and plotting. This function
    expects a CSV with the time in the first column and signal voltages in the following columns. The first line
    must have the signal labels. If the label starts with character "T", it's considered to be a trigger signal.
    There can be more than one trigger signal in the file.
    """
    cfg_scale = common.cfg['scale']
    cfg_ts = common.cfg['ts']
    with open(filepath, "r") as f:
        cr = csv.reader(f)
        header = next(cr)
        raw_data = list(cr)

    common.info(f"loaded {len(raw_data)} datapoints from {filepath}.")
    x_axis = [float(raw_data[i][0]) for i in range(0, len(raw_data))]
    new_ts = cfg_ts
    if new_ts is None:
        new_ts = float(raw_data[1][0]) - float(raw_data[0][0]) * cfg_scale/xscale
        if not math.isclose(abs(min(x_axis) - max(x_axis)), 0.0):
            common.warning("the time axis seems to have different intervals between some points, please verify.")
        common.info("inferred sampling period from data (%0.3f)." % new_ts)
        common.info("if this is wrong, please use/correct --sample-period option.")
    else:
        new_ts = cfg_ts
        common.info("using sampling period of %0.3f [s]." % new_ts)

    triggers = []
    signals = []
    for i in range(1, len(raw_data[0])):
        # This is a trigger signal
        if 'T' == header[i][-1:] or '_HT' in header[i]:
            common.info(f"found trigger signal {header[i]}")
            tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
            triggers.append({'name': header[i], 'vector': tv})
            continue

        # This is a normal signal
        tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
        signals.append({'name': header[i], 'vector': tv})

    return {'x_axis': x_axis, 'signals': signals, 'triggers': triggers, 'ts': new_ts}


def load_csvz(filepath: str, xscale: float = 1e-6):
    """
    Loads a CSV file into a data structure that is easier to use for signal processing and plotting. This function
    expects a CSV with the time in the first column and signal voltages in the following columns. The first line
    must have the signal labels. If the label starts with character "T", it's considered to be a trigger signal.
    There can be more than one trigger signal in the file.
    """
    cfg_scale = common.cfg['scale']
    cfg_ts = common.cfg['ts']
    with gzip.open(filepath, "rb") as f:
        cr = csv.reader(f)
        header = next(cr)
        raw_data = list(cr)

    common.info(f"loaded {len(raw_data)} datapoints from {filepath}.")
    x_axis = [float(raw_data[i][0]) for i in range(0, len(raw_data))]
    new_ts = cfg_ts
    if new_ts is None:
        new_ts = float(raw_data[1][0]) - float(raw_data[0][0]) * cfg_scale/xscale
        if not math.isclose(abs(min(x_axis) - max(x_axis)), 0.0):
            common.warning("the time axis seems to have different intervals between some points, please verify.")
        common.info("inferred sampling period from data (%0.3f)." % new_ts)
        common.info("if this is wrong, please use/correct --sample-period option.")
    else:
        new_ts = cfg_ts * 1e6
        common.info("using sampling period of %0.3f [s]." % new_ts)

    triggers = []
    signals = []
    for i in range(1, len(raw_data[0])):
        # This is a trigger signal
        if 'T' == header[i][-1:] or '_HT' in header[i]:
            common.info(f"found trigger signal {header[i]}")
            tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
            triggers.append({'name': header[i], 'vector': tv})
            continue

        # This is a normal signal
        tv = [float(raw_data[j][i]) for j in range(0, len(raw_data))]
        signals.append({'name': header[i], 'vector': tv})

    return {'x_axis': x_axis, 'signals': signals, 'triggers': triggers, 'ts': new_ts}


def load_pklz(filepath: str, xscale: float = 1e-6):
    raise NotImplementedError


@app.command()
def load(filepath: str, format: str = None, xscale: float = 1e-6):
    """
    Loads data from a local file. The file format can be: compressed CSV (.csvz), CSV (.csv), or compressed binary
    pickle (.pklz). The format is automatically determined from the filename extension. To override the format
    argument can be provided.

    The xscale is the x-axis scale in seconds. For instance, if a row has x set to 0.010 and this value is
    microseconds, the xscale should be 1e-6. Then, the x-axis is also controlled by the cfg_scale, which sets the
    final scale used for calculations and plotting. If, for example we'd like to have the values in micro-seconds,
    we use cfg_scale of 1e-6. Thus, the final value of x_i', from a raw x of x_i, is given by:

    x_i' = x_i * cfg_scale/xscale

    """
    if format is not None:
        if not (format in ('csvz', 'csv', 'pklz')):
            return common.error(f"Invalid or unsupported format provided ({format})!")
        filetype = format
    else:
        filetype = filepath[-3:]
        if not (filetype in ('csvz', 'csv', 'pklz')):
            return common.error(f"Invalid or unsupported file extension ({filetype})!")

    if filetype == 'csv':
        data_aux = load_csv(filepath=filepath, xscale=xscale)
    elif filetype == 'csvz':
        data_aux = load_csvz(filepath=filepath, xscale=xscale)
    elif filetype == 'pklz':
        data_aux = load_pklz(filepath=filepath)
    else:
        raise ValueError(f"Invalid or unsupported file extension ({filetype})!")

    common.finish(data_aux)


@app.command()
def save(filepath: str, format: str = 'csvz'):
    """
    Saves the current data to a file. By default the format is compressed CSV file.
    """
    if format == 'pklz':
        common.info(f"Saving data to {filepath} in GZIPed Pickle format...")
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(common.data_aux, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif format == 'csv':
        common.info(f"Saving data to {filepath} in CSV format...")
        with open(filepath, 'w') as f:
            csvw = csv.writer(f)
            rows = data_dict_to_list(common.data_aux)
            csvw.writerows(rows)
    elif format == 'csvz':
        common.info(f"Saving data to {filepath} in GZIPed CSV format...")
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            csvw = csv.writer(f)
            rows = data_dict_to_list(common.data_aux)
            csvw.writerows(rows)
    else:
        return common.error(f"Invalid or unsupported format provided ({format})!")

    common.finish(common.data_aux)


@app.command()
def signals(names: list[str]):
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


@app.command()
def xzoom(xstart: float = None, xend: float = None):
    """
    Filter data from a specific x-axis interval, so triggers, time axis and signal will be copied
    and only points between xstart and xend will remain. If xstart is None or xend is None, it will
    default to the minimum or maximum value accordingly.
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


@app.command()
def subtract(pos: str, neg: str, dest: str, append: bool = False):
    """
    Subtract two signals and save the result in a specific signal name (dest). If append is not set, the previous data
    structure is cleared and only the new signal will persist.
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
    new_data['signals'].append({
        'name': dest,
        'vector': np.subtract(pos_signal['vector'], neg_signal['vector']).tolist()
    })
    common.finish(new_data)


@app.command()
def multiply(source: str, multiplier: float, dest: str, append: bool = False):
    """
    Multiply two signals with multiplier and save the result in a specific signal name (dest). If append is not set,
    the previous data structure is cleared and only the new signal will persist.
    """
    data = common.data_aux
    common.info(f"multiplying signal {source} with {multiplier} (append={append})")
    if append:
        new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': data['triggers'],
                    'ts': data['ts']}
    else:
        new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    signal = data_get_signal(data, source)
    new_data['signals'].append({
        'name': dest,
        'vector': np.multiply(signal['vector'], multiplier).tolist()
    })
    common.finish(new_data)


@app.command()
def average(name: str = 'avg_signal', append: bool = False):
    """
    Calculate the average between all signals. By default, the new signal name is 'avg_signal' but this can be changed
    with the `name` argument. If append is not set, the previous data structure is cleared and only the new signal
    will persist.
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


@app.command()
def remove_mean():
    """
    Remove the mean from the signal. Essentialy, the mean is calculated for each signal and the mean value is
    subtracted for all signal values.
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


@app.command()
def normalize():
    """
    Normalize the signals based on max and min values.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        v = np.array(s['vector'])
        v = (v - v.min()) / (v.max() - v.min())
        new_signal = {'name': s['name'], 'vector': v.tolist()}
        new_data['signals'].append(new_signal)
    common.finish(new_data)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    return sos


@app.command()
def hp(cutoff: float, order: int = 5):
    """
    High-pass filter with cutoff frequency specified by cutoff argument. The order can be changed from the default (5)
    using the `order` argument.
    """
    data = common.data_aux
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    sos = butter_highpass(cutoff=cutoff, fs=1.0/data['ts'], order=order)
    for s in data['signals']:
        common.info("filtering signal {:s} using HP filter with cutoff frequency of {:0.3f} and order {:d}".format(
            s['name'], cutoff, order))
        new_signal = {'name': s['name'], 'vector': signal.sosfiltfilt(sos, s['vector'])}
        new_data['signals'].append(new_signal)
    common.finish(new_data)
