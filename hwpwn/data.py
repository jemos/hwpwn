import typer
from . import common

app = typer.Typer(callback=common.default_typer_callback)


def process_data(data_in: dict, config: dict):
    data_aux = data_in
    for op in config['operations']:
        key = list(op.keys())[0]
        opd = op[key] or {}
        if key == 'signals':
            data_aux = data_subset_signals(data_aux, op[key])
        elif key == 'zoom':
            tstart = float(opd['tstart'])
            tend = float(opd['tend'])
            data_aux = data_zoom(data_aux, tstart=tstart, tend=tend)
        elif key == 'filter':
            ftype = opd['type']
            fcut = float(opd['fcut'])
            order = int(opd['order'])
            if ftype == 'HP':
                data_aux = data_filter_highpass(data_aux, cutoff=fcut, order=order)
            else:
                logging.error(f"Invalid or unsupported filter type: {ftype}")
        elif key == 'remove_dc':
            data_aux = data_remove_dc(data_aux)
        elif key == 'phase_correct':
            method = 'xcorr'
            if 'method' in opd:
                method = opd['method']
            if method not in ('xcorr', 'mwxcorr'):
                logging.error(f"bad method in phasecorr ({method})")
                sys.exit(-1)
            capidx = int(opd['capidx']) if 'capidx' in opd else None
            refname = opd['refname'] if 'refname' in opd else None
            winsize = float(opd['winsize']) if 'winsize' in opd else 3.0
            preskip = float(opd['preskip']) if 'preskip' in opd else 0.5
            posskip = float(opd['posskip']) if 'posskip' in opd else 0.5
            if method == 'xcorr':
                data_aux = data_xcorr(data_aux, capidx, refname)
            if method == 'mwxcorr':
                data_aux = data_mwxcorr(data_aux, winsize=winsize, preskip=preskip, posskip=posskip)
        elif key == 'average':
            append_op = opd['append'] if 'append' in opd else False
            avg_sig_name = opd['avg_sig_name'] if 'avg_sig_name' in opd else 'avg_signals'
            avg_trig_name = opd['avg_trig_name'] if 'avg_trig_name' in opd else 'avg_triggers'
            data_aux = data_average(data_aux, append=append_op, avg_sig_name=avg_sig_name,avg_trig_name=avg_trig_name)
        elif key == 'export':
            output = opd['output'] if 'output' in opd else "data_export_%s.csvz" % datetime.utcnow().isoformat()
            data_export(data_aux, output=output)
        elif key == 'import':
            filename = opd['filename']
        elif key == 'normalize':
            data_aux = data_normalize(data_aux)
        elif key == 'subtract':
            append = opd['append'] if 'append' in opd else False
            data_aux = data_subtract(data_aux, pos=opd['pos'], neg=opd['neg'], dest=opd['dest'], append=append)
        elif key == 'multiply':
            append = opd['append'] if 'append' in opd else False
            data_aux = data_multiply(data_aux, source=opd['source'], multiplier=opd['multiplier'], dest=opd['dest'],
                                     append=append)
        elif key == 'crop':
            data_aux = data_crop(data_aux, min_y=opd['min_y'])
        elif key == 'mltest':
            data_aux = data_mltest(data_aux)
        elif key == 'dtw':
            data_aux = data_dtw(data_aux)
        elif key == 'plot':
            pngfile = opd['pngfile'] if 'pngfile' in opd else None
            alpha = opd['alpha'] if 'alpha' in opd else 1.0
            ylabel = opd['ylabel'] if 'ylabel' in opd else None
            plot = opd['type'] if 'type' in opd else 'time'
            legendloc = opd['legendloc'] if 'legendloc' in opd else 'upper right'
            if plot == 'time':
                plot_time(data_aux, title=opd['title'], legend=opd['legend'], pngfile=pngfile, grid=opd['grid'],
                          alpha=alpha, ylabel=ylabel)
            elif plot == 'xy':
                plot_xy(data_aux, title=opd['title'], legend=opd['legend'], pngfile=pngfile, grid=opd['grid'],
                          alpha=alpha, ylabel=ylabel)
            else:
                plot_freq(data_aux, title=opd['title'], legend=opd['legend'], pngfile=pngfile, grid=opd['grid'],
                          legendloc=legendloc)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    return sos


def butter_highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y


def data_filter_highpass(data: dict, cutoff: float, order: int = 5):
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    sos = butter_highpass(cutoff=cutoff, fs=1.0/data['ts'], order=order)
    for s in data['signals']:
        logging.info("filtering signal {:s} using HP filter with cutoff frequency of {:0.3f} and order {:d}".format(
            s['name'], cutoff, order))
        new_signal = {'name': s['name'], 'vector': signal.sosfiltfilt(sos, s['vector'])}
        new_data['signals'].append(new_signal)
    return new_data


def data_mwxcorr(data: dict, winsize: float = 3.0, preskip: float = 0.5, posskip: float = 0.5):
    # Fill names if missing (default: all names)
    names = []
    for s in data['signals']:
        names.append(s['name'])
    for s in data['triggers']:
        names.append(s['name'])

    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': [], 'ts': data['ts']}

    winsizen = int(winsize*1e-6/data['ts'])
    logging.info(f"window size is {winsizen} points")
    x_axis = np.array(data['x_axis'])
    logging.info(f"time axis has {len(x_axis)} points")
    x_axis = x_axis[0:int(len(x_axis)/winsizen)*winsizen]
    logging.info(f"gap corrected time axis has {len(x_axis)} points")

    # save corrected x_axis
    new_data['x_axis'] = x_axis

    for s_idx in range(0, len(data['signals'])):
        signal = data['signals'][s_idx]
        logging.info(f"processing signal {signal['name']} ...")

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
                #logging.info(f"at window {widx+1}, i_start={i_start}, i_end={i_end}")
                sub_ref = ref_vector[i_start:i_end]
                sub_orig = orig[i_start:i_end]

                # calculate lag
                #logging.info("performing cross-correction to identify lag...")
                corr = np.correlate(sub_ref, sub_orig, mode='full')
                max_corr = max(corr)
                max_idx = np.argmax(corr) - len(sub_ref)

                logging.info(f"lagging signal by {max_idx} samples.")
                aux = np.roll(sub_orig, max_idx)

                #new_t = new_t + x_axis
                vector = np.concatenate((vector, aux), axis=0)

        # save generated signal
        logging.info(f"new vector length {len(vector)}")
        new_signal = {'name': signal['name'], 'vector': vector}
        new_data['signals'].append(new_signal)

    for s_idx in range(0, len(data['triggers'])):
        signal = data['triggers'][s_idx]
        logging.info(f"adjusting trigger signal length {signal['name']} ...")
        orig = signal['vector']
        vector = orig[:len(x_axis)]
        logging.info(f"new vector length {len(vector)}")
        new_signal = {'name': signal['name'], 'vector': vector}
        new_data['triggers'].append(new_signal)

    return new_data


def data_zoom(data: dict, tstart: float = None, tend: float = None):
    """
    Filter data from a specific time interval, so triggers, time axis and signal will be copied
    and only points between tstart and tend will remain. If tstart is None or tend is None, it will
    default to the minimum or maximum time range accordingly.
    """
    if tstart is None and tend is None:
        return data

    # Fill default values for tstart and tend, if these are not provided
    if tstart is None:
        tstart = min(data['x_axis'])
    if tend is None:
        tend = max(data['x_axis'])

    logging.info("extracting signals points within the time interval between {:0.3f} and {:0.3f}".format(tstart, tend))

    x_axis = np.array(data['x_axis'])
    idx_list = np.where(np.logical_and(x_axis < tend, x_axis > tstart))
    x_axis = x_axis[idx_list]

    new_data = {'x_axis': x_axis, 'signals': [], 'triggers': [], 'ts': data['ts']}

    for s in data['signals']:
        new_vector = np.array(s['vector'])[idx_list]
        new_signal = {'name': s['name'], 'vector': new_vector}
        new_data['signals'].append(new_signal)

    for t in data['triggers']:
        new_vector = np.array(t['vector'])[idx_list]
        new_trigger = {'name': t['name'], 'vector': new_vector}
        new_data['triggers'].append(new_trigger)

    return new_data


def data_subset_signals(data: dict, names: list[str]):
    if names is None:
        return data
    logging.info("filtering signals to include only {:s}".format(", ".join(names)))

    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': [], 'ts': data['ts']}
    for s in data['signals']:
        if s['name'] not in names:
            continue
        new_signal = {'name': s['name'], 'vector': s['vector']}
        new_data['signals'].append(new_signal)

    for s in data['triggers']:
        if s['name'] not in names:
            continue
        new_trigger = {'name': s['name'], 'vector': s['vector']}
        new_data['triggers'].append(new_trigger)

    logging.info(f"resulting signals {len(new_data['signals'])} and triggers {len(new_data['triggers'])}")
    return new_data


def operation_triggers(data: dict, names: list[str]):
    if names is None:
        return data
    logging.info("filtering triggers to include only {:s}".format(", ".join(names)))

    new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': [], 'ts': data['ts']}
    for t in data['triggers']:
        if t['name'] not in names:
            continue
        new_trigger = {'name': t['name'], 'vector': t['vector']}
        new_data['triggers'].append(new_trigger)

    logging.info(f"resulting trigger count {len(new_data['triggers'])}")
    return new_data


def data_remove_dc(data: dict):
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        mean = np.mean(s['vector'])
        vector = np.subtract(s['vector'], mean)
        new_mean = np.mean(vector)
        logging.info(
            "removing DC component of signal {:s}, with mean {:0.3f} ...".format(s['name'], mean))
        new_signal = {'name': s['name'], 'vector': vector}
        new_data['signals'].append(new_signal)

    return new_data


def data_normalize(data: dict):
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        v = np.array(s['vector'])
        v = (v - v.min()) / (v.max() - v.min())
        new_signal = {'name': s['name'], 'vector': v}
        new_data['signals'].append(new_signal)
    return new_data


def plot_time(data: dict, title: str = None, grid: bool = False, pngfile: str = None, legend: bool = False,
              alpha: float = 1.0, ylabel: str = None):

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 24

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    pltcolor = iter(cm.rainbow(np.linspace(0, 1, len(data['signals']) + len(data['triggers']) + 1)))
    fig, ax = plt.subplots()

    if grid is True:
        logging.info("enabling grid.")
        plt.grid(color='lightgray', alpha=0.5, zorder=1)

    fig.set_size_inches(13, 5)
    fig.gca()

    ax.set_xlabel('Time (us)')
    ax.set_ylabel(ylabel or 'mV')
    if len(data['triggers']):
        ax_trig = ax.twinx()
        ax_trig.set_ylabel('V')

    for s in data['signals']:
        logging.info(f"plotting signal {s['name']} ...")
        c = next(pltcolor)
        ax.plot(data['x_axis'], np.multiply(s['vector'], 1000), label=s['name'], c=c, alpha=alpha, linewidth=0.5)
        if 'markers' in s and len(s['markers']):
            ax.plot(s['markers_x'], np.multiply(s['markers'], 1000), label=s['name'], marker='o', c=c, linestyle='None')

    for s in data['triggers']:
        logging.info(f"plotting trigger signal {s['name']} ...")
        c = next(pltcolor)
        ax_trig.plot(data['x_axis'], np.multiply(s['vector'], 1), label=s['name'], c=c, alpha=1.0, linewidth=0.5)

    if legend is True:
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    return ax


def plot_xy(data: dict, title: str = None, grid: bool = False, pngfile: str = None, legend: bool = False,
            alpha: float = 1.0, ylabel: str = None):

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 24

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    pltcolor = iter(cm.rainbow(np.linspace(0, 1, len(data['signals']) + len(data['triggers']) + 1)))
    fig, ax = plt.subplots()

    if grid is True:
        logging.info("enabling grid.")
        plt.grid(color='lightgray', alpha=0.5, zorder=1)

    fig.set_size_inches(13, 5)
    fig.gca()

    ax.set_xlabel(ylabel or 'mV')
    ax.set_ylabel(ylabel or 'mV')
    if len(data['triggers']):
        ax_trig = ax.twinx()
        ax_trig.set_ylabel('V')

    x_axis = np.multiply(data['signals'][0]['vector'], 1000)
    for s in data['signals'][1:]:
        logging.info(f"plotting signal {s['name']} ...")
        c = next(pltcolor)
        ax.scatter(x_axis, np.multiply(s['vector'], 1000), label=s['name'], c=c, alpha=alpha)
        if 'markers' in s and len(s['markers']):
            ax.plot(s['markers_x'], np.multiply(s['markers'], 1000), label=s['name'], marker='o', c=c, linestyle='None')

    if legend is True:
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    return ax


def plot_bar(data: dict, title: str = None, grid: bool = False, pngfile: str = None, legend: bool = False):
    first = True
    sns_data = []
    x_axis = data['x_axis']
    cols = ['idx'] + [s['name'] for s in data['signals']]
    for i in range(0, len(x_axis)):
        row = [x_axis[i]]
        for s in data['signals']:
            row.append(s['vector'][i])
        sns_data.append(row)

    sns_data = np.array(sns_data)
    df = pandas.DataFrame(data=sns_data, columns=cols)
    df1 = df.drop(df.index[100:])
    sns.catplot(data=df1, kind='bar', x='idx', y='x_1')
    #sns.show()

    if pngfile is not None:
        plt.savefig(pngfile)
    plt.show()


def calc_fft(vector: list[float], ts: float):

    sr = 1/ts
    x = vector
    X = np.fft.fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T

    data = {'freq': freq, 'mag': np.abs(X)}
    return data


def plot_freq(data: dict, title: str = None, grid: bool = False, pngfile: str = None, legend: bool = True,
              legendloc: str = "upper right"):

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 24

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    pltcolor = iter(cm.rainbow(np.linspace(0, 1, len(data['signals']) + len(data['triggers']) + 1)))
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 5)
    fig.gca()

    if grid is True:
        logging.info("enabling grid.")
        plt.grid(color='lightgray', alpha=0.5, zorder=1)

    ax.set_xlabel('Freq [Hz]')
    ax.set_ylabel('log(|X(freq)|)')

    for signal in data['signals']:
        fft = calc_fft(vector=signal['vector'], ts=data['ts'])
        c = next(pltcolor)
        #ax.stem(fft['freq']*1e-6, fft['mag'], 'b', markerfmt=" ", basefmt="-b", color=c, alpha=0.8, label=signal['name'])
        ax.plot(fft['freq'], fft['mag'], label=signal['name'], c=c, alpha=0.8, linewidth=1.0)
        #ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e3, 32e6)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if legend:
        fig.legend(loc=legendloc, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)
    plt.show()
    return ax


def data_min_max_points(data: dict):
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        logging.info("calculating min, max, points of signal {:s} ...".format(s['name']))
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
                logging.info(f"max_idx={max_idx} max_v={max_v}")
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
                logging.error("Unexpected state.")
                sys.exit(-1)
            #new_vector.append(0.0)

        new_signal = {'name': s['name'], 'vector': s['vector'], 'markers': markers, 'markers_x': markers_x}
        new_data['signals'].append(new_signal)

    return new_data


def data_markers_to_signals(data: dict):
    new_data = {'x_axis': None, 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}

    lens = []
    for s in data['signals']:
        lens.append(len(s['markers']))
    max_idx = min(lens)

    for s in data['signals']:
        logging.info("converting markers to signals of signal {:s} ...".format(s['name']))
        new_signal = {'name': s['name'], 'vector': s['markers'][:max_idx]}
        new_data['signals'].append(new_signal)

    new_data['x_axis'] = range(0, max_idx)
    return new_data


def data_catch22(data: dict):
    new_data = {'x_axis': None, 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}

    lens = []
    for s in data['signals']:
        lens.append(len(s['markers']))
    max_idx = min(lens)

    for s in data['signals']:
        logging.info("converting markers to signals of signal {:s} ...".format(s['name']))
        new_signal = {'name': s['name'], 'vector': s['markers'][:max_idx]}
        new_data['signals'].append(new_signal)

    new_data['x_axis'] = range(0, max_idx)
    return new_data


def data_get_signal(data: dict, name: str):
    for s in data['signals']:
        if s['name'] == name:
            return s
    return None


def data_xcorr(data: dict, capidx: int = None, refname: str = None):
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}

    if refname is None:
        ref_vector = data['signals'][0]['vector']
        ref_name = data['signals'][0]['name']
    else:
        ref_signal = data_get_signal(data, refname)
        ref_name = refname
        if ref_signal is None:
            logging.error(f"unable to find signal with name {refname}!")
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
        logging.info(f"correcting signal {s['name']} lag by {max_idx} points...")
        aux = np.roll(s['vector'], max_idx)
        new_signal = {'name': s['name'], 'vector': aux}
        new_data['signals'].append(new_signal)
    return new_data


def data_plot(data: dict, plot: str = 'time', title: str = None, grid: bool = False, legend: bool = False,
              pngfile: str = None, ylabel: str = None):
    if plot == 'time':
        plot_time(data=data, title=title, grid=grid, legend=legend, pngfile=pngfile, ylabel=ylabel)
    elif plot == 'freq':
        plot_freq(data=data, title=title, grid=grid, legend=legend, pngfile=pngfile)
    elif plot == 'bar':
        plot_bar(data=data, title=title, grid=grid, legend=legend, pngfile=pngfile)


def data_average(data: dict, append: bool = False, avg_sig_name: str = 'avg_signal', avg_trig_name: str = 'avg_trig'):
    logging.info("averaging signals")
    if append:
        new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': data['triggers'],
                    'ts': data['ts']}
    else:
        new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': [], 'ts': data['ts']}
    vectors = [s['vector'] for s in data['signals']]
    new_data['signals'].append({'name': avg_sig_name, 'vector': np.mean(vectors, axis=0)})
    vectors = [s['vector'] for s in data['triggers']]
    new_data['triggers'].append({'name': avg_trig_name, 'vector': np.mean(vectors, axis=0)})
    return new_data


def data_subtract(data: dict, pos: str, neg: str, dest: str, append: bool = False):
    logging.info(f"calculating signal subtract {dest} = {pos} - {neg} (append={append})")
    if append:
        new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': data['triggers'],
                    'ts': data['ts']}
    else:
        new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    pos_signal = data_get_signal(data, pos)
    neg_signal = data_get_signal(data, neg)
    new_data['signals'].append({'name': dest, 'vector': np.subtract(pos_signal['vector'], neg_signal['vector'])})
    return new_data


def data_multiply(data: dict, source: str, multiplier: float, dest: str, append: bool = False):
    logging.info(f"multiplying signal {source} with {multiplier} (append={append})")
    if append:
        new_data = {'x_axis': data['x_axis'], 'signals': data['signals'], 'triggers': data['triggers'],
                    'ts': data['ts']}
    else:
        new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    signal = data_get_signal(data, source)
    new_data['signals'].append({'name': dest, 'vector': np.multiply(signal['vector'], multiplier)})
    return new_data


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


def data_export(data: dict, output: str, format: str = 'pklz'):
    if format == 'pklz':
        logging.info(f"Saving data to {output} in GZIPed Pickle format...")
        with gzip.open(output, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if format == 'csv':
        logging.info(f"Saving data to {output} in CSV format...")
        with open(output, 'w') as f:
            csvw = csv.writer(f)
            rows = data_dict_to_list(data)
            csvw.writerows(rows)
        return

    if format == 'csvz':
        logging.info(f"Saving data to {output} in GZIPed CSV format...")
        with gzip.open(output, 'wb') as f:
            csvw = csv.writer(f)
            rows = data_dict_to_list(data)
            csvw.writerows(rows)
        return

    logging.error(f"Invalid or unsupported format provided ({format})!")
    sys.exit(-1)


def data_crop(data: dict, min_y: float):
    logging.info(f"cropping signals at min_y={min_y}")
    new_data = {'x_axis': data['x_axis'], 'signals': [], 'triggers': data['triggers'], 'ts': data['ts']}
    for s in data['signals']:
        v = np.array(s['vector'])
        idx = np.where(v > min_y)
        new_v = np.zeros(len(v))
        new_v[idx] = v[idx]
        new_data['signals'].append({'name': s['name'], 'vector': new_v})
    return new_data


def data_mltest(data_in: dict):
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    from dtw import dtw
    import matplotlib.pyplot as plt

    # Generate some example signals
    signal1 = np.array(data_in['signals'][0]['vector'])
    signal2 = np.array(data_in['signals'][1]['vector'])

    pprint(signal1)
    pprint(signal2)

    # Concatenate the signals into a matrix
    data = np.vstack([signal1, signal2])

    # Standardize the data
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    # Compute the pairwise distances using DTW
    dist, _, _, _ = pairwise_distances(data_std, metric=dtw)

    #plt.scatter(signal1, signal2)
    #plt.show()

    # Compute the linkage matrix using Ward's method
    Z = linkage(dist, method='ward')

    # Plot the dendrogram
    #plt.figure(figsize=(10, 5))
    #dendrogram(Z)
    #plt.show()


def data_dtw(data_in: dict):
    import numpy as np
    import matplotlib.pyplot as plt
    from fastdtw import fastdtw

    # Generate two example signals with different phases
    signal1 = np.array(data_in['signals'][0]['vector'][0:1000])
    signal2 = np.array(data_in['signals'][1]['vector'][0:1000])

    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    distance, warp_path = fastdtw(signal1, signal2, dist=euclidean_distance)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Remove the border and axes ticks
    fig.patch.set_visible(False)
    ax.axis('off')

    for [map_x, map_y] in warp_path:
        ax.plot([map_x, map_y], [signal1[map_x], signal2[map_y]], '-k')

    ax.plot(signal1, color='blue', marker='o', markersize=10, linewidth=5)
    ax.plot(signal2, color='red', marker='o', markersize=10, linewidth=5)
    plt.show()