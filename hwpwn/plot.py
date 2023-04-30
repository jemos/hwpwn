r"""
This module commands provide a simple way to plot the data, while supporting different plot types, including:

* Time-series plot.
* Frequency based plot.
* Cross-correlation plot.

"""
import numpy as np
import typer
from matplotlib import pyplot as plt, cm
from scipy import signal

from . import common

app = typer.Typer()


@app.command()
def time(title: str = None, grid: bool = True, pngfile: str = None, legend: str = 'UR' or bool, alpha: float = 1.0,
         ylabel: str = 'x', yscale: float = 1.0, xunit: str = 'NA', yunit: str = 'NA', xlabel: str = 'Time',
         linewidth: float = 1.0, show: bool = True, xstart: float = None, xend: float = None, dpi: int = 100):
    r"""
    Time-series plot of the data. The x-axis in the data is used as the time-series axis for all the signals and
    triggers. This command supports providing multiple parameters but all are optional.

    :param title: The title of the plot, shown in the top center location.
    :param grid: If set to `True` enables grid.
    :param pngfile: If not `None`, will save the plot to a PNG file whose location is given by this parameter.
    :param legend: If not None, will show the legend in the location specified: UR=upper-right, UL=upper-left,
                   LR=lower-right, or LL=lower-left. By default it's set to UR (upper-right).
    :param alpha: A float number that sets the opacity of the signals plotted (0 completely transparent, 1.0 completely
                  visible.)
    :param ylabel: The label shown in the y-axis.
    :param yscale: Allows to adjust plot values scale with this factor.
    :param xunit: A string with the x-axis unit name (i.e., S, mA, etc.)
    :param yunit: A string with the y-axis unit name (i.e., V, mA, etc.)
    :param xlabel: A string with the x-axis label (by default, will be set to "Time").
    :param linewidth: The width of the signal lines (by default set to 1.0).
    :param show: Whether or not to show the plot. If :paramref:`pngfile` is used and this is set to `False`, it will
                 not show the plot and will only save it to a PNG file.
    :param xstart: Allows to set a x-axis start interval for the plot, so that only a subset of the data is plotted.
    :param xend: Allows to set a x-axis end interval for the plot, so that only a subset of the data is plotted.
    """
    data = common.data_aux

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
    fig, ax = plt.subplots(dpi=dpi)

    if grid is True:
        common.info("enabling grid.")
        plt.grid(color='lightgray', alpha=0.5, zorder=1)

    fig.set_size_inches(13, 5)
    fig.gca()
    #fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=None, hspace=None)
    #fig.tight_layout()

    ax.set_xlabel(f'{xlabel} [{xunit}]')
    ax.set_ylabel(f'{ylabel} [{yunit}]')
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)

    if xstart is not None and xend is not None:
        ax.set_xlim(xstart, xend)

    if len(data['triggers']):
        ax_trig = ax.twinx()
        ax_trig.set_ylabel('Triggers [V]')
        ax_trig.spines['left'].set_linewidth(2.0)
        ax_trig.spines['bottom'].set_linewidth(2.0)
        ax_trig.spines['right'].set_linewidth(2.0)
        ax_trig.spines['top'].set_linewidth(2.0)

    for s in data['signals']:
        common.info(f"plotting signal {s['name']} with scale {yscale} ...")
        c = next(pltcolor)
        ax.plot(data['x_axis'], np.multiply(s['vector'], yscale), label=s['name'], c=c, alpha=alpha,
                linewidth=linewidth)

    for s in data['triggers']:
        common.info(f"plotting trigger signal {s['name']} ...")
        c = next(pltcolor)
        ax_trig.plot(data['x_axis'], np.multiply(s['vector'], 1), label=s['name'], c=c, alpha=1.0, linewidth=linewidth)

    if legend is not False:
        if legend == 'UR' or legend is True:
            fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        elif legend == 'UL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
        elif legend == 'BL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 0), bbox_transform=ax.transAxes)
        elif legend == 'BR':
            fig.legend(loc='upper left', bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)

    if show is True:
        plt.show()
    return common.finish(data)


@app.command()
def xy(title: str = 'X-Y Plot', grid: bool = False, pngfile: str = None, legend: str = 'UR', alpha: float = 1.0,
       ylabel: str = None):
    data = common.data_aux

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
        common.info("enabling grid.")
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
        common.info(f"plotting signal {s['name']} ...")
        c = next(pltcolor)
        ax.scatter(x_axis, np.multiply(s['vector'], 1000), label=s['name'], c=c, alpha=alpha)

    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend is not False:
        if legend == 'UR' or legend is True:
            fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        elif legend == 'UL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
        elif legend == 'BL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 0), bbox_transform=ax.transAxes)
        elif legend == 'BR':
            fig.legend(loc='upper left', bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)

    plt.show()
    return common.finish(data)


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


@app.command()
def freq(title: str = 'Frequency Response', grid: bool = True, pngfile: str = None, legend: str = 'UR',
         ylabel: str = None, yunit: str = 'NA', ts: str = '1.0', show: bool = True, yscale: str = 'log',
         xscale: str = 'log'):
    """
    Frequency response plot for all signals.

    :param title: The title of the plot, shown in the top center location.
    :param grid: If set to `True` enables grid.
    :param pngfile: If not `None`, will save the plot to a PNG file whose location is given by this parameter.
    :param legend: If set to `True`, will show the legend of the signals plotted.
    :param ylabel: The label shown in the y-axis.
    :param yscale: Allows to adjust plot values scale with this factor.
    :param yunit: A string with the y-axis unit name (i.e., V, mA, etc.)
    :param show: Whether or not to show the plot. If :paramref:`pngfile` is used and this is set to `False`, it will
                 not show the plot and will only save it to a PNG file.
    :param legend: If not None, will show the legend in the location specified: UR=upper-right, UL=upper-left,
                   LR=lower-right, or LL=lower-left. By default it's set to UR (upper-right).
    :param ts: The sample period.
    :param xscale: The type of scale to use for the x-axis. Accepted values are 'log' and 'linear'. By default it
                   is set to 'log'.
    """
    data = common.data_aux

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
    fig, ax = plt.subplots(dpi=300)
    fig.set_size_inches(13, 5)
    fig.gca()

    if grid is True:
        common.info("enabling grid.")
        plt.grid(color='lightgray', alpha=0.5, zorder=1)

    ax.set_xlabel('Freq [Hz]')
    #ax.set_ylabel('log(|X(freq)|)')

    #ax.set_xlabel(f'{xlabel} [{xunit}]')
    ax.set_ylabel(f'{ylabel} [{yunit}]')

    for signal in data['signals']:
        fft = calc_fft(vector=signal['vector'], ts=float(ts))
        c = next(pltcolor)
        #ax.stem(fft['freq']*1e-6, fft['mag'], 'b', markerfmt=" ", basefmt="-b", color=c, alpha=0.8, label=signal['name'])
        ax.plot(fft['freq'], fft['mag'], label=signal['name'], c=c, alpha=0.8, linewidth=1.0)
        #ax.set_yscale('log')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(1e3, 32e6)

    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend is not False:
        if legend == 'UR' or legend is True:
            fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        elif legend == 'UL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
        elif legend == 'BL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 0), bbox_transform=ax.transAxes)
        elif legend == 'BR':
            fig.legend(loc='upper left', bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)
    if show:
        plt.show()
    return common.finish(data)


@app.command()
def xcorr(sig1: str, sig2: str, title: str = 'Cross Correlation', grid: bool = True, pngfile: str = None,
          legend: str = 'UR', linewidth: float = 1.0, minlag: int = None, maxlag: int = None,
          mode: str = 'full', show: bool = False):
    """
    Cross-correlation plots between two signals.

    :param sig1: The first signal name.
    :param sig2: The second signal name.
    :param title: The title of the plot, shown in the top center location.
    :param grid: If set to `True` enables grid.
    :param pngfile: If not `None`, will save the plot to a PNG file whose location is given by this parameter.
    :param legend: If set to `True`, will show the legend of the signals plotted.
    :param linewidth: The width of the signal lines (by default set to 1.0).
    :param minlag: Crop the x-axis lag value starting from this `minlag`.
    :param maxlag: Crop the x-axis lag value ending at this `maxlag`.
    :param mode: The cross-correlation mode to employ (refer to numpy correlate function documentation.)
    :param show: Whether or not to show the plot. If :paramref:`pngfile` is used and this is set to `False`, it will
                 not show the plot and will only save it to a PNG file.
    :param legend: If not None, will show the legend in the location specified: UR=upper-right, UL=upper-left,
                   LR=lower-right, or LL=lower-left. By default it's set to UR (upper-right).
    """
    from . import data
    data_aux = common.data_aux
    sig1d = data.data_get_signal(data_aux, sig1)
    sig2d = data.data_get_signal(data_aux, sig2)

    # Compute cross-correlation
    x = data_aux['x_axis']
    vector1 = sig1d['vector']
    vector2 = sig2d['vector']
    corr = np.correlate(vector1, vector2, mode=mode)

    # Compute phase shift
    lag = np.argmax(corr) - len(corr) // 2

    # Apply phase correction
    common.info(f"lag detected of {lag} between {sig1d['name']} and {sig2d['name']}")

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

    # Plot cross-correlation
    fig, ax = plt.subplots(dpi=300)

    if grid is True:
        common.info("enabling grid.")
        ax.grid(color='lightgray', alpha=0.5, zorder=1)

    fig.set_size_inches(13, 5)

    ax.plot(np.arange(-len(corr) // 2, len(corr) // 2), corr, linewidth=linewidth, label='Cross-Correlation')
    ax.axvline(x=lag, color='r', linestyle='--', linewidth=0.5, label='Max Correlation')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Cross-Correlation')
    if maxlag and minlag:
        ax.set_xlim([minlag, maxlag])

    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend is not False:
        if legend == 'UR' or legend is True:
            fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        elif legend == 'UL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
        elif legend == 'BL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 0), bbox_transform=ax.transAxes)
        elif legend == 'BR':
            fig.legend(loc='upper left', bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)

    if show is True:
        plt.show()
    return common.finish(data_aux)


@app.command()
def phasecorr(sig1: str, sig2: str, title: str = 'Phase Correlation', grid: bool = True, pngfile: str = None,
              legend: str = 'UR', linewidth: float = 1.0):
    r"""
    Phase correlation analysis plot.

    :param sig1: The first signal name.
    :param sig2: The second signal name.
    :param title: The title of the plot, shown in the top center location.
    :param grid: If set to `True` enables grid.
    :param pngfile: If not `None`, will save the plot to a PNG file whose location is given by this parameter.
    :param legend: If set to `True`, will show the legend of the signals plotted.
    :param linewidth: The width of the signal lines (by default set to 1.0).
    :param legend: If not None, will show the legend in the location specified: UR=upper-right, UL=upper-left,
                   LR=lower-right, or LL=lower-left. By default it's set to UR (upper-right).
    """
    from . import data
    data_aux = common.data_aux
    sig1d = data.data_get_signal(data_aux, sig1)
    sig2d = data.data_get_signal(data_aux, sig2)

    # Compute phase correlation
    x = data_aux['x_axis']
    vector1 = sig1d['vector']
    vector2 = sig2d['vector']
    signal_product = np.fft.fft(vector1) * np.conj(np.fft.fft(vector2))
    phase_corr = np.fft.ifft(signal_product)
    lag = np.argmax(phase_corr.real)

    # Apply phase correction
    vector2_corrected = np.roll(vector2, lag)

    # Plot signals
    fig, axs = plt.subplots(3, 1, dpi=300)

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

    if grid is True:
        common.info("enabling grid.")
        plt.grid(color='lightgray', alpha=0.5, zorder=1)

    fig.set_size_inches(13, 5)
    fig.gca()

    axs[0].plot(x, vector1)
    axs[0].set_ylabel('Vector 1')
    axs[1].plot(x, vector2)
    axs[1].set_ylabel('Vector 2')
    axs[2].plot(x, vector2_corrected)
    axs[2].set_ylabel('Vector 2 Corrected')
    axs[2].set_xlabel('Time (s)')

    # Plot cross-correlation
    fig, ax = plt.subplots()
    ax.plot(np.arange(-len(phase_corr)//2, len(phase_corr)//2), phase_corr.real, linewidth=linewidth)
    ax.axvline(x=lag, color='r', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Phase-Correlation')

    plt.gca().spines['left'].set_linewidth(2.0)
    plt.gca().spines['bottom'].set_linewidth(2.0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if legend is not False:
        if legend == 'UR' or legend is True:
            fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        elif legend == 'UL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
        elif legend == 'BL':
            fig.legend(loc='upper left', bbox_to_anchor=(0, 0), bbox_transform=ax.transAxes)
        elif legend == 'BR':
            fig.legend(loc='upper left', bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)

    plt.show()
    return common.finish(data_aux)


@app.command()
def butterhp(cutoff: float, order: int, title: str = 'High Pass Filter', grid: bool = True, pngfile: str = None,
             legend: bool = False, linewidth: float = 1.0, show: bool = True):

    data = common.data_aux

    fs = 1.0 / common.config_get('ts', fail_on_missing=True)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, 'highpass')

    common.info("butterhp: fs=%0.3e fc=%0.3e nfc=%0.3e" % (fs, cutoff, normal_cutoff))

    fig, ax = plt.subplots(dpi=300)

    # Compute frequency response of the filter
    f, H = signal.freqz(b, a, fs=fs)

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

    if grid is True:
        common.info("enabling grid.")
        plt.grid(color='lightgray', alpha=0.5, zorder=1)

    fig.set_size_inches(13, 5)
    fig.gca()

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.plot(f, 20 * np.log10(abs(H)), linewidth=linewidth)
    ax.set_xscale('log')

    plt.gca().spines['left'].set_linewidth(2.0)
    plt.gca().spines['bottom'].set_linewidth(2.0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if legend is True:
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)
    if show:
        plt.show()
    return common.finish(data)
