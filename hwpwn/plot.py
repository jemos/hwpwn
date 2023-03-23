import numpy as np
import typer
from matplotlib import pyplot as plt, cm

from . import common

app = typer.Typer(callback=common.default_typer_callback)


@app.command()
def time(title: str = None, grid: bool = True, pngfile: str = None, legend: bool = True, alpha: float = 1.0,
         ylabel: str = None, yscale: float = 1.0, xunit: str = 'NA', yunit: str = 'NA', xlabel: str = 'Time',
         linewidth: float = 1.0):
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
    #fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=None, hspace=None)
    fig.tight_layout()

    ax.set_xlabel(f'{xlabel} [{xunit}]')
    ax.set_ylabel(f'{ylabel} [{yunit}]')
    if len(data['triggers']):
        ax_trig = ax.twinx()
        ax_trig.set_ylabel('V')

    for s in data['signals']:
        common.info(f"plotting signal {s['name']} ...")
        c = next(pltcolor)
        ax.plot(data['x_axis'], np.multiply(s['vector'], yscale), label=s['name'], c=c, alpha=alpha,
                linewidth=linewidth)
        # if 'markers' in s and len(s['markers']):
        # ax.plot(s['markers_x'], np.multiply(s['markers'], yscale), label=s['name'], marker='o', c=c, linestyle='None')

    for s in data['triggers']:
        common.info(f"plotting trigger signal {s['name']} ...")
        c = next(pltcolor)
        ax_trig.plot(data['x_axis'], np.multiply(s['vector'], 1), label=s['name'], c=c, alpha=1.0, linewidth=linewidth)

    if legend is True:
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)

    plt.gca().spines['left'].set_linewidth(2.0)
    plt.gca().spines['bottom'].set_linewidth(2.0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    return common.finish(data)


def xy(title: str = None, grid: bool = False, pngfile: str = None, legend: bool = False, alpha: float = 1.0,
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


def freq(title: str = None, grid: bool = False, pngfile: str = None, legend: bool = True,
         legend_location: str = 'upper right'):
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
    fig.set_size_inches(13, 5)
    fig.gca()

    if grid is True:
        common.info("enabling grid.")
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
        fig.legend(loc=legend_location, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    if title is not None:
        plt.title(title)
    if pngfile is not None:
        plt.savefig(pngfile)
    plt.show()
    return common.finish(data)
