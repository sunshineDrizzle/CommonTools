import numpy as np

from scipy.stats import ttest_ind, sem
from matplotlib import pyplot as plt
from commontool.io.io import CsvReader
from commontool.algorithm.plot import auto_bar_width


def calc_mean_sem(samples, output_file, sample_names=None):
    """
    calculate mean and sem for each sample

    :param samples: sequence
        a sequence of samples
    :param output_file: str
    :param sample_names: sequence
        a sequence of sample names
    """
    if sample_names is None:
        sample_names = list(map(str, range(1, len(samples)+1)))
    else:
        assert len(samples) == len(sample_names)
    sample_names.insert(0, 'sample_name')

    means = ['mean']
    sems = ['sem']
    for sample in samples:
        means.append(str(np.nanmean(sample)))
        sems.append(str(sem(sample)))

    lines = list()
    lines.append(','.join(sample_names))
    lines.append(','.join(means))
    lines.append(','.join(sems))
    open(output_file, 'w+').writelines('\n'.join(lines))


def ttest_ind_pairwise(samples1, samples2, output_file, sample_names=None):
    """
    Do two sample t test pairwise between samples1 and samples2

    :param samples1: sequence
        a sequence of samples
    :param samples2: sequence
        a sequence of samples
    :param output_file: str
    :param sample_names: sequence
        a sequence of sample names
    """
    assert len(samples1) == len(samples2)
    sample_num = len(samples1)
    if sample_names is None:
        sample_names = list(map(str, range(1, sample_num+1)))
    else:
        assert len(sample_names) == sample_num
    sample_names.insert(0, 'sample_name')

    ts = ['t']
    ps = ['p']
    for idx in range(sample_num):
        sample1 = samples1[idx]
        sample2 = samples2[idx]
        t, p = ttest_ind(sample1, sample2)
        ts.append(str(t))
        ps.append(str(p))

    lines = list()
    lines.append(','.join(sample_names))
    lines.append(','.join(ts))
    lines.append(','.join(ps))
    open(output_file, 'w+').writelines('\n'.join(lines))


def plot_mean_sem(mean_sem_files, items=None, sample_names=None, xlabel='', ylabel=''):
    """

    :param mean_sem_files: sequence
        a sequence of file paths which are generated from 'calc_mean_sem'
    :param items: sequence
        a sequence of item names corresponding to the 'mean_sem_files'
    :param sample_names: collection
        a collection of sample names of interested
    :param xlabel: str
    :param ylabel: str

    :returns: fig, ax
    """
    fig, ax = plt.subplots()
    x = None
    width = None
    rects_list = []
    item_num = len(mean_sem_files)
    for idx, mean_sem_file in enumerate(mean_sem_files):
        mean_sem_dict = CsvReader(mean_sem_file).to_dict(1)
        if sample_names is None:
            sample_names = mean_sem_dict['sample_name']
        if x is None:
            x = np.arange(len(sample_names))
            width = auto_bar_width(x, item_num)
        y = [float(mean_sem_dict['mean'][mean_sem_dict['sample_name'].index(i)]) for i in sample_names]
        sems = [float(mean_sem_dict['sem'][mean_sem_dict['sample_name'].index(i)]) for i in sample_names]
        rects = ax.bar(x+width*idx, y, width, color='k', alpha=1./((idx+1)/2+0.5), yerr=sems)
        rects_list.append(rects)
    if items is not None:
        assert item_num == len(items)
        ax.legend(rects_list, items)
    ax.set_xticks(x+width/2.0*(item_num-1))
    ax.set_xticklabels(sample_names)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')

    plt.tight_layout()
    return fig, ax


def plot_compare(ps, sample_names, ts=None, title=''):
    """

    :param ps: sequence
        a sequence of p values
    :param sample_names: sequence
        a sequence of sample names corresponding to the 'ps'
    :param ts: sequence
        a sequence of t values corresponding to the 'ps'
    :param title: str

    :returns: fig, ax
        when 'ts' is None
    :returns: fig, ax, ax_twin
        when 'ts' is not None
    """
    sample_num = len(sample_names)
    assert sample_num != 0
    assert len(ps) == sample_num

    fig, ax = plt.subplots()
    x = np.arange(sample_num)
    width = auto_bar_width(x)
    rects_p = ax.bar(x, ps, width, color='g', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    ax.set_ylabel('p', color='g')
    ax.tick_params('y', colors='g')
    ax.axhline(0.05)
    ax.axhline(0.01)
    ax.axhline(0.001)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')

    if ts is not None:
        assert len(ts) == sample_num
        ax_twin = ax.twinx()
        rects_t = ax_twin.bar(x, ts, width, color='b', alpha=0.5)
        ax_twin.legend([rects_t, rects_p], ['t', 'p'])
        ax_twin.set_ylabel('t', color='b')
        ax_twin.tick_params('y', colors='b')
        return fig, ax, ax_twin

    plt.tight_layout()
    return fig, ax
