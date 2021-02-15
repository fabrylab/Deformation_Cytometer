import matplotlib.pyplot as plt
import matplotlib
from Neural_Network.includes.data_handling import *
import pandas as pd

ylables_ref = {"strain_diff": "strain error", "recall": "recall", "precision": "precision"}
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 20)

cmap = matplotlib.cm.get_cmap("plasma")
line_props = dict(color="r", alpha=0.3)
flier_props = dict(marker="o", markersize=10)
plt.rcParams.update({'font.size': 20})


def set_limits(ax, cdb_files, networks, dist1, dist2, ylim=None):

    plot_lenght = dist2 * (len(np.unique(cdb_files)) - 1) + dist1 * len(np.unique(networks))
    ax.set_xlim(0 - plot_lenght * 0.1, plot_lenght * 1.1)
    if not ylim is None:
        ax.set_ylim(ylim)


def define_network_plot_position(networks):

    pos = {x: networks[networks == x].index[0] for x in np.unique(networks)}
    return pos


def define_db_plot_position(cdb_files, networks):

    cdb_file = {x: cdb_files[cdb_files == x].index[0] // len(np.unique(networks)) for x in np.unique(cdb_files)}
    return cdb_file


def setup_legend_handles(ax, pos, colors):

    # adding empty labels
    for net, i in sorted(list(pos.items()), key=lambda x: x[1]):
        ax.plot(np.NaN, np.NaN, '-', color=colors[i], label=net)


def set_plt_dist_color(pos, width, dist1):

    dist2 = (len(pos.keys()) + 1) * (width + dist1)
    colors = {i: cmap(v) for i, v in enumerate(np.linspace(0, 1, len(pos.keys())))}
    return dist2, colors


def unpack_values(results, err_type):

    GTmatches, SysMatches, networks, cdb_files, GT, Pred, error_values = results["GT Match"], results["Pred Match"], \
                                                                         results[
                                                                             "network"], results["database"], results[
                                                                             "GT"], results["Pred"], results[err_type]
    return GTmatches, SysMatches, networks, cdb_files, GT, Pred, error_values


def set_ticks(ax, cdb_files, pos, cdb_file, dist1, dist2, err_type, xlabel):
    # noting the number of matches and ground truth

    if xlabel:
        xticks_pos = np.array(list(cdb_file.values())) * dist2 + (dist1 * len(pos.keys())) / 2
        xticks = [os.path.split(x)[1] for x in list(np.unique(cdb_files))]
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(xticks)
    else:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax.set_ylabel(ylables_ref[err_type])


def make_bar_plot(results, err_type, ax, dist1=0.2, width=0.17, use_abs=True, xlabel=False, ylim=None):

    GTmatches, SysMatches, networks, cdb_files, GT, Pred, error_measure = unpack_values(results, err_type)
    pos = define_network_plot_position(networks)
    cdb_file = define_db_plot_position(cdb_files, networks)
    dist2, colors = set_plt_dist_color(pos, width, dist1)

    for gt, sys, gtm, sysm, net, cf, values in zip(GT, Pred, GTmatches, SysMatches, networks, cdb_files, error_measure):
        position = pos[net] * dist1 + cdb_file[cf] * dist2
        color = colors[pos[net]]
        if use_abs:
            values = np.abs(values)
        ax.bar([position], [np.mean(values)], width=width, color=color)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

    set_limits(ax, cdb_files, networks, dist1, dist2, ylim=ylim)
    # tick labels
    set_ticks(ax, cdb_files, pos, cdb_file, dist1, dist2,
              err_type, xlabel)
    return ax


def make_box_plot(results, err_type, ax, dist1=0.2, width=0.17, xlabel=False):

    GTmatches, SysMatches, networks, cdb_files, GT, Pred, error_values = unpack_values(results, err_type)
    pos = define_network_plot_position(networks)
    cdb_file = define_db_plot_position(cdb_files, networks)
    dist2, colors = set_plt_dist_color(pos, width, dist1)

    for gt, sys, gtm, sysm, net, cf, values in zip(GT, Pred, GTmatches, SysMatches, networks, cdb_files, error_values):
        positions = pos[net] * dist1 + cdb_file[cf] * dist2
        bbox_props = dict(color=colors[pos[net]], alpha=0.9)
        ax.boxplot([values], whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props,
                   positions=[positions])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        if len(values) == 0:
            ax.plot(positions, 0, "o", color=colors[pos[net]])
    ylim = np.max(np.abs(ax.get_ylim()))
    ylim = (-ylim, ylim)
    set_limits(ax, cdb_files, networks, dist1, dist2, ylim=ylim)
    ax.axhline(0, ls="--")
    # tick labels
    set_ticks(ax, cdb_files, pos, cdb_file, dist1, dist2,
              err_type, xlabel)

    return ax


def add_legend(results, ax, dist1=0.2, width=0.17):

    GTmatches, SysMatches, networks, cdb_files, GT, Pred, error_values = unpack_values(results, "strain_diff")
    pos = define_network_plot_position(networks)
    dist2, colors = set_plt_dist_color(pos, width, dist1)
    setup_legend_handles(ax, pos, colors)  # adding empty labels
    set_limits(ax, cdb_files, networks, dist1, dist2, ylim=(0, 0.8))
    ax.legend(bbox_to_anchor=(0.5, 1.3), loc="upper center", ncol=len(pos) // 3)
    ax.axis("off")
    return ax


def add_detect_numbers(results, ax, dist1, width):

    GTmatches, SysMatches, networks, cdb_files, GT, Pred, error_values = unpack_values(results, "strain_diff")
    pos = define_network_plot_position(networks)
    cdb_file = define_db_plot_position(cdb_files, networks)
    dist2, colors = set_plt_dist_color(pos, width, dist1)
    set_limits(ax, cdb_files, networks, dist1, dist2, ylim=(0, 0.3))
    y_high = ax.get_ylim()[1]
    for gtm, sysm, net, cf, gts in zip(GTmatches, SysMatches, networks, cdb_files, GT):
        positions = pos[net] * dist1 + cdb_file[cf] * dist2
        ax.text(positions, y_high * 0, str(len(sysm)) + " | " + str(len(gts)), rotation="70")
    ax.axis("off")


def plot_results(res, out_folder, name, width=0.2, dist=0.6, use_pylustrator=False):

    n_db = len(np.unique(res["network"]))
    n_network = len(np.unique(res["database"]))
    if use_pylustrator:
        # whole plot needs to be a bit smaller to comfortably fit into the pylustrator window
        plt.rcParams.update({'font.size': 16})
        figsize = (30 * n_network * width * dist * n_db, 7.5)
    else:
        figsize = (40 * n_network * width * dist * n_db, 10)
    fig, axs = plt.subplots(6, 1, figsize=figsize)
    add_legend(res, axs[0], dist1=0.2, width=0.17)
    add_detect_numbers(res, axs[1], dist1=0.2, width=0.17)

    make_bar_plot(res, "recall", axs[2], dist1=dist, width=width, ylim=(0, 1))
    make_bar_plot(res, "precision", axs[3], dist1=dist, width=width, ylim=(0, 1))
    make_bar_plot(res, "strain_diff", axs[4], dist1=dist, width=width, use_abs=True)
    make_box_plot(res, "strain_diff", axs[5], dist1=dist, width=width, xlabel=True)
    plt.tight_layout()
    if not use_pylustrator:
        plt.savefig(os.path.join(out_folder, name))
    return fig
