# Amplitude.py
from config import Paths, Colors, Filters, Subplots
from data_manager import *
import os
import sqlite3
import pandas
import math
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import figure
from matplotlib.backends.backend_pdf import PdfPages
import statistics

def plot_amplitude(datafile):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    amplitudes = {}
    (chan1, chan2) = determine_active_channels(datafile)
    connection = sqlite3.connect(datafile)
    for channel in (chan1, chan2):
        data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
        data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
        amplitude_data = data['Amplitude (V)']
        t_50_data = data['t_50 (s)']
        t_90_data = data["t_90 (s)"]
        time_over_90_data = data['Time over 90% (s)']
        amplitudes[channel] = []
        for i in range(n_position):
            for j in range(n_triggers):
                amplitude = amplitude_data[i,j,1]
                if math.isnan(amplitude) or amplitude > 0: # HARDCODED AMPLITUDE ??
                    continue
                time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
                if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                    continue
                peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
                    continue
                amplitudes[channel].append(amplitude)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax1, ax2 = axes.flatten()
    subplots = (ax1, ax2)
    plot = 0 
    for channel in (chan1, chan2):
        (n, bins, patches) = subplots[plot].hist(amplitudes[channel], bins=100, density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Channel {channel}", color=Colors.CB_CYCLE[plot])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        # mu, std = statistics.mean(amplitudes[channel]), statistics.stdev(amplitudes[channel])
        # (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
        fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
        # fit_y_axis = gaussian(fit_x_axis, muf, stdf)
        # subplots[plot].plot(fit_x_axis, fit_y_axis, color = "r")
        subplots[plot].axvline(x=Filters.AMPLITUDE_THRESHOLD, color='k', linestyle='--', label='Amplitude Threshold')
        subplots[plot].set_xlabel(f"Amplitude (V)")
        subplots[plot].set_ylabel(f"Frequency")
        subplots[plot].legend(loc = "upper left")
        plot += 1
    fig.suptitle(f'{datafile[48:57]}, {datafile[58:62]}')
    plt.show()
    return None

def plot_noise(datafile):
    active_channels = determine_active_channels(datafile)
    noise = {}

    connection = sqlite3.connect(datafile)
    for channel in active_channels:
        data = pandas.read_sql(
            f"""
            SELECT n_position, n_trigger, `Noise (V)`
            FROM dataframe_table
            WHERE n_pulse = 1 AND n_channel = {channel}
            """,
            connection
        )
        data.set_index(['n_position', 'n_trigger'], inplace=True)
        noise[channel] = data['Noise (V)'].to_numpy()
    connection.close()

    plt.figure(figsize=(8, 6))
    for channel in active_channels:
        values = noise[channel]
        values = values[numpy.isfinite(values)]
        if len(values) < 10:
            continue
        mu = numpy.median(values)
        std = numpy.std(values)
        if std <= 0:
            continue
        bin_min = mu - 4 * std
        bin_max = mu + 4 * std
        n_bins = 50
        bins = numpy.linspace(bin_min, bin_max, n_bins)
        plt.hist(
            values,
            bins=bins,
            histtype='step',
            lw=1.5,
            density=True,
            label=f"Channel {channel}",
            color=Colors.CB_CYCLE[channel]
        )
    plt.xlabel(r"Noise (V)")
    plt.ylabel("Frequency")
    plt.title(f"Noise distribution — {datafile[48:57]}, {datafile[58:62]}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_amplitude_against_t_peak(datafile, time_variable = "t_90 (s)"):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    result = {} #channel: ([time],[amplitude])
    active_channels = determine_active_channels(datafile)
    connection = sqlite3.connect(datafile)
    for channel in active_channels:
        data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `{time_variable}`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)`,`Noise (V)` FROM dataframe_table WHERE n_channel=={channel}", connection)
        data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
        time_data = data[f"{time_variable}"]
        amplitude_data = data['Amplitude (V)']
        t_50_data = data['t_50 (s)']
        time_over_90_data = data['Time over 90% (s)']

        result[channel] = ([],[])
        for i in range(n_position):
            for j in range(n_triggers):
                if math.isnan(amplitude_data[i,j,1]):
                    continue
                if amplitude_data[i,j,1] > 0: # HARDCODED AMPLITUDE ??
                    continue
                time_diff = t_50_data[i,j,2] - t_50_data[i,j,1]
                if time_diff < Filters.TIME_DIFF_MIN * 1e-9 or time_diff > Filters.TIME_DIFF_MAX * 1e-9:
                    continue
                time = (time_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                amplitude = amplitude_data[i,j,1]
                result[channel][0].append(time)
                result[channel][1].append(amplitude)
    connection.close()

    fig, axes = plt.subplots(ncols=2, figsize=(12,5))
    ax1, ax2 = axes.flatten()
    subplots = (ax1, ax2)
    plot=0
    for channel in active_channels:
        subplots[plot].plot(result[channel][0], result[channel][1], ".", label=f'Channel {channel}', markersize=2, color=Colors.CB_CYCLE[plot])
        subplots[plot].set_xlim(4,9)
        subplots[plot].set_xlabel(f"Peak Time (ns)")
        subplots[plot].set_ylabel(f"Amplitude (V)")
        subplots[plot].axvline(x=Filters.PEAK_TIME_MIN, color='red', linestyle='--', label='Peak Time Min')
        subplots[plot].axvline(x=Filters.PEAK_TIME_MAX, color='blue', linestyle='--', label='Peak Time Max')
        subplots[plot].legend(loc = "best")
        plot += 1

    fig.suptitle(f'Amplitude against Peak Time, {datafile[48:57]}, {datafile[58:62]}')
    plt.show()
    return None

def plot_amplitude_of_one_pad(datafile, channel, pad_positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    amplitudes = []
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    for i in range(n_position):
        if i not in pad_positions:
            continue
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude) or amplitude > Filters.AMPLITUDE_THRESHOLD: 
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
                continue
            amplitudes.append(amplitude)

    mu, std = statistics.mean(amplitudes), statistics.stdev(amplitudes)
    bin_min = mu - 4 * std; bin_max = mu + 4 * std
    custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
    (n, bins, patches) = plt.hist(amplitudes, bins="auto", density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
    fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
    fit_y_axis = gaussian(fit_x_axis, muf, stdf)
    if Subplots.AMPLITUDE_ONE_PAD:
        plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
        plt.xlabel(f"Amplitude (V)")
        plt.ylabel(f"Frequency")
        plt.legend(loc = "best")
        plt.title(f'{datafile[48:57]}, {datafile[58:62]}, Channel {channel}')
        # plt.show()
    return (muf, stdf)


def plot_amplitude_everything(directory_in_str = "Data/"):
    final_plot = {} # {sensor: channel: ([voltages], [mean amplitude], [std amplitude (error)])}
    with PdfPages(f"amplitude.pdf") as pdf:
        directory = os.fsencode(directory_in_str)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.startswith('.'): #    if filename.startswith('.'):   # to ignore any hidden file
                continue
            directory2_in_str = f"{directory_in_str}{filename}/"
            directory2 = os.fsencode(directory2_in_str)

            final_plot[filename] = {}
            
            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                print(folder)
                filename2 = os.fsdecode(folder)
                if filename2.startswith('.'): #      # to ignore any hidden file
                    continue                
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                if filename2 in ("50V","70V","80V","90V","100V","110V","120V","130V", "140V", "150V", "160V", "170V", "180V", "190V"):
                    data_file       = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                    positions_file  = f"{directory3_in_str}positions.pickle"
                    (chan1, chan2) = determine_active_channels(data_file)
                    pad_positions = {}
                    pad_positions[chan1] = get_pad_positions(data_file, positions_file, chan1)
                    pad_positions[chan2] = get_pad_positions(data_file, positions_file, chan2)
                    break

            final_plot[filename][chan1] = ([],[],[]); final_plot[filename][chan2] = ([],[],[])

            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                print(folder)                
                filename2 = os.fsdecode(folder)
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file         = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2       = f"{directory3_in_str}measured_data.sqlite"
                positions_file    = f"{directory3_in_str}positions.pickle"
                print(f"{filename}, {filename2}")
                (voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)

                plt.clf()
                (muf, stdf) = plot_amplitude_of_one_pad(data_file, chan1, pad_positions[chan1])
                final_plot[filename][chan1][0].append(voltage)
                final_plot[filename][chan1][1].append(abs(muf)) # take absolute value of amplitude
                final_plot[filename][chan1][2].append(stdf)
                if Subplots.AMPLITUDE_ONE_PAD:
                    plt.title(f'{filename}, {filename2}, Channel {chan1}')
                    fig = plt.gcf()
                    pdf.savefig(fig, dpi = 100)
                    plt.clf()

                (muf, stdf) = plot_amplitude_of_one_pad(data_file, chan2, pad_positions[chan2])
                final_plot[filename][chan2][0].append(voltage)
                final_plot[filename][chan2][1].append(abs(muf)) # take absolute value of amplitude
                final_plot[filename][chan2][2].append(stdf)
                if Subplots.AMPLITUDE_ONE_PAD:
                    plt.title(f'{filename}, {filename2}, Channel {chan2}')
                    fig = plt.gcf()
                    pdf.savefig(fig, dpi = 100)
                    plt.clf()
                
        plt.clf()
        color_counter = 0
        linestyle_counter = 0
        for sensor in final_plot:
            for channel in final_plot[sensor]:
                x_axis = final_plot[sensor][channel][0]
                y_axis = final_plot[sensor][channel][1]
                y_err  = final_plot[sensor][channel][2]
                if linestyle_counter % 2 == 0:
                    linestyle = "-"
                else:
                    linestyle = "--"
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = Colors.CB_CYCLE[color_counter], label = f"{sensor}, Ch {channel}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        plt.gca().invert_xaxis()
        plt.title("Mean amplitude against Bias Voltage")
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Amplitude (V)")
        plt.legend(loc = "best")
        fig = plt.gcf()
        pdf.savefig(fig, dpi = 100)
        save_results(final_plot, analysis="Amplitude")
    return final_plot

def plot_2D_separate(datafile, positions): 
    # makes 2D plot of each channel, and projection of one edge
    n_position, n_triggers, n_channels = query_dataset(datafile)
    (x,y) = get_positions(positions)
    (active_channel_1, active_channel_2) = determine_active_channels(datafile)
    amplitudes = {}
    connection = sqlite3.connect(datafile)
    for channel in (active_channel_1, active_channel_2):
        amplitudes[channel] = []
        data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
        data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
        amplitude_data = data['Amplitude (V)']
        t_50_data = data['t_50 (s)']
        t_90_data = data["t_90 (s)"]
        time_over_90_data = data['Time over 90% (s)']
        for i in range(n_position):
            result = []
            for j in range(n_triggers):
                amplitude = amplitude_data[i,j,1]
                if math.isnan(amplitude) or amplitude > Filters.AMPLITUDE_THRESHOLD:
                    continue
                time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
                if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                    continue
                peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
                    continue
                result.append(amplitude)
            if result == []:
                amplitudes[channel].append(0)
            else:
                amplitudes[channel].append(statistics.mean(result))
        
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': amplitudes[1]})
    heatmap = data_frame.pivot_table(index='x', columns='y', values='z', aggfunc='mean')
    sns.heatmap(heatmap, annot=False, cmap="plasma_r", ax=ax2, square=True, yticklabels = 5, xticklabels = 5) #vmax=0
    ax2.set_title("Channel 1")
    ax2.set_ylabel(r"y ($\mu$m)")
    ax2.set_xlabel(r"x ($\mu$m)")
    ax2.invert_xaxis()

    # list of y edges
    smallest_list = (pandas.Series(data_frame['y'].unique()).nsmallest(10)).to_list()
    largest_list  = (pandas.Series(data_frame['y'].unique()).nlargest(10)).to_list()
    # select rows
    edge1 = data_frame[data_frame["y"].isin(smallest_list)]
    edge2 = data_frame[data_frame["y"].isin(largest_list)]
    # remove zero amplitude values ----
    # edge1 = edge1[edge1["z"].abs() > 1e-6]
    # edge2 = edge2[edge2["z"].abs() > 1e-6]
    # projection
    edge1_projection     = edge1.groupby('x')['z'].mean()
    edge1_projection_sig = edge1.groupby('x')['z'].std()
    edge2_projection     = edge2.groupby('x')['z'].mean()
    edge2_projection_sig = edge2.groupby('x')['z'].std()
    # convert to numpy arrays
    edge1_amplitudes = edge1_projection.to_numpy()
    edge2_amplitudes = edge2_projection.to_numpy()
    # pick correct side ??
    if numpy.mean(edge1_amplitudes) < numpy.mean(edge2_amplitudes):
        x_axis = edge1_projection.index.to_numpy()
        y_axis = edge1_projection.to_numpy()
        err    = edge1_projection_sig.to_numpy()
    else:
        x_axis = edge2_projection.index.to_numpy()
        y_axis = edge2_projection.to_numpy()
        err    = edge2_projection_sig.to_numpy()
    ax1.plot(y_axis, x_axis, marker='.', linestyle='-', color=Colors.CB_CYCLE[0], label=f"Channel {0}", markersize=5)
    ax1.invert_yaxis()
    ax1.set_ylabel(r"y ($\mu$m)")
    ax1.set_xlabel(r"Mean Amplitude (V)")
    ax1.set_title("Channel 1")
    ax1.errorbar(y_axis, x_axis, xerr = err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)

    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': amplitudes[2]})
    heatmap = data_frame.pivot_table(index='x', columns='y', values='z', aggfunc='mean')
    sns.heatmap(heatmap, annot=False, cmap="plasma_r", ax=ax4, square=True, yticklabels = 5, xticklabels = 5)
    ax4.set_title("Channel 2")
    ax4.set_ylabel(r"y ($\mu$m)")
    ax4.set_xlabel(r"x ($\mu$m)")
    ax4.invert_xaxis()
    
    smallest_list = (pandas.Series(data_frame['y'].unique()).nsmallest(10)).to_list()
    largest_list = (pandas.Series(data_frame['y'].unique()).nlargest(10)).to_list()
    edge1 = data_frame[data_frame["y"].isin(smallest_list)]
    edge2 = data_frame[data_frame["y"].isin(largest_list)]
    edge1_projection = edge1.groupby(f'x')['z'].mean(); edge1_projection_sig = edge1.groupby(f'x')['z'].std()
    edge2_projection = edge2.groupby(f'x')['z'].mean(); edge2_projection_sig = edge2.groupby(f'x')['z'].std()
    edge1_amplitudes = edge1_projection.to_numpy()
    edge2_amplitudes = edge2_projection.to_numpy()
    if numpy.mean(edge1_amplitudes) < numpy.mean(edge2_amplitudes):
        x_axis = edge1_projection.index.to_numpy()
        y_axis = edge1_projection.to_numpy()
        err = edge1_projection_sig.to_numpy()
    else:
        x_axis = edge2_projection.index.to_numpy()
        y_axis = edge2_projection.to_numpy()
        err = edge2_projection_sig.to_numpy()
    ax3.plot(y_axis, x_axis, marker='.', linestyle='-', color=Colors.CB_CYCLE[0], label=f"Channel {0}", markersize=5)
    ax3.invert_yaxis()
    ax3.set_ylabel(r"y ($\mu$m)")
    ax3.set_xlabel(r"Mean Amplitude (V)")
    ax3.set_title("Channel 2")
    ax3.errorbar(y_axis, x_axis, xerr = err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
    fig.suptitle(f'{datafile[5:11]}, {datafile[12:16]}')
    plt.tight_layout()
    plt.show()
    return None               

def project_onto_y_one_channel_amplitude(datafile, positions, channel, sensor_strip_positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    (x,y) = get_positions(positions)
    amplitudes = {} # {y position: [list of amplitudes]}
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    # all_amps = amplitude_data.values
    # min_val = numpy.nanmin(all_amps)
    # max_val = numpy.nanmax(all_amps)

    for i in sensor_strip_positions:
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude) or amplitude > 0.02: 
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
                continue

            if y[i] not in amplitudes:
                amplitudes[y[i]] = []
            #norm_amplitude = (amplitude - min_val) / (max_val - min_val)
            amplitudes[y[i]].append(amplitude)

    result = {"x axis": [], "y axis": [], "y error": []}

    for y_position in sorted(amplitudes):
        hist = amplitudes[y_position]
        if len(hist) == 1: # toss
            continue
        mu, std = statistics.mean(hist), statistics.stdev(hist)

        if len(hist) < 10: # no fit if less than this amount of points
            # result["x axis"].append(y_position)
            # result["y axis"].append(abs(mu))
            # result["y error"].append(std) 
            continue 

        bin_min = mu - 4 * std; bin_max = mu + 4 * std
        bin_width = 0.0005 # hardcoded bin width
        n_bins = round( (bin_max - bin_min) / bin_width )
        custom_bins = numpy.linspace(bin_min, bin_max, n_bins ,endpoint=True)
        (n, bins, patches) = plt.hist(hist, bins=custom_bins, density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
        result["x axis"].append(y_position)
        result["y axis"].append(abs(muf))
        result["y error"].append(stdf)  

        # plt.title(f"Histogram amplitudes — y = {y_position}")
        # plt.xlabel("Amplitude")
        # plt.ylabel("Frequency")
        # plt.legend()
        # fig = plt.gcf()
        # plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
        # pdf.savefig(fig, dpi = 100)
        # plt.show()
    return result

def plot_amplitude_along_y_axis(datafile, positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    (x, y) = get_positions(positions)
    print(x, y)
    (active_ch1, active_ch2) = determine_active_channels(datafile)

    ampl = {active_ch1: [], active_ch2: []}
    charge = {active_ch1: [], active_ch2: []}

    connection = sqlite3.connect(datafile)

    for channel in (active_ch1, active_ch2):
        df = pandas.read_sql(
            f"""SELECT n_position,n_trigger,n_pulse, 
                    `t_90 (s)`, `Time over 90% (s)`,
                    `Amplitude (V)`, `Collected charge (V s)`, `t_50 (s)` 
                FROM dataframe_table 
                WHERE n_channel=={channel}""",
            connection
        )
        df.set_index(['n_position','n_trigger','n_pulse'], inplace=True)

        amplitude = df['Amplitude (V)']
        collected = df['Collected charge (V s)']
        t50 = df['t_50 (s)']
        t90 = df['t_90 (s)']
        dt90 = df['Time over 90% (s)']
        x_positions = []

        for i in range(n_position):
            A, C = [], []
            if x[i] != InterpadConfig.Y_POSITION_MID_PAD:
                continue
            x_positions.append(y[i])

            for j in range(n_triggers):
                amp = amplitude.get((i,j,1), numpy.nan)
                chg = collected.get((i,j,1), numpy.nan)
                # filtres
                if math.isnan(amp) or amp > -0.00:
                    continue
                if math.isnan(chg) or chg > -0.00:
                    continue

                td = (t50[i, j, 2] - t50[i, j, 1]) * 1e9
                if td < Filters.TIME_DIFF_MIN or td > Filters.TIME_DIFF_MAX:
                    continue

                peak = (t90[i,j,1] + 0.5*dt90[i,j,1]) * 1e9
                if peak < Filters.PEAK_TIME_MIN or peak > Filters.PEAK_TIME_MAX:
                    continue

                A.append(amp)
                C.append(chg)

            ampl[channel].append(statistics.mean(A) if A else 0)
            charge[channel].append(statistics.mean(C) if C else 0)
    connection.close()
    fig, axes = plt.subplots(ncols=2, figsize=(12,5))
    ax1, ax2 = axes.flatten()

    for channel in (active_ch1, active_ch2):

        # --- NORMALISATION --- uncomment if needed
        # ampl[channel] = ampl[channel] / numpy.max(numpy.abs(ampl[channel]))
        # charge[channel] = charge[channel] / numpy.max(numpy.abs(charge[channel]))

        # --- TRI DES DONNÉES ---
        zipped = list(zip(x_positions, ampl[channel], charge[channel]))
        zipped_sorted = sorted(zipped, key=lambda t: t[0])  # tri selon Y

        y_sorted, ampl_sorted, charge_sorted = zip(*zipped_sorted)

        # --- PLOTS TRIÉS ---
        ax1.plot(
            y_sorted, numpy.abs(ampl_sorted),
            marker='o', linestyle='-', markersize=3,
            label=f'Channel {channel}', color=Colors.CB_CYCLE[channel]
        )

        ax2.plot(
            y_sorted, numpy.abs(charge_sorted),
            marker='o', linestyle='-', markersize=3,
            label=f'Channel {channel}', color=Colors.CB_CYCLE[channel]
        )

    ax1.set_xlabel("Y Position")
    ax1.set_ylabel("Amplitude (V)")
    ax1.set_title(f"Amplitude for Y = {InterpadConfig.Y_POSITION_MID_PAD}")
    ax1.legend(loc="best")

    ax2.set_xlabel("Y Position")
    ax2.set_ylabel("Collected Charge (V s)")
    ax2.set_title(f"Collected Charge for Y = {InterpadConfig.Y_POSITION_MID_PAD}")
    ax2.legend(loc="best")

    fig.suptitle(f'Amplitude and Collected Charge projected on X axis, {datafile[48:57]}, {datafile[58:62]}')
    plt.tight_layout()
    plt.show()

    return None


