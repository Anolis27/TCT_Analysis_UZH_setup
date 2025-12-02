import numpy
import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy import stats
import sqlite3
import pandas
import pickle
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import seaborn as sns
import math
import statistics
import os
import time
import PyPDF2

# from natsort import natsorted

CB_color_cycle = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
    '#984ea3', '#999999', '#e41a1c', '#dede00',
    '#56B4E9', 
    '#E69F00', 
    '#009E73', 
    '#F0E442', 
    '#0072B2', 
    '#CC79A7'  
]
colors = ["Red","Green","Blue","Yellow","Cyan","Magenta","Orange","Purple","Pink","Brown","Black","White","Gray","DarkRed","DarkGreen","DarkBlue","LightGray","LightGreen","LightBlue","LightCoral"]

######## FILES AND DIRECTORIES ########
test_directory = os.path.expanduser("C:/Users/mathi/Documents/UZH/LGAD_ANALYSIS/Data/V2_TW5/50V")
test_datafile  = os.path.abspath(f"{test_directory}/parsed_from_waveforms.sqlite")
test_datafile2 = os.path.abspath(f"{test_directory}/measured_data.sqlite")
test_positions = os.path.abspath(f"{test_directory}/positions.pickle")

######## FILTERS GLOBAL PARAMETERS ########
AMPLITUDE_THRESHOLD = -0.08  # V -0.07
TIME_DIFF_MIN = 98         # ns 98
TIME_DIFF_MAX = 99.5       # ns 99.5
PEAK_TIME_MIN = 4.5       # ns 4.5
PEAK_TIME_MAX = 6.5       # ns 6.5
INTERPAD_REGION_MIN = -125         # um -75
INTERPAD_REGION_MAX = 125
        # um 75
###########################################

def sigmoid(x, x0, b, L, k):
    y = L / (1 + numpy.exp(-k*(x-x0))) + b
    return (y)

def gaussian(x, mu, sig):
    return 1./(numpy.sqrt(2.*numpy.pi)*sig)*numpy.exp(-numpy.power((x - mu)/sig, 2.)/2)

def query_dataset(datafile):
    #print(f"Querying dataset...")
    global n_position; global n_triggers; global n_channels
    connection = sqlite3.connect(datafile)
    (chan1, chan2) = determine_active_channels(datafile)
    query = f"SELECT n_position FROM dataframe_table WHERE n_trigger = 0 and n_pulse = 1 and n_channel = {chan1}"
    filtered_data = pandas.read_sql_query(query, connection)
    n_position = len(filtered_data)
    query = f"SELECT n_trigger FROM dataframe_table WHERE n_position = 0 and n_pulse = 1 and n_channel = {chan1}"
    filtered_data = pandas.read_sql_query(query, connection)
    n_triggers = len(filtered_data)
    query = "SELECT n_channel FROM dataframe_table WHERE n_position = 0 and n_pulse = 1 and n_trigger = 0"
    filtered_data = pandas.read_sql_query(query, connection)
    n_channels = len(filtered_data)
    connection.close()
    #print(f"{n_position} positions, {n_triggers} triggers and {n_channels} channels found")
    return None

def get_positions(positions):
    # get the (x,y) positions from the saved data
    # print(f"Getting position data...")
    positions_data = pandas.read_pickle(positions)
    positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
    for _ in {'x','y'}: # remove offset so (0,0) is the center
        positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()    
    x_data = positions_data['x (m)']
    y_data = positions_data['y (m)']
    x = []; y = []
    for i in range(n_position):
        x.append(round(x_data[i]*1e6))
        y.append(round(y_data[i]*1e6))
    return (x,y)

def get_bias_voltage(datafile):
    connection = sqlite3.connect(datafile)
    query = "SELECT `Bias voltage (V)` FROM dataframe_table "
    filtered_data = pandas.read_sql_query(query, connection)
    connection.close()
    voltages = filtered_data.to_numpy()
    # Calculate average voltage and round to nearest 5V
    voltage = 5 * round((sum(voltages)/len(voltages))[0] / 5)
    # Take uncertainty to be the diff
    uncertainty = abs(max(voltages) - min(voltages))[0] / 2
    if uncertainty > 5:
        print(f"Warning: Uncertainty of bias voltage is large, potential problem with power supply")
        print(f"Voltage: {voltage}, Uncertainty: {uncertainty}")
    return (voltage, uncertainty)

def get_channel_amplitude(datafile, channel, pulse_no = 1):
    query_dataset(datafile)
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    amplitudes = []
    for i in range(n_position):
        result = []
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD:
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                continue
            result.append(amplitude)
        if result == []:
            amplitudes.append(0)
        else:
            amplitudes.append(statistics.mean(result))
    return amplitudes

# Hardcoded to return channels 1 and 2 for now
# def determine_active_channels(datafile):
##    query_dataset(datafile)
##    result = {}; list_to_sort = []
##    for channel in range(1, n_channels + 1):
##        amplitudes = get_channel_amplitude(datafile, channel, pulse_no = 1, method = "median")
##        result[round(sum(amplitudes),3)] = channel
##        list_to_sort.append(round(sum(amplitudes),3))
##        list_to_sort = sorted(list_to_sort)
##    return tuple(sorted((result[list_to_sort[0]], result[list_to_sort[1]])))
    # return (3,4)

def determine_active_channels(datafile):
    connection = sqlite3.connect(datafile)
    query = "SELECT n_channel, `Amplitude (V)` FROM dataframe_table"
    df = pandas.read_sql_query(query, connection)
    connection.close()
    if df.empty:
        return ()
    df["AmpAbs"] = df["Amplitude (V)"].abs()
    pairs = {
        (1, 2): None,
        (3, 4): None,
    }
    mean_values = {}

    for pair in pairs:
        chA, chB = pair

        df_pair = df[df["n_channel"].isin(pair)]

        if df_pair.empty:
            mean_values[pair] = 0
        else:
            mean_values[pair] = df_pair["AmpAbs"].mean()
    # chose the pair with the highest mean amplitude
    best_pair = max(mean_values, key=mean_values.get)
    if mean_values[best_pair] == 0:
        return ()
    return best_pair

def plot_amplitude(datafile, positions):
    query_dataset(datafile)
    amplitudes = {}
    (chan1, chan2) = determine_active_channels(datafile)
    connection = sqlite3.connect(datafile)
    for channel in (chan1, chan2):
        pad_positions = get_pad_positions(datafile, positions, channel)
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
                if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                    continue
                peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                    continue
                amplitudes[channel].append(amplitude)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax1, ax2 = axes.flatten()
    subplots = (ax1, ax2)
    for channel in (chan1, chan2):
        (n, bins, patches) = subplots[channel - 1].hist(amplitudes[channel], bins=100, density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Channel {channel}")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mu, std = statistics.mean(amplitudes[channel]), statistics.stdev(amplitudes[channel])
        (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
        fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
        fit_y_axis = gaussian(fit_x_axis, muf, stdf)
        subplots[channel - 1].plot(fit_x_axis, fit_y_axis, color = "r")
        subplots[channel - 1].set_xlabel(f"Amplitude (V)")
        subplots[channel - 1].set_ylabel(f"Number")
        subplots[channel - 1].legend(loc = "upper left")
    fig.suptitle(f'{datafile[5:11]}, {datafile[12:16]}')
    plt.show()
    return None

def plot_collected_charge(datafile):
    query_dataset(datafile)
    collected_charges = {}
    connection = sqlite3.connect(datafile)
    for channel in range(1, n_channels + 1):
        data = pandas.read_sql(f"SELECT n_position,n_trigger,`Collected charge (V s)` FROM dataframe_table WHERE n_pulse==1 and n_channel=={channel}", connection)
        data.set_index(['n_position','n_trigger'], inplace=True)
        collected_charge_data = data['Collected charge (V s)']
        collected_charges[channel] = collected_charge_data.to_numpy()
        collected_charges[channel] = collected_charges[channel] * 1e9
    connection.close()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    for channel in (1,2):       
        bin_min = -0.5; bin_max = 0.5       # HARDCODED BINS ??
        custom_bins = numpy.linspace(bin_min, bin_max, 100 ,endpoint=True)
        ax1.hist(collected_charges[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(collected_charges[channel])) * numpy.ones(len(collected_charges[channel])))
        ax1.set_xlabel(r"Collected Charge (n V s)")
        ax1.set_ylabel(f"Frequency")
        ax1.legend(loc = "best")

    for channel in (3,4):
        bin_min = -0.1; bin_max = 0.02       # HARDCODED BINS ??
        bin_min = -0.02
        custom_bins = numpy.linspace(bin_min, bin_max, 100 ,endpoint=True)
        ax2.hist(collected_charges[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(collected_charges[channel])) * numpy.ones(len(collected_charges[channel])))
        ax2.set_xlabel(r"Collected Charge (n V s)")
        ax2.set_ylabel(f"Frequency")
        ax2.legend(loc = "best")
    fig.suptitle(f'Collected Charge')
    plt.show()
    return None

def plot_noise(datafile, fig, ax1, ax2):
    query_dataset(datafile)
    noise = {}
    connection = sqlite3.connect(datafile)
    for channel in range(1, n_channels + 1):
        data = pandas.read_sql(f"SELECT n_position,n_trigger,`Noise (V)` FROM dataframe_table WHERE n_pulse==1 and n_channel=={channel}", connection)
        data.set_index(['n_position','n_trigger'], inplace=True)
        noise_data = data['Noise (V)']
        noise[channel] = noise_data.to_numpy()
    connection.close()
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    for channel in (1,2):
        bin_min = numpy.nanmedian(noise[channel]) - 0/2; bin_max = numpy.nanmedian(noise[channel]) + 0
        custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
        ax1.hist(noise[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(noise[channel])) * numpy.ones(len(noise[channel])))
        
        ax1.set_xlabel(r"Noise (V)")
        ax1.set_ylabel(f"Frequency")
        ax1.legend(loc = "best")

    for channel in (3,4):
        bin_min = 0; bin_max = 0
        custom_bins = numpy.linspace(bin_min, bin_max, 100 ,endpoint=True)
        ax2.hist(noise[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(noise[channel])) * numpy.ones(len(noise[channel])))
        ax2.set_xlabel(r"Noise (V)")
        ax2.set_ylabel(f"Frequency")
        ax2.legend(loc = "best")
    fig.suptitle(f'{datafile}')
    # plt.show()
    return None

def plot_time_difference_t50(datafile):
    query_dataset(datafile)
    time_difference = {}
    connection = sqlite3.connect(datafile)
    for channel in range(1, n_channels + 1):
        time_difference[channel] = []
        data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse,`t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
        data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
        time_data = data['t_50 (s)']
        for i in range(n_position):
            for j in range(n_triggers):
                time_diff = (time_data[i,j,2] - time_data[i,j,1]) * 1e9
                time_difference[channel].append(time_diff)
    connection.close()
    connection.close()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    for channel in (1,2):
        bin_min = 97; bin_max = 101
        bin_min = 98; bin_max = 100
        custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
        ax1.hist(time_difference[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(time_difference[channel])) * numpy.ones(len(time_difference[channel])))
        ax1.set_xlabel(r"Time (ns)")
        ax1.set_ylabel(f"Frequency")
        ax1.legend(loc = "best")

    for channel in (3,4):
        ax2.hist(time_difference[channel], bins=100, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(time_difference[channel])) * numpy.ones(len(time_difference[channel])))
        ax2.set_xlabel(r"Time (ns)")
        ax2.set_ylabel(f"Frequency")
        ax2.legend(loc = "best")
    fig.suptitle(f'Time Difference (t_50)')
    plt.show()
    return None

def plot_amplitude_against_t_peak(datafile, time_variable = "t_90 (s)"):
    query_dataset(datafile)
    result = {} #channel: ([time],[amplitude])
    connection = sqlite3.connect(datafile)
    for channel in range(1, n_channels + 1):
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
                if time_diff < TIME_DIFF_MIN * 1e-9 or time_diff > TIME_DIFF_MAX * 1e-9:
                    continue
                time = (time_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                amplitude = amplitude_data[i,j,1]
                result[channel][0].append(time)
                result[channel][1].append(amplitude)
    connection.close()

    fig, axes = plt.subplots(ncols=2, figsize=(12,5))
    ax1, ax2 = axes.flatten()
    subplots = (ax1, ax2)
    for channel in (1,2):
        subplots[channel - 1].plot(result[channel][0], result[channel][1], ".", label=f'Channel {channel}', markersize=2, color=CB_color_cycle[channel])
        subplots[channel - 1].set_xlim(4,9)
        subplots[channel - 1].set_ylim(-2,0)
        subplots[channel - 1].set_xlabel(f"Peak Time (ns)")
        subplots[channel - 1].set_ylabel(f"Amplitude (V)")
        subplots[channel - 1].legend(loc = "best")

    fig.suptitle(f'Amplitude against Peak Time, {datafile[5:11]}, {datafile[12:15]}')
    plt.show()
    return None

def plot_amplitude_of_one_pad(datafile, channel, pad_positions):
    query_dataset(datafile)
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
            if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD: 
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
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
    plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
    plt.xlabel(f"Amplitude (V)")
    plt.ylabel(f"Frequency")
    plt.legend(loc = "best")
    plt.title(f'{datafile[5:11]}, {datafile[12:16]}, Channel {channel}')
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
                if filename2 in ("50V","80V","100V", "130V"):
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
                plt.title(f'{filename}, {filename2}, Channel {chan1}')
                fig = plt.gcf()
                pdf.savefig(fig, dpi = 100)

                plt.clf()
                (muf, stdf) = plot_amplitude_of_one_pad(data_file, chan2, pad_positions[chan2])
                final_plot[filename][chan2][0].append(voltage)
                final_plot[filename][chan2][1].append(abs(muf)) # take absolute value of amplitude
                final_plot[filename][chan2][2].append(stdf)
                plt.title(f'{filename}, {filename2}, Channel {chan2}')
                fig = plt.gcf()
                pdf.savefig(fig, dpi = 100)
                
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
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}, Ch {channel}")
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
    return final_plot

def plot_collected_charge_of_one_pad(datafile, channel, pad_positions):
    query_dataset(datafile)
    collected_charges = []
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `Collected charge (V s)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    collected_charge_data = data['Collected charge (V s)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    for i in range(n_position):
        if i not in pad_positions or i<105 or i>1155:
            continue
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            collected_charge = collected_charge_data[i,j,1] * 1e9
            if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD:
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                continue
            if math.isnan(collected_charge) or collected_charge > 0:
                continue
            collected_charges.append(collected_charge)

    mu, std = statistics.mean(collected_charges), statistics.stdev(collected_charges)
    bin_min = mu - 4 * std; bin_max = mu + 4 * std
    custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
    (n, bins, patches) = plt.hist(collected_charges, bins=custom_bins, density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
    fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
    fit_y_axis = gaussian(fit_x_axis, muf, stdf)
    plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
    plt.xlabel(f"Collected Charge (nVs)")
    plt.ylabel(f"Frequency")
    plt.legend(loc = "best")
    plt.title(f'{datafile[5:11]}, {datafile[12:16]}, Channel {channel}')
    return (muf, stdf)

def plot_collected_charge_everything(directory_in_str = "Data/"):
    final_plot = {} # {sensor: channel: ([voltages], [mean amplitude], [std amplitude (error)])}
    with PdfPages(f"collected_charge.pdf") as pdf:
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
                if filename2 in ("50V", "80V","100V","130V"):
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
                if filename2.startswith('.'): #      # to ignore any hidden file
                    continue                
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file         = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2       = f"{directory3_in_str}measured_data.sqlite"
                positions_file    = f"{directory3_in_str}positions.pickle"
                print(f"{filename}, {filename2}")
                (voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)

                plt.clf()
                (muf, stdf) = plot_collected_charge_of_one_pad(data_file, chan1, pad_positions[chan1])
                final_plot[filename][chan1][0].append(voltage)
                final_plot[filename][chan1][1].append(abs(muf)) # take absolute value of amplitude
                final_plot[filename][chan1][2].append(stdf)
                plt.title(f'{filename}, {filename2}, Channel {chan1}')
                fig = plt.gcf()
                pdf.savefig(fig, dpi = 100)

                plt.clf()
                (muf, stdf) = plot_collected_charge_of_one_pad(data_file, chan2, pad_positions[chan2])
                final_plot[filename][chan2][0].append(voltage)
                final_plot[filename][chan2][1].append(abs(muf)) # take absolute value of amplitude
                final_plot[filename][chan2][2].append(stdf)
                plt.title(f'{filename}, {filename2}, Channel {chan2}')
                fig = plt.gcf()
                pdf.savefig(fig, dpi = 100)
                
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
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}, Ch {channel}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        plt.gca().invert_xaxis()
        plt.title("Collected Charge against Bias Voltage")
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Collected Charge (nVs)")
        plt.legend(loc = "best")
        fig = plt.gcf()
        pdf.savefig(fig, dpi = 100)
    return None

def plot_time_resolution_of_one_pad(datafile, channel, pad_positions):
    query_dataset(datafile)
    time_differences = []
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
            if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD:
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                continue
            if math.isnan(time_diff):
                continue
            time_differences.append(time_diff)

    mu, std = statistics.mean(time_differences), statistics.stdev(time_differences)
    bin_min = mu - 4 * std; bin_max = mu + 4 * std
    custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
    (n, bins, patches) = plt.hist(time_differences, bins=custom_bins, density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
    stdf_err = math.sqrt(covf[1][1])

    fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
    fit_y_axis = gaussian(fit_x_axis, muf, stdf)
    plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
    plt.xlabel(f"Time Difference Between Pulse 1 and 2 (ns)")
    plt.ylabel(f"Frequency")
    plt.legend(loc = "best")
    plt.title(f'{datafile[5:11]}, {datafile[12:16]}, Channel {channel}')
    return (stdf, stdf_err)

def plot_time_resolution_everything(directory_in_str = "Data/"):
    final_plot = {} # {sensor: channel: ([voltages], [mean amplitude], [std amplitude (error)])}
    with PdfPages(f"time_resolution.pdf") as pdf:
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
                if filename2 in ("50V", "80V","100V","130V"):
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
                if filename2.startswith('.'): #      # to ignore any hidden file
                    continue                
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file         = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2       = f"{directory3_in_str}measured_data.sqlite"
                positions_file    = f"{directory3_in_str}positions.pickle"
                print(f"{filename}, {filename2}")
                (voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)

                plt.clf()
                (stdf, stdf_err) = plot_time_resolution_of_one_pad(data_file, chan1, pad_positions[chan1])
                final_plot[filename][chan1][0].append(voltage)
                final_plot[filename][chan1][1].append(stdf) 
                final_plot[filename][chan1][2].append(stdf_err)
                plt.title(f'{filename}, {filename2}, Channel {chan1}')
                fig = plt.gcf()
                pdf.savefig(fig, dpi = 100)

                plt.clf()
                (stdf, stdf_err) = plot_time_resolution_of_one_pad(data_file, chan2, pad_positions[chan2])
                final_plot[filename][chan2][0].append(voltage)
                final_plot[filename][chan2][1].append(stdf) 
                final_plot[filename][chan2][2].append(stdf_err)
                plt.title(f'{filename}, {filename2}, Channel {chan2}')
                fig = plt.gcf()
                pdf.savefig(fig, dpi = 100)
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
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}, Ch {channel}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        plt.gca().invert_xaxis()
        plt.title("Time Resolution against Bias Voltage")
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Time Resolution (ns)")
        plt.legend(loc = "best")
        fig = plt.gcf()
        pdf.savefig(fig, dpi = 100)
    return None
    
def plot_2D_separate(datafile, positions): 
    # makes 2D plot of each channel, and projection of one edge
    query_dataset(datafile)
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
                if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD:
                    continue
                time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
                if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                    continue
                peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
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
    ax1.plot(y_axis, x_axis, marker='.', linestyle='-', color=CB_color_cycle[0], label=f"Channel {0}", markersize=5)
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
    ax3.plot(y_axis, x_axis, marker='.', linestyle='-', color=CB_color_cycle[0], label=f"Channel {0}", markersize=5)
    ax3.invert_yaxis()
    ax3.set_ylabel(r"y ($\mu$m)")
    ax3.set_xlabel(r"Mean Amplitude (V)")
    ax3.set_title("Channel 2")
    ax3.errorbar(y_axis, x_axis, xerr = err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
    fig.suptitle(f'{datafile[5:11]}, {datafile[12:16]}')
    plt.tight_layout()
    plt.show()
    return None               

def get_pad_positions(datafile, positions, channel): 
    # retrurns list of position indices that correspond to signal
    query_dataset(datafile)
    (x,y) = get_positions(positions)
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    amplitudes = []
    # filter out the meaningfull amplitudes
    for i in range(n_position):
        result = []
        #print(f"n_position: {i}")
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD:
                #print(f"amplitude: {amplitude}")
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                #print(f"time diff: {time_diff}")
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                #print(f"peak time: {peak_time}")
                continue
            result.append(amplitude)
        if result == []:
            amplitudes.append(0)
        else:
            amplitudes.append(statistics.mean(result))
    # build dataframe with x, y, amplitudes
    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': amplitudes})

    # select only intersting region
    # data_frame = data_frame[(data_frame["x"] >= -50) & (data_frame["x"] <= 50) & (data_frame["y"] >= -50) & (data_frame["y"] <= 50)]

    # build dataframe with x, y, amplitudes
    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': amplitudes})
    # return indices (n_position) where amplitude is non-zero (above numerical noise)
    mask = data_frame['z'].abs() > 1e-6
    pad_position = data_frame.index[mask].tolist()
    # pad_position = data_frame.index.tolist()
    if not pad_position:
        print(f"[Channel {channel}] No non-zero amplitudes found -> no pad positions")
    return pad_position

def plot_pad_positions(datafile, positions):
    query_dataset(datafile)
    (x, y) = get_positions(positions)  # lists of positions in µm
    (active_channel_1, active_channel_2) = determine_active_channels(datafile)
    pad_positions_1 = get_pad_positions(datafile, positions, active_channel_1)
    pad_positions_2 = get_pad_positions(datafile, positions, active_channel_2)

    plt.figure(figsize=(8, 6))
    # all positions as light points
    plt.scatter(x, y, s=4, color='lightgrey', label='All Positions', zorder=1)

    # channel 1 pads
    if pad_positions_1:
        xs1 = [x[i] for i in pad_positions_1]
        ys1 = [y[i] for i in pad_positions_1]
        plt.scatter(xs1, ys1, s=30, color=CB_color_cycle[0], edgecolor='k', label=f'Pad Positions Ch {active_channel_1}', zorder=3)

    # channel 2 pads
    if pad_positions_2:
        xs2 = [x[i] for i in pad_positions_2]
        ys2 = [y[i] for i in pad_positions_2]
        plt.scatter(xs2, ys2, s=30, color=CB_color_cycle[1], edgecolor='k', label=f'Pad Positions Ch {active_channel_2}', zorder=4)

    plt.xlabel(r"x ($\mu$m)")
    plt.ylabel(r"y ($\mu$m)")
    plt.title(f'Pad Positions for {datafile[5:11]}, {datafile[12:16]}')
    plt.legend(loc='best')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    return None

def get_sensor_strip_positions(datafile, positions, channel):
    query_dataset(datafile)
    (x,y) = get_positions(positions)
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    amplitudes = []
    for i in range(n_position):
        result = []
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude) or amplitude > 0: # HARDCODED TRESHOLD??
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                continue
            result.append(amplitude)
        if result == []:
            amplitudes.append(0)
        else:
            amplitudes.append(statistics.mean(result))

    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': amplitudes})

    # select only intersting region
    data_frame = data_frame[(data_frame["x"] >= -50) & (data_frame["x"] <= 50) & (data_frame["y"] >= -50) & (data_frame["y"] <= 50)]

    points_from_edge = 2 # this is hardcoded number
    smallest_list = (pandas.Series(data_frame['y'].unique()).nsmallest(points_from_edge)).to_list()
    largest_list = (pandas.Series(data_frame['y'].unique()).nlargest(points_from_edge)).to_list()
    bins_in_y = len((pandas.Series(data_frame['y'].unique())).to_list())
    bins_in_y_we_want = bins_in_y - points_from_edge
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
        valid_y = (pandas.Series(data_frame['y'].unique()).nsmallest(bins_in_y_we_want)).to_list()
    else:
        x_axis = edge2_projection.index.to_numpy()
        y_axis = edge2_projection.to_numpy()
        err = edge2_projection_sig.to_numpy()
        valid_y = (pandas.Series(data_frame['y'].unique()).nlargest(bins_in_y_we_want)).to_list()
    peak_index = numpy.where(y_axis == min(y_axis))
    peak_error = err[peak_index]
    peak_max = min(y_axis) + peak_error; peak_min = min(y_axis) - peak_error
    valid_x = []
    for value in y_axis:
        value_index = numpy.where(y_axis == value)
        value_error = err[value_index]
        value_max = value + value_error; value_min = value - value_error
        if peak_min <= value_max <= peak_max or peak_min <= value_min <= peak_max or value_min <= peak_max <= value_max or value_min <= peak_min <= value_max:
            x_value = x_axis[value_index]
            valid_x.append(x_value[0])
    pad_position = [] # list of n_positions that is the pad
    position_frame = pandas.DataFrame({'x': x, 'y': y})

    # valid_y = (pandas.Series(data_frame['y'].unique())).to_list()

    for x_value in valid_x:
        for y_value in valid_y:
            row_index = position_frame.query(f"x == {x_value} and y == {y_value}").index[0]
            pad_position.append(row_index)
    # If you want to see which x and y is selected
    print(f"Valid x: {valid_x}")
    print(f"Valid y: {valid_y}")
    print(f"pad position strip: {pad_position}")
    return pad_position

def plot_sensor_strip_positions(datafile, positions):
    # Load and prepare data
    query_dataset(datafile)
    x, y = get_positions(positions)
    active_ch1, active_ch2 = determine_active_channels(datafile)

    sensor_pos_ch1 = get_sensor_strip_positions(datafile, positions, active_ch1)
    sensor_pos_ch2 = get_sensor_strip_positions(datafile, positions, active_ch2)

    # Prepare figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    titles = [
        "Channel 1 + Channel 2",
        f"Channel {active_ch1} Only",
        f"Channel {active_ch2} Only",
    ]

    # Helper to draw one subplot
    def draw_subplot(ax, show_ch1=True, show_ch2=True):
        ax.scatter(x, y, s=4, color='lightgrey', label="All Positions", zorder=1)

        # Channel 1
        if show_ch1 and sensor_pos_ch1:
            xs1 = [x[i] for i in sensor_pos_ch1]
            ys1 = [y[i] for i in sensor_pos_ch1]
            ax.scatter(
                xs1, ys1, s=30, color=CB_color_cycle[0],
                edgecolor='k', label=f"Ch {active_ch1}", zorder=3
            )

        # Channel 2
        if show_ch2 and sensor_pos_ch2:
            xs2 = [x[i] for i in sensor_pos_ch2]
            ys2 = [y[i] for i in sensor_pos_ch2]
            ax.scatter(
                xs2, ys2, s=30, color=CB_color_cycle[1],
                edgecolor='k', label=f"Ch {active_ch2}", zorder=4
            )

        ax.set_xlabel(r"x ($\mu$m)")
        ax.set_ylabel(r"y ($\mu$m)")
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')

    # --- Create the three subplots ---
    draw_subplot(axes[0], show_ch1=True, show_ch2=True)   # both channels
    draw_subplot(axes[1], show_ch1=True, show_ch2=False)  # only channel 1
    draw_subplot(axes[2], show_ch1=False, show_ch2=True)  # only channel 2

    # Titles
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    plt.suptitle(f"Sensor Strip Positions — {datafile[5:11]}, {datafile[12:16]}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return None


def project_onto_y_two_channels(datafile, positions, channel1, channel2, sensor_strip_positions1, sensor_strip_positions2, pdf):
    query_dataset(datafile)
    (x,y) = get_positions(positions)
    amplitudes = {} # {y position: [list of amplitudes]}
    counter = {}
    result = {"x axis": [], "y axis": [], "y error": []}
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse,n_channel, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table", connection)
    data.set_index(['n_position','n_trigger','n_pulse','n_channel'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    sensor_strip_positions = list(set(sensor_strip_positions1 + sensor_strip_positions2))
    # for normalisation 
    pad_positions = get_pad_positions(datafile, positions, channel1)
    (ch1_norm, ch1_norm_err) = plot_amplitude_of_one_pad(datafile, channel1, pad_positions)
    pad_positions = get_pad_positions(datafile, positions, channel2)
    (ch2_norm, ch2_norm_err) = plot_amplitude_of_one_pad(datafile, channel2, pad_positions)
    for i in sensor_strip_positions:
        for j in range(n_triggers):
            amplitude1 = amplitude_data[i,j,1,channel1]
            time_diff1 = (t_50_data[i,j,2,channel1] - t_50_data[i,j,1,channel1]) * 1e9
            peak_time1 = (t_90_data[i,j,1,channel1] + 0.5 * time_over_90_data[i,j,1,channel1]) * 1e9
            amplitude2 = amplitude_data[i,j,1,channel2]
            time_diff2 = (t_50_data[i,j,2,channel2] - t_50_data[i,j,1,channel2]) * 1e9
            peak_time2 = (t_90_data[i,j,1,channel2] + 0.5 * time_over_90_data[i,j,1,channel2]) * 1e9
            # if any of them fail cut, set to 0
            if math.isnan(amplitude1) or amplitude1 > AMPLITUDE_THRESHOLD:
                amplitude1 = 0
            if time_diff1 < TIME_DIFF_MIN or time_diff1 > TIME_DIFF_MAX:
                amplitude1 = 0
            if peak_time1 < PEAK_TIME_MIN or peak_time1 > PEAK_TIME_MAX:
                amplitude1 = 0
            if math.isnan(amplitude2) or amplitude2 > AMPLITUDE_THRESHOLD: # FOR SOME REASON WAS 0
                amplitude2 = 0
            if time_diff2 < TIME_DIFF_MIN or time_diff2 > TIME_DIFF_MAX:
                amplitude2 = 0
            if peak_time2 < PEAK_TIME_MIN or peak_time2 > PEAK_TIME_MAX:
                amplitude2 = 0
            # if both fails cut, go next
            if amplitude1 + amplitude2 == 0:
                continue

            if y[i] not in amplitudes:
                amplitudes[y[i]] = []
            normalised_amplitude = amplitude1/ch1_norm + amplitude2/ch2_norm
            amplitudes[y[i]].append(normalised_amplitude)

    for y_position in sorted(amplitudes):
        plt.clf()
        hist = amplitudes[y_position]
        mu, std = statistics.mean(hist), statistics.stdev(hist)
        bin_min = mu - 4 * std; bin_max = mu + 4 * std
        bin_width = 0.0005 # hardcoded bin width (initial value = 0.05)
        n_bins = round( (bin_max - bin_min) / bin_width )
        custom_bins = numpy.linspace(bin_min, bin_max, n_bins ,endpoint=True)
        (n, bins, patches) = plt.hist(hist, bins="auto", density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
        fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
        fit_y_axis = gaussian(fit_x_axis, muf, stdf)
        result["x axis"].append(y_position)
        result["y axis"].append(abs(muf))
        result["y error"].append(stdf)
        plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
        plt.xlabel(f"Amplitude")
        plt.ylabel(f"Frequency")
        plt.legend(loc = "best")
        plt.title(f'{datafile[5:11]}, {datafile[12:16]}, Channel {channel1} + {channel2}, y: {y_position} ${{\mu}}$m, N: {len(hist)}')
        fig = plt.gcf()
        pdf.savefig(fig, dpi = 100)
    return result

def project_onto_y_one_channel(datafile, positions, channel, sensor_strip_positions):
    query_dataset(datafile)
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
            if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
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

        plt.title(f"Histogram amplitudes — y = {y_position}")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.legend()

        #plt.show()
    return result

def get_interpad_distance(datafile, positions, channel1, sensor_strip_positions1,
                          channel2, sensor_strip_positions2, pdf):

    channel1_result = project_onto_y_one_channel(datafile, positions, channel1, sensor_strip_positions1)
    channel2_result = project_onto_y_one_channel(datafile, positions, channel2, sensor_strip_positions2)
    both_channels   = project_onto_y_two_channels(datafile, positions, channel1, channel2,
                                                  sensor_strip_positions1, sensor_strip_positions2, pdf)
    plt.clf()

    # ============================
    # Channel 1
    # ============================
    x_axis = numpy.array(channel1_result["x axis"])
    y_axis = numpy.array(channel1_result["y axis"])
    y_err  = numpy.array(channel1_result["y error"])

    print(y_axis)

    # --- Normalize once: scale data to [0, 1] ---
    y_min = y_axis.min()
    y_max = y_axis.max()

    y_norm = (y_axis - y_min) / (y_max - y_min)
    y_err_norm = y_err / (y_max - y_min)   # error scales identically

    # --- Sigmoid fit on normalized data ---
    guess = [numpy.median(x_axis), 1.0, 0.0, 1.0]
    popt, pcov = curve_fit(
        sigmoid, x_axis, y_norm, p0=guess,
        sigma=y_err_norm, absolute_sigma=True,
        maxfev=10000, method="dogbox"
    )

    ch1_x0 = popt[0]
    ch1_x0_uncertainty = math.sqrt(pcov[0][0])

    # --- Generate fit curve ---
    fine_x = numpy.linspace(x_axis.min(), x_axis.max(), 500)
    fit_norm = sigmoid(fine_x, *popt)

    # --- Plot Ch 1 ---
    plt.plot(fine_x, fit_norm, "-", label=f"Ch {channel1} Fit", color=CB_color_cycle[0])
    plt.plot(x_axis, y_norm, ".", markersize=3, label=f"Ch {channel1} Data", color=CB_color_cycle[0])
    plt.errorbar(x_axis, y_norm, yerr=y_err_norm, ls="none", ecolor="k",
                 elinewidth=1, capsize=2)
    plt.axvline(x=ch1_x0, ymin=0, ymax=1, color='k', label=f'x{channel1} = {round(ch1_x0,2)}', ls = (0, (5, 10)), linewidth=0.6)

    # ============================
    # Channel 2
    # ============================
    x_axis = numpy.array(channel2_result["x axis"])
    y_axis = numpy.array(channel2_result["y axis"])
    y_err  = numpy.array(channel2_result["y error"])

    # --- Normalize once: scale data to [0, 1] ---
    y_min = y_axis.min()
    y_max = y_axis.max()

    y_norm = (y_axis - y_min) / (y_max - y_min)
    y_err_norm = y_err / (y_max - y_min)   # error scales identically

    # --- Sigmoid fit ---
    guess = [numpy.median(x_axis), 1.0, 0.0, 1.0]
    popt, pcov = curve_fit(sigmoid, x_axis, y_norm, p0=guess,
                           maxfev=10000, method="dogbox")

    ch2_x0 = popt[0]
    ch2_x0_uncertainty = math.sqrt(pcov[0][0])

    # --- Fit curve ---
    fine_x = numpy.linspace(x_axis.min(), x_axis.max(), 500)
    fit_norm = sigmoid(fine_x, *popt)

    # --- Plot Ch 2 ---
    plt.plot(fine_x, fit_norm, "-", label=f"Ch {channel2} Fit", color=CB_color_cycle[1])
    plt.plot(x_axis, y_norm, ".", markersize=3, label=f"Ch {channel2} Data", color=CB_color_cycle[1])
    plt.errorbar(x_axis, y_norm, yerr=y_err_norm, ls="none", ecolor="k",
                 elinewidth=1, capsize=2)
    plt.axvline(x=ch2_x0, ymin=0, ymax=1, color='k', label=f'x{channel2} = {round(ch2_x0,2)}', ls = (0, (5, 10)), linewidth=0.6)


    # ============================
    # Sum of channels (no fit)
    # ============================
    plt.plot(both_channels["x axis"], both_channels["y axis"], ".-",
             markersize=3, linewidth=1, color=CB_color_cycle[3],
             label="Ch 1+2 Data")
    plt.errorbar(both_channels["x axis"], both_channels["y axis"],
                 yerr=both_channels["y error"], ls="none",
                 ecolor="k", elinewidth=1, capsize=2)

    # ============================
    # Plot settings
    # ============================
    plt.title(f"{datafile[5:11]}, {datafile[12:16]}")
    plt.gca().invert_xaxis()
    plt.xlabel(r"x ($\mu$m)")
    plt.ylabel("Amplitude (V)")
    plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    #plt.show()

    # ============================
    # Return interpad distance
    # ============================
    return (abs(ch1_x0 - ch2_x0), ch1_x0_uncertainty + ch2_x0_uncertainty)



def plot_interpad_distance_against_bias_voltage_v2(directory_in_str = "Data/"):
    result = {}
    # Open a PDF file to save all plots
    with PdfPages('interpad_distance.pdf') as pdf:
        directory = os.fsencode(directory_in_str)
        # file processing
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.startswith('.'): #    if filename.startswith('.'):   # to ignore any hidden file
                continue
            result[filename] = {}
            interpad = []; interpad_err = []
            bias_volt = []; bias_volt_err = []
            directory2_in_str = f"{directory_in_str}{filename}/"
            directory2 = os.fsencode(directory2_in_str)
            
            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                print(folder)
                filename2 = os.fsdecode(folder)
                if filename2.startswith('.'):  # to ignore any hidden file
                    continue   
        
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                if filename2 in ("50V","80V","100V", "130V"):
                    data_file       = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                    positions_file  = f"{directory3_in_str}positions.pickle"
                    (chan1, chan2) = determine_active_channels(data_file)

                    print(f"Current Sensor: {filename} | Active Channels: {chan1} and {chan2}")
                    # Get sensor strip positions for both channels
                    sensor_strip_positions = {}
                    sensor_strip_positions[chan1] = get_sensor_strip_positions(data_file, positions_file, chan1)
                    sensor_strip_positions[chan2] = get_sensor_strip_positions(data_file, positions_file, chan2)
                    break
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
                data_file      = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2    = f"{directory3_in_str}measured_data.sqlite"
                positions_file = f"{directory3_in_str}positions.pickle"
                (voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)
    
                print(f"Current Sensor: {filename} | Voltage: {filename2}")
                plt.clf()
                (interpad_distance, interpad_distance_uncertainty) = get_interpad_distance(data_file, positions_file, chan1, sensor_strip_positions[chan1], chan2, sensor_strip_positions[chan2], pdf)
                fig = plt.gcf() # get current figure  
                pdf.savefig(fig, dpi = 100, bbox_inches='tight')
                bias_volt.append(voltage)
                bias_volt_err.append(voltage_uncertainty)
                interpad.append(interpad_distance)
                interpad_err.append(interpad_distance_uncertainty)
            result[filename]["x"] = bias_volt
            result[filename]["x_err"] = bias_volt_err
            result[filename]["y"] = interpad
            result[filename]["y_err"] = interpad_err
        plt.close()
        color_index = 0
        for key in result:
            x_ax = result[key]["x"]
            y_ax = result[key]["y"]
            x_ax_err = result[key]["x_err"]
            y_ax_err = result[key]["y_err"]
            plt.plot(x_ax, y_ax, 'o', label=f"{key}", markersize=3, color=CB_color_cycle[color_index], linestyle="--", linewidth=1)
            plt.errorbar(x_ax, y_ax, xerr = x_ax_err, yerr = y_ax_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
            plt.title(f"Interpad Distance against Bias Voltage")
            plt.xlabel(r"Bias Voltage (V)")
            plt.ylabel(r"Interpad Distance ($\mu$m)")
            plt.legend(loc='best', ncol = 1)
            color_index += 1
        plt.gca().invert_xaxis()
        fig = plt.gcf() # get current figure
        pdf.savefig(fig, dpi = 100)
    plt.show()
    return None

def plot_time_difference_histogram(datafile, positions, y_position, pdf):
    (chan1, chan2) = determine_active_channels(datafile)
    query_dataset(datafile)
    (x,y) = get_positions(positions)
    time_differences = [] # list of time differences
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_channel,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table", connection)
    data.set_index(['n_position','n_trigger','n_pulse','n_channel'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']

    for i in range(n_position):
        if y[i] != y_position:
            continue
        for j in range(n_triggers):
            for channel in (chan1, chan2):
                amplitude = amplitude_data[i,j,1,channel]
                if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD:
                    continue
                time_diff = (t_50_data[i,j,2, channel] - t_50_data[i,j,1, channel]) * 1e9
                if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                    continue
                peak_time = (t_90_data[i,j,1, channel] + 0.5 * time_over_90_data[i,j,1, channel]) * 1e9
                if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                    continue
                if y[i] >= INTERPAD_REGION_MIN and y[i] <= INTERPAD_REGION_MAX: # only pad region HARDCODED
                    continue
                time_differences.append(time_diff)
    mu, std = statistics.mean(time_differences), statistics.stdev(time_differences)
    bin_min = mu - 4 * std; bin_max = mu + 4 * std
    custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
    (n, bins, patches) = plt.hist(time_differences, bins="auto", density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
    fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
    fit_y_axis = gaussian(fit_x_axis, muf, stdf)

    plt.clf()
    plt.hist(time_differences, bins=50, color=CB_color_cycle[0], alpha=0.7)
    plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
    plt.title(f"Time Difference Histogram at y = {y_position} µm — {datafile[5:11]}, {datafile[12:16]}")
    plt.xlabel("Time Difference (ns)")
    plt.ylabel("Counts")
    plt.legend(loc = "best")
    plt.tight_layout()
    #plt.show()  
    fig = plt.gcf() # get current figure
    pdf.savefig(fig, dpi = 100, bbox_inches='tight')
    return None

def plot_time_resolution_interpad_region(datafile, positions, pdf):
    (chan1, chan2) = determine_active_channels(datafile)
    query_dataset(datafile)
    (x,y) = get_positions(positions)
    time_differences = {} # {y position: [list of time differences]}
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_channel,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table", connection)
    data.set_index(['n_position','n_trigger','n_pulse','n_channel'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']

    for i in range(n_position):
        for j in range(n_triggers):
            for channel in (chan1, chan2):
                amplitude = amplitude_data[i,j,1,channel]
                if math.isnan(amplitude) or amplitude > AMPLITUDE_THRESHOLD:
                    continue
                time_diff = (t_50_data[i,j,2, channel] - t_50_data[i,j,1, channel]) * 1e9
                if time_diff < TIME_DIFF_MIN or time_diff > TIME_DIFF_MAX:
                    continue
                peak_time = (t_90_data[i,j,1, channel] + 0.5 * time_over_90_data[i,j,1, channel]) * 1e9
                if peak_time < PEAK_TIME_MIN or peak_time > PEAK_TIME_MAX:
                    continue
                if y[i] < INTERPAD_REGION_MIN or y[i] > INTERPAD_REGION_MAX: # only interpad region
                    continue

                if y[i] not in time_differences:
                    time_differences[y[i]] = []
                time_differences[y[i]].append(time_diff)

    result = {"x axis": [], "y axis": [], "y error": []}

    for y_position in sorted(time_differences):
        hist = time_differences[y_position]
        if len(hist) < 2: # toss
            continue
        mu, std = statistics.mean(hist), statistics.stdev(hist)
        # bin_min = mu - 4 * std; bin_max = mu + 4 * std
        # custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
        # plt.clf()
        # (n, bins, patches) = plt.hist(hist, bins=custom_bins, density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")
        # bin_centers = (bins[:-1] + bins[1:]) / 2
        # (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
        # plt.plot(numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True), gaussian(numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True), muf, stdf), color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
        # plt.title(f"Time Difference Histogram at y = {y_position} µm — {datafile[5:11]}, {datafile[12:16]}")
        # plt.xlabel("Time Difference (ns)")
        # plt.ylabel("Counts")
        # plt.legend(loc = "best")
        # plt.tight_layout()
        # fig = plt.gcf() # get current figure
        # pdf.savefig(fig, dpi = 100, bbox_inches='tight')

        result["x axis"].append(y_position)
        result["y axis"].append(std)
        result["y error"].append(std / math.sqrt(2 * (len(hist) - 1)))  # standard error of the standard deviation

    plt.clf()
    plt.plot(result["x axis"], result["y axis"], ".-",
            markersize=3, linewidth=1, color=CB_color_cycle[0],
            label=f"Time Resolution")
    plt.errorbar(result["x axis"], result["y axis"],
                yerr=result["y error"], ls="none",
                ecolor="k", elinewidth=1, capsize=2)
    plt.title(f"Time Resolution vs y Position — {datafile[5:11]}, {datafile[12:16]}")
    plt.gca().invert_xaxis()
    plt.xlabel(r"y Position ($\mu$m)")
    plt.ylabel("Time Resolution (ns)")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    #plt.show()  
    fig = plt.gcf() # get current figure
    pdf.savefig(fig, dpi = 100, bbox_inches='tight')
    return result

def plot_time_resolution_interpad_region_everything(directory_in_str = "Data/"):
    # Open a PDF file to save all plots
    final_plot = {}
    with PdfPages('timing_interpad_region.pdf') as pdf:
        directory = os.fsencode(directory_in_str)
        # file processing
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.startswith('.'): #    if filename.startswith('.'):   # to ignore any hidden file
                continue

            directory2_in_str = f"{directory_in_str}{filename}/"
            directory2 = os.fsencode(directory2_in_str)
            
            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                print(folder)
                filename2 = os.fsdecode(folder)
                if filename2.startswith('.'):  # to ignore any hidden file
                    continue   
        
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file       = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                positions_file  = f"{directory3_in_str}positions.pickle"
                print(f"Current Sensor: {filename} | Voltage: {filename2}")
                final_plot[filename2] = plot_time_resolution_interpad_region(data_file, positions_file, pdf)
            plt.clf()
            plt.title(f"Time Resolution in Interpad Region vs bias Voltage")
            plt.xlabel(r"y Position ($\mu$m)")
            plt.ylabel("Time Resolution (ns)")
            plt.gca().invert_xaxis()
            plt.tight_layout()
            colors_index = 0
            for key in final_plot:
                x_ax = final_plot[key]["x axis"]
                y_ax = final_plot[key]["y axis"]
                y_ax_err = final_plot[key]["y error"]
                plt.plot(x_ax, y_ax, 'o-', label=f"{key}", markersize=3, color=CB_color_cycle[colors_index], linewidth=1)
                plt.errorbar(x_ax, y_ax, yerr = y_ax_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                colors_index += 1
            #plt.ylim(bottom=0.01, top=0.1) # ajust scale
            #plt.legend(loc='upper right', ncol = 1)
            plt.legend(loc='best', ncol = 1)
            fig = plt.gcf() # get current figure
            pdf.savefig(fig, dpi = 100, bbox_inches='tight')
    return None


###################################################################################################################
###################################################################################################################
###################################################################################################################
################################################### For testing ###################################################
print(f"Script started")
start_time = time.time()


#plot_amplitude_everything()  #"1_plot_amplitude_everything" in Desktop/analysis_tct/Plts_of_Analysis2 ##working - problem that a lot of noise is taken into the amplitude
#plot_collected_charge(test_datafile)  #"2_plot_collected_charge" in Desktop/analysis_tct/Plts_of_Analysis2 # working
#plot_amplitude(test_datafile, test_positions)   #"3_plot_amplitude" in Desktop/analysis_tct/Plts_of_Analysis2 # working
#plot_2D_separate(test_datafile, test_positions)   #"4_plot_2D_separate" in Desktop/analysis_tct/Plts_of_Analysis2. # working
plot_interpad_distance_against_bias_voltage_v2()   #"5_plot_interpad_distance_against_bias_voltage_v2" in Desktop/analysis_tct/Plts_of_Analysis2. # not working
#plot_collected_charge_everything()   #6_plot_collected_charge_everything  in Desktop/analysis_tct/Plts_of_Analysis2 # not working
#project_onto_y_two_channels(test_datafile, test_positions, 1, 2, get_sensor_strip_positions(test_datafile, test_positions, 1), get_sensor_strip_positions(test_datafile, test_positions, 1), pdf) # not working (pdf not defined)
#"7_project_onto_y_two_channels" in Desktop/analysis_tct/Plts_of_Analysis2
#project_onto_y_one_channel(test_datafile, test_positions, 1, get_sensor_strip_positions(test_datafile, test_positions, 1)) # not working
#project_onto_y_two_channels(test_datafile, test_positions, 1, 2, get_sensor_strip_positions(test_datafile, test_positions, 1), get_sensor_strip_positions(test_datafile, test_positions, 1), pdf) # not working
#project_onto_y_one_channel(test_datafile, test_positions, 1, get_sensor_strip_positions(test_datafile, test_positions, 1)) # not working
#plot_time_difference_t50(test_datafile) # Working
#plot_time_resolution_everything()
#plot_time_resolution_interpad_region_everything() 
#plot_pad_positions(test_datafile, test_positions)
#plot_sensor_strip_positions(test_datafile, test_positions)
# query_dataset(test_datafile)
# print(get_positions(test_positions))
#print(get_channel_amplitude(test_datafile, 1))


time_taken = round(time.time() - start_time)

minutes = time_taken // 60
seconds = time_taken - 60*minutes 
print(f"--- Runtime: {minutes} minutes {seconds} seconds ---")
################################################### For testing ###################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
