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

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'] #color blind friendly colors
colors = ["Red","Green","Blue","Yellow","Cyan","Magenta","Orange","Purple","Pink","Brown","Black","White","Gray","DarkRed","DarkGreen","DarkBlue","LightGray","LightGreen","LightBlue","LightCoral"]

test_directory = os.path.expanduser("/Users/leenadiehl/cernbox/Documents/Zurich/CASSIA_TCT/Data/Areascans/EPI_M2_Central/52V") ### directory used for e.g. 2d scan images
test_datafile  = os.path.abspath(f"{test_directory}/parsed_from_waveforms.sqlite")
test_datafile2 = os.path.abspath(f"{test_directory}/measured_data.sqlite")
test_positions = os.path.abspath(f"{test_directory}/positions.pickle")

def sigmoid(x, x0, b, L, k):
    y = L / (1 + numpy.exp(-k*(x-x0))) + b
    return (y)

def gaussian(x, mu, sig):
    return 1./(numpy.sqrt(2.*numpy.pi)*sig)*numpy.exp(-numpy.power((x - mu)/sig, 2.)/2)

def query_dataset(datafile):
    # print(f"Querying dataset...")
    global n_position; global n_triggers; global n_channels
    connection = sqlite3.connect(datafile)
    query = "SELECT n_position FROM dataframe_table WHERE n_trigger == 0 and n_pulse == 1 and n_channel == 1"
    filtered_data = pandas.read_sql_query(query, connection)
    n_position = len(filtered_data)
    query = "SELECT n_trigger FROM dataframe_table WHERE n_position == 0 and n_pulse == 1 and n_channel == 1"
    filtered_data = pandas.read_sql_query(query, connection)
    n_triggers = len(filtered_data)
    query = "SELECT n_channel FROM dataframe_table WHERE n_position == 0 and n_pulse == 1 and n_trigger == 0"
    filtered_data = pandas.read_sql_query(query, connection)
    n_channels = len(filtered_data)
    connection.close()
    #print(f"{n_position} positions, {n_triggers} triggers and {n_channels} channels found")
    return None

def get_positions(positions):
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
    for i in range(n_position) :
        result = []
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude) or amplitude > -0.0:
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < 98 or time_diff > 99.5:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < 4.5 or peak_time > 6.5:
                continue
            result.append(amplitude)
        if result == []:
            amplitudes.append(0)
        else:
            amplitudes.append(statistics.mean(result))
    return amplitudes
# Hardcoded to return channels 1 and 2 for now
def determine_active_channels(datafile):
##    query_dataset(datafile)
##    result = {}; list_to_sort = []
##    for channel in range(1, n_channels + 1):
##        amplitudes = get_channel_amplitude(datafile, channel, pulse_no = 1, method = "median")
##        result[round(sum(amplitudes),3)] = channel
##        list_to_sort.append(round(sum(amplitudes),3))
##        list_to_sort = sorted(list_to_sort)
##    return tuple(sorted((result[list_to_sort[0]], result[list_to_sort[1]])))
    return (1,2)

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
            if i not in pad_positions :
                continue
            for j in range(n_triggers):
                amplitude = amplitude_data[i,j,1]
                if math.isnan(amplitude) or amplitude > 0:
                    continue
                time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
                if time_diff < 98 or time_diff > 99.5:
                    continue
                peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                if peak_time < 4.5 or peak_time > 6.5:
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
        bin_min = -4; bin_max = 2
        custom_bins = numpy.linspace(bin_min, bin_max, 100 ,endpoint=True)
        ax1.hist(collected_charges[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(collected_charges[channel])) * numpy.ones(len(collected_charges[channel])))
        ax1.set_xlabel(r"Collected Charge (n V s)")
        ax1.set_ylabel(f"Frequency")
        ax1.legend(loc = "best")

    for channel in (3,4):
        bin_min = -0.1; bin_max = 0.02
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
        bin_min = numpy.nanmedian(noise[channel]) - 0.01/2; bin_max = numpy.nanmedian(noise[channel]) + 0.01
        custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
        ax1.hist(noise[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=CB_color_cycle[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(noise[channel])) * numpy.ones(len(noise[channel])))
        
        ax1.set_xlabel(r"Noise (V)")
        ax1.set_ylabel(f"Frequency")
        ax1.legend(loc = "best")

    for channel in (3,4):
        bin_min = 0; bin_max = 0.03
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
                if amplitude_data[i,j,1] > 0:
                    continue
                time_diff = t_50_data[i,j,2] - t_50_data[i,j,1]
                if time_diff < 98e-9 or time_diff > 99.5e-9:
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
        #if i not in pad_positions:
         #   continue
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude):# or amplitude > -0.0: #or amplitude > 0
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < 98 or time_diff > 99.5:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
           # if peak_time < 4.5 or peak_time > 6.5:
            #    continue
            amplitudes.append(amplitude)

    mu, std = statistics.mean(amplitudes), statistics.stdev(amplitudes)
    bin_min = mu - 4 * std; bin_max = mu + 4 * std
    custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
    (n, bins, patches) = plt.hist(amplitudes, bins="auto", density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
    fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
    fit_y_axis = gaussian(fit_x_axis, muf, stdf)
    # plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
    plt.xlabel(f"Amplitude (V)")
    plt.ylabel(f"Frequency")
    plt.legend(loc = "best")
    plt.title(f'{datafile[5:11]}, {datafile[12:16]}, Channel {channel}')
    # plt.show()
    return (muf, stdf)

def plot_amplitude_everything(directory_in_str = "Data/Amp40db/"):
    final_plot = {} # {sensor: channel: ([voltages], [mean amplitude], [std amplitude (error)])}
    with PdfPages(f"Amplitudes_Amp40db.pdf") as pdf:
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
                #print(folder)
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
                    #pad_positions[chan2] = get_pad_positions(data_file, positions_file, chan2)
                    break

            final_plot[filename][chan1] = ([],[],[])#; final_plot[filename][chan2] = ([],[],[])

            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                #print(folder)                
                filename2 = os.fsdecode(folder)
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file         = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2       = f"{directory3_in_str}measured_data.sqlite"
                positions_file    = f"{directory3_in_str}positions.pickle"
                print(f"{filename}, {filename2}")
                voltage = int(folder.rstrip('V'))
                #(voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)

                plt.clf()
                (muf, stdf) = plot_amplitude_of_one_pad(data_file, chan1, pad_positions[chan1])
                final_plot[filename][chan1][0].append(voltage)
                final_plot[filename][chan1][1].append(abs(muf)) # take absolute value of amplitude
                final_plot[filename][chan1][2].append(stdf)
                plt.title(f'{filename}, {filename2}, Channel {chan1}')
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
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        #plt.gca().invert_xaxis()
        plt.title("Amplitude against Bias Voltage")
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
        if i not in pad_positions:
            continue
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            collected_charge = collected_charge_data[i,j,1] * 1e9
            if math.isnan(amplitude):# or amplitude > -0.0:
                continue
            #time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            #if time_diff < 98 or time_diff > 99.5:
             #   continue
           # peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            #if peak_time < 4.5 or peak_time > 6.5:
             #   continue
            if math.isnan(collected_charge):# or collected_charge > 0:
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

def plot_collected_charge_everything(directory_in_str = "Data/Amp13db/"):
    final_plot = {} # {sensor: channel: ([voltages], [mean amplitude], [std amplitude (error)])}
    with PdfPages(f"Charges_Amp13db.pdf") as pdf:
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
                #print(folder)
                voltage = int(folder.rstrip('V'))
                filename2 = os.fsdecode(folder)
                if filename2.startswith('.'): #      # to ignore any hidden file
                    continue                
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                if filename2 in ("50V", "80V","100V","130V"):
                    data_file       = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                    positions_file  = f"{directory3_in_str}positions.pickle"
                    (chan1,chan2) = determine_active_channels(data_file)
                    pad_positions = {}
                    pad_positions[chan1] = get_pad_positions(data_file, positions_file, chan1)
                   # pad_positions[chan2] = get_pad_positions(data_file, positions_file, chan2)
                    break

            final_plot[filename][chan1] = ([],[],[])#; final_plot[filename][chan2] = ([],[],[])

            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                #print(folder)
                filename2 = os.fsdecode(folder)
                if filename2.startswith('.'): #      # to ignore any hidden file
                    continue                
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file         = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2       = f"{directory3_in_str}measured_data.sqlite"
                positions_file    = f"{directory3_in_str}positions.pickle"
                print(f"{filename}, {filename2}")
                #(voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)
                voltage = int(folder.rstrip('V'))

                plt.clf()
                (muf, stdf) = plot_collected_charge_of_one_pad(data_file, chan1, pad_positions[chan1])
                final_plot[filename][chan1][0].append(voltage)
                final_plot[filename][chan1][1].append(abs(muf)) # take absolute value of amplitude
                final_plot[filename][chan1][2].append(stdf)
                #print(final_plot) ## used to determine the no-gain collected charge
                plt.title(f'{filename}, {filename2}, Channel {chan1}')
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
                #print(y_axis)
                if linestyle_counter % 2 == 0:
                    linestyle = "-"
                else:
                    linestyle = "--"
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        #plt.gca().invert_xaxis()
        plt.title("Collected Charge against Bias Voltage")
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Collected Charge (nVs)")
        plt.yscale('log')
        plt.legend(loc = "best")
        fig = plt.gcf()
        pdf.savefig(fig, dpi = 100)
        
        plt.clf()
        color_counter = 0
        linestyle_counter = 0
        for sensor in final_plot:
            for channel in final_plot[sensor]:
                x_axis = final_plot[sensor][channel][0]
                #print(x_axis)

                if sensor=='Cz_M2':
                    y_axis = [v/0.024656973404562417 for v in final_plot[sensor][channel][1]]##0.024656973404562417 for 13db data, ## 0.2675723493130779 for 40db 
                    y_err  = [v/0.024656973404562417   for v in final_plot[sensor][channel][2]] 
                if sensor=='Cz_M3':
                    y_axis = [v/0.031230967316183182 for v in final_plot[sensor][channel][1]]##0.031230967316183182, 0.37675421015661653
                    y_err  = [v/0.031230967316183182 for v in final_plot[sensor][channel][2]]               
                if sensor=='Cz_M4':
                    y_axis = [v/0.03478349665285043 for v in final_plot[sensor][channel][1]]##0.03478349665285043 , 0.40983032371474054
                    y_err  = [v/0.03478349665285043 for v in final_plot[sensor][channel][2]]
                    #print(y_axis)
                if sensor=='EPI_M2':
                    y_axis = [v/0.011346677075655688 for v in final_plot[sensor][channel][1]]##0.011346677075655688, 0.13445607084067035
                    y_err  = [v/0.011346677075655688   for v in final_plot[sensor][channel][2]]
                if sensor=='EPI_M3':
                    y_axis = [v/0.014079314920107514 for v in final_plot[sensor][channel][1]]##0.014079314920107514, 0.15687367497284913
                    y_err  = [v/0.014079314920107514 for v in final_plot[sensor][channel][2]]
                if sensor=='EPI_M4':
                    y_axis = [v/0.013434635798924633 for v in final_plot[sensor][channel][1]]##0.013434635798924633, 0.1539891988126712
                    y_err  = [v/0.013434635798924633 for v in final_plot[sensor][channel][2]]

                if linestyle_counter % 2 == 0:
                    linestyle = "-"
                else:
                    linestyle = "--"
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        #plt.gca().invert_xaxis()
        plt.title("Gain against Bias Voltage")
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Gain")
        plt.yscale('log')
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
        #if i not in pad_positions:
         #   continue
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude) or amplitude > -0:
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < 98.5 or time_diff > 99.3:
                continue
            #peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            #print(peak_time)
            #if peak_time < 6 or peak_time > 8:
             #continue
            if math.isnan(time_diff):
                continue
            time_differences.append(time_diff)
            #print(time_differences)
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

def plot_time_resolution_everything(directory_in_str = "Data/Amp40db/"):
    final_plot = {} # {sensor: channel: ([voltages], [mean amplitude], [std amplitude (error)])}
    with PdfPages(f"Timing_Amp40db.pdf") as pdf:
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
                    #pad_positions[chan2] = get_pad_positions(data_file, positions_file, chan2)
                    break

            final_plot[filename][chan1] = ([],[],[])#; final_plot[filename][chan2] = ([],[],[])

            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                #print(folder)
                filename2 = os.fsdecode(folder)
                if filename2.startswith('.'): #      # to ignore any hidden file
                    continue                
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file         = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2       = f"{directory3_in_str}measured_data.sqlite"
                positions_file    = f"{directory3_in_str}positions.pickle"
                print(f"{filename}, {filename2}")
                #(voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)

                plt.clf()
                voltage = int(folder.rstrip('V'))
                (stdf, stdf_err) = plot_time_resolution_of_one_pad(data_file, chan1, pad_positions[chan1])
                final_plot[filename][chan1][0].append(voltage)
                final_plot[filename][chan1][1].append(stdf) 
                final_plot[filename][chan1][2].append(stdf_err)
                plt.title(f'{filename}, {filename2}, Channel {chan1}')
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
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        #plt.gca().invert_xaxis()
        plt.title("Time Resolution against Bias Voltage")
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Time Resolution (ns)")
        plt.legend(loc = "best")
        fig = plt.gcf()
        pdf.savefig(fig, dpi = 100)
    return None
    


def plot_toa_of_one_pad(datafile, channel, pad_positions):
    query_dataset(datafile)
    times = []
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    t_50_data = data['t_50 (s)']
    for i in range(n_position):
        #if i not in pad_positions:
         #   continue
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            if math.isnan(amplitude):# or amplitude > -0:
                continue
            toa = (t_50_data[i,j,1]) * 1e9
            if toa < 6.1 or toa >8:
                continue
            if math.isnan(toa):
                continue
            times.append(toa)
            #print(time_differences)
    mu, std = statistics.mean(times), statistics.stdev(times)
    bin_min = mu - 4 * std; bin_max = mu + 4 * std
    custom_bins = numpy.linspace(bin_min, bin_max, 50 ,endpoint=True)
    (n, bins, patches) = plt.hist(times, bins=custom_bins, density=True, stacked=False ,histtype='stepfilled', alpha = 0.5, lw=1, label=f"Data (${{\mu}}$ = {round(mu,2)}, ${{\sigma}}$ = {round(std,2)})")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    (muf, stdf), covf = curve_fit(gaussian, bin_centers, n, maxfev=10000, p0=[mu, std])
    stdf_err = math.sqrt(covf[1][1])

    fit_x_axis = numpy.linspace(min(bin_centers), max(bin_centers), 200, endpoint=True)
    fit_y_axis = gaussian(fit_x_axis, muf, stdf)
    plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
    plt.xlabel(f"Time of Arrival (50%) (ns)")
    plt.ylabel(f"Frequency")
    plt.legend(loc = "best")
    plt.title(f'{datafile[5:11]}, {datafile[12:16]}, Channel {channel}')
    return (stdf, stdf_err)




def plot_toa_everything(directory_in_str = "Data/Amp13db/"): #### Look only at the time of arrival of the first pulse
    final_plot = {} # {sensor: channel: ([voltages], [mean amplitude], [std amplitude (error)])}
    with PdfPages(f"Jitter_Amp13db.pdf") as pdf:
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
                    #pad_positions[chan2] = get_pad_positions(data_file, positions_file, chan2)
                    break

            final_plot[filename][chan1] = ([],[],[])#; final_plot[filename][chan2] = ([],[],[])

            for folder in sorted(
                (f.decode() if isinstance(f, bytes) else f
                 for f in os.listdir(directory2)
                 if not (f.decode() if isinstance(f, bytes) else f).startswith('.')),
                key=lambda x: int(x.rstrip('V'))
            ):
                #print(folder)
                filename2 = os.fsdecode(folder)
                if filename2.startswith('.'): #      # to ignore any hidden file
                    continue                
                directory3_in_str = f"{directory2_in_str}{filename2}/"
                data_file         = f"{directory3_in_str}parsed_from_waveforms.sqlite"
                data_file_2       = f"{directory3_in_str}measured_data.sqlite"
                positions_file    = f"{directory3_in_str}positions.pickle"
                print(f"{filename}, {filename2}")
                #(voltage, voltage_uncertainty) = get_bias_voltage(data_file_2)

                plt.clf()
                voltage = int(folder.rstrip('V'))
                (stdf, stdf_err) = plot_toa_of_one_pad(data_file, chan1, pad_positions[chan1])
                final_plot[filename][chan1][0].append(voltage)
                final_plot[filename][chan1][1].append(stdf) 
                final_plot[filename][chan1][2].append(stdf_err)
                plt.title(f'{filename}, {filename2}, Channel {chan1}')
                fig = plt.gcf()
                pdf.savefig(fig, dpi = 100)

                #plt.clf()
                #(stdf, stdf_err) = plot_time_resolution_of_one_pad(data_file, chan2, pad_positions[chan2])
                #final_plot[filename][chan2][0].append(voltage)
                #final_plot[filename][chan2][1].append(stdf) 
                #final_plot[filename][chan2][2].append(stdf_err)
                #plt.title(f'{filename}, {filename2}, Channel {chan2}')
                #fig = plt.gcf()
                #pdf.savefig(fig, dpi = 100)
                

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
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = CB_color_cycle[color_counter], label = f"{sensor}")
                plt.errorbar(x_axis, y_axis, yerr = y_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                linestyle_counter += 1
            color_counter += 1
        #plt.gca().invert_xaxis()
        plt.title("Jitter against Bias Voltage")
        plt.xlabel("Bias Voltage (V)")
        plt.ylabel("Jitter (ns)")
        plt.legend(loc = "best")
        fig = plt.gcf()
        pdf.savefig(fig, dpi = 100)
    return None
    









    
def plot_2d_maps(datafile,positions):
    query_dataset(datafile)
    (x,y) = get_positions(positions)
    (active_channel_1, active_channel_2) = determine_active_channels(datafile)
    amplitudes = {}
    collected_charges={}
    times={}
    connection = sqlite3.connect(datafile)
    channel=active_channel_1
    amplitudes[channel] = []
    collected_charges[channel] =[]
    times[channel] =[]
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `Collected charge (V s)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    collected_charge_data = data['Collected charge (V s)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    for i in range(n_position):
            result = []
            result2 = []
            results3 =[]
            for j in range(n_triggers):
                amplitude = amplitude_data[i,j,1]
                if math.isnan(amplitude):# or amplitude > -0.0:
                    continue
                collected_charge = collected_charge_data[i,j,1] * 1e9     
                if math.isnan(collected_charge):# or collected_charge > 0:
                    continue
                #time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
                #if time_diff < 98 or time_diff > 99.5:
                #    continue
                #peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
                #if peak_time < 4.5 or peak_time > 6.5:
                 #   continue
                result.append(amplitude)
                result2.append(collected_charge)
                #if math.isnan(time_diff):
                 #   continue
                #results3.append(time_diff)
            if result == []:
                amplitudes[channel].append(0)
            else:
                amplitudes[channel].append(statistics.mean(result))            
            if result2 ==[]:
               collected_charges[channel].append(0)
            else:
               collected_charges[channel].append(statistics.mean(result2))                
            
            
            if results3 ==[]:
               times[channel].append(0)
            else:
               times[channel].append(statistics.mean(results3))                

    fig, ax = plt.subplots( figsize=(10,10))
    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': amplitudes[1]})
    heatmap = data_frame.pivot_table(index='x', columns='y', values='z', aggfunc='mean')
    sns.heatmap(heatmap, annot=False, vmax=0, cmap="plasma_r", ax=ax, square=True, yticklabels = 5, xticklabels = 5) #vmax=0
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.xlabel(r"x ($\mu$m)",fontsize = 18)
    plt.ylabel(r"y ($\mu$m)",fontsize = 18)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Amplitude [V]",fontsize = 18)
    plt.tight_layout()
    plt.savefig('plots/Amps_EpiM2_Central_52V.pdf', format='pdf', dpi=1200)
    
    #plt.show()
        
    fig, ax = plt.subplots( figsize=(10,10))
    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': collected_charges[1]})
    heatmap = data_frame.pivot_table(index='x', columns='y', values='z', aggfunc='mean')
    sns.heatmap(heatmap, annot=False,  vmax=0, cmap="plasma_r", ax=ax, square=True, yticklabels = 5, xticklabels = 5,cbar_kws={"shrink": 0.8}) #vmax=0
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.xlabel(r"x ($\mu$m)",fontsize = 18)
    plt.ylabel(r"y ($\mu$m)",fontsize = 18)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Collected Charge [Vns]",fontsize = 18)
    plt.tight_layout()
    #plt.show()
    plt.savefig('plots/Charge_EpiM2_Central_52V.pdf', format='pdf', dpi=1200)
    
    return None  


def get_pad_positions(datafile, positions, channel): # gives only the central pad positions
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
            if math.isnan(amplitude) or amplitude > -0.0:
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < 98 or time_diff > 99.5:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < 4.5 or peak_time > 6.5:
                continue
            result.append(amplitude)
        if result == []:
            amplitudes.append(0)
        else:
            amplitudes.append(statistics.mean(result))
    data_frame = pandas.DataFrame({'x': x, 'y': y, 'z': amplitudes})
    points_from_edge = 10 # this is hardcoded number
    smallest_list = (pandas.Series(data_frame['y'].unique()).nsmallest(points_from_edge)).to_list()
    largest_list = (pandas.Series(data_frame['y'].unique()).nlargest(points_from_edge)).to_list()
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
        valid_y = smallest_list
    else:
        x_axis = edge2_projection.index.to_numpy()
        y_axis = edge2_projection.to_numpy()
        err = edge2_projection_sig.to_numpy()
        valid_y = largest_list
    peak_index = numpy.where(y_axis == min(y_axis))
    peak_error = err[peak_index]
    peak_max = min(y_axis) + peak_error; peak_min = min(y_axis) - peak_error
    valid_x = []
    for value in y_axis:
        value_index = numpy.where(y_axis == value)
        value_error = err[value_index]
        value_max = value + value_error; value_min = value - value_error
        if ((peak_min <= value_max).any() and (value_max <= peak_max).any()) or \
            ((peak_min <= value_min).any() and (value_min <= peak_max).any()) or \
            ((value_min <= peak_max).any() and (peak_max <= value_max).any()) or \
            ((value_min <= peak_min).any() and (peak_min <= value_max).any()):
            x_value = x_axis[value_index]
            valid_x.append(x_value[0])
    pad_position = [] # list of n_positions that is the pad
    position_frame = pandas.DataFrame({'x': x, 'y': y})
    for x_value in valid_x:
        for y_value in valid_y:
            row_index = position_frame.query(f"x == {x_value} and y == {y_value}").index[0]
            pad_position.append(row_index)
    return pad_position



###################################################################################################################
###################################################################################################################
###################################################################################################################
################################################### For testing ###################################################
print(f"Script started")
start_time = time.time()


#plot_amplitude_everything()  #"1_plot_amplitude_everything" in Desktop/analysis_tct/Plts_of_Analysis2 ##working - problem that a lot of noise is taken into the amplitude
#plot_collected_charge(test_datafile)  #"2_plot_collected_charge" in Desktop/analysis_tct/Plts_of_Analysis2 # working
#plot_amplitude(test_datafile, test_positions)   #"3_plot_amplitude" in Desktop/analysis_tct/Plts_of_Analysis2 # working
plot_2d_maps(test_datafile, test_positions)
#plot_collected_charge_everything()   #6_plot_collected_charge_everything  in Desktop/analysis_tct/Plts_of_Analysis2 # not working
#plot_time_resolution_everything()
#plot_toa_everything()

time_taken = round(time.time() - start_time)

minutes = time_taken // 60
seconds = time_taken - 60*minutes 
print(f"--- Runtime: {minutes} minutes {seconds} seconds ---")
################################################### For testing ###################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################