# SCRIPT FOR CHARGE COLLECTION ANALYSIS
from config import Paths, Colors, Filters
from amplitude import *
from data_manager import *
import sqlite3
import pandas
import numpy
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import math
import statistics


def plot_collected_charge(datafile):
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
        ax1.hist(collected_charges[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=Colors.CB_CYCLE[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(collected_charges[channel])) * numpy.ones(len(collected_charges[channel])))
        ax1.set_xlabel(r"Collected Charge (n V s)")
        ax1.set_ylabel(f"Frequency")
        ax1.legend(loc = "best")

    for channel in (3,4):
        bin_min = -0.1; bin_max = 0.02       # HARDCODED BINS ??
        bin_min = -0.02
        custom_bins = numpy.linspace(bin_min, bin_max, 100 ,endpoint=True)
        ax2.hist(collected_charges[channel], bins=custom_bins, stacked=False ,histtype='step', edgecolor=Colors.CB_CYCLE[channel], lw=1, label=f"Channel {channel}", weights= (1 / len(collected_charges[channel])) * numpy.ones(len(collected_charges[channel])))
        ax2.set_xlabel(r"Collected Charge (n V s)")
        ax2.set_ylabel(f"Frequency")
        ax2.legend(loc = "best")
    fig.suptitle(f'Collected Charge')
    plt.show()
    return None

def plot_collected_charge_of_one_pad(datafile, channel, pad_positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
            if math.isnan(amplitude) or amplitude > Filters.AMPLITUDE_THRESHOLD:
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
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
                plt.plot(x_axis, y_axis, marker = "o", markersize = 2, linestyle = linestyle, linewidth = 1, color = Colors.CB_CYCLE[color_counter], label = f"{sensor}, Ch {channel}")
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
        save_results(final_plot, analysis="Collected_charge")
    return None

def project_onto_y_one_channel_charge(datafile, positions, channel, sensor_strip_positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    (x,y) = get_positions(positions)
    amplitudes = {} # {y position: [list of amplitudes]}
    collected_charges = {} # {y position: [list of collected charge]}
    connection = sqlite3.connect(datafile)
    data = pandas.read_sql(f"SELECT n_position,n_trigger,n_pulse, `t_90 (s)`, `Time over 90% (s)`,`Amplitude (V)`, `Collected charge (V s)`, `t_50 (s)` FROM dataframe_table WHERE n_channel=={channel}", connection)
    data.set_index(['n_position','n_trigger','n_pulse'], inplace=True)
    amplitude_data = data['Amplitude (V)']
    collected_charge_data = data['Collected charge (V s)']
    t_50_data = data['t_50 (s)']
    t_90_data = data["t_90 (s)"]
    time_over_90_data = data['Time over 90% (s)']
    # all_amps = amplitude_data.values
    # min_val = numpy.nanmin(all_amps)
    # max_val = numpy.nanmax(all_amps)

    for i in sensor_strip_positions:
        for j in range(n_triggers):
            amplitude = amplitude_data[i,j,1]
            collected_charge = collected_charge_data[i,j,1] * 1e9
            if math.isnan(amplitude) or amplitude > 0.02: 
                continue
            time_diff = (t_50_data[i,j,2] - t_50_data[i,j,1]) * 1e9
            if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
                continue

            if y[i] not in collected_charges:
                collected_charges[y[i]] = []
            collected_charges[y[i]].append(collected_charge)

    result = {"x axis": [], "y axis": [], "y error": []}

    for y_position in sorted(collected_charges):
        hist = collected_charges[y_position]
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

        # plt.title(f"Histogram collected charge — y = {y_position}")
        # plt.xlabel("Collected Charge (V s)")
        # plt.ylabel("Frequency")
        # plt.legend()
        # fig = plt.gcf()
        # plt.plot(fit_x_axis, fit_y_axis, color = "r", label=f"Fit (${{\mu}}$ = {round(muf,2)}, ${{\sigma}}$ = {round(stdf,2)})")
        # pdf.savefig(fig, dpi = 100)
        # plt.show()
    return result