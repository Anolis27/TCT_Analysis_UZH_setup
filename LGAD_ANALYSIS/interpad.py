# interpad.py
from config import Paths, Colors, Filters, InterpadConfig
from data_manager import *
from amplitude import *
from timing import *
import statistics
import sqlite3
import pandas
import numpy
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages


def sigmoid(x, x0, b, L, k):
    y = L / (1 + numpy.exp(-k*(x-x0))) + b
    return (y)

def get_interpad_distance(datafile, positions, channel1, sensor_strip_positions1,
                          channel2, sensor_strip_positions2, pdf):

    channel1_result = project_onto_y_one_channel(datafile, positions, channel1, sensor_strip_positions1)
    channel2_result = project_onto_y_one_channel(datafile, positions, channel2, sensor_strip_positions2)
    # both_channels   = project_onto_y_two_channels(datafile, positions, channel1, channel2,
    #                                               sensor_strip_positions1, sensor_strip_positions2, pdf)
    plt.clf()

    # ============================
    # Channel 1
    # ============================
    x_axis = numpy.array(channel1_result["x axis"])
    y_axis = numpy.array(channel1_result["y axis"])
    y_err  = numpy.array(channel1_result["y error"])
    #print(y_axis)

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

    # === get x at 50% ===
    ch1_x0 = popt[0]
    ch1_L = popt[2]
    ch1_k = popt[3]
    print(f"ch1_k: {ch1_k}, ch1_L: {ch1_L}, ch1_x0: {ch1_x0}")
    ch1_x0_uncertainty = math.sqrt(pcov[0][0])
    # === Compute x at INTERPAD_FRACTION of the sigmoid ===
    ch1_xfrac = ch1_x0 - (1/ch1_k) * numpy.log(1/(InterpadConfig.INTERPAD_FRACTION if ch1_L > 0 else (1 - InterpadConfig.INTERPAD_FRACTION)) - 1)
    # === Compute uncertainty on x at INTERPAD_FRACTION of the sigmoid ===
    # partial derivatives
    ch1_dx_dx0 = 1
    ch1_dx_dk  = numpy.log(1/InterpadConfig.INTERPAD_FRACTION - 1) / (ch1_k**2)
    # uncertainties
    ch1_sigma_x0 = numpy.sqrt(pcov[0,0])
    ch1_sigma_k  = numpy.sqrt(pcov[3,3])
    cov_x0k  = pcov[0,3]  # covariance x0-k
    # error propagation
    ch1_sigma_xfrac = numpy.sqrt( (ch1_dx_dx0*ch1_sigma_x0)**2 + (ch1_dx_dk*ch1_sigma_k)**2 + 2*ch1_dx_dx0*ch1_dx_dk*cov_x0k )


    # --- Generate fit curve ---
    fine_x = numpy.linspace(x_axis.min(), x_axis.max(), 500)
    fit_norm = sigmoid(fine_x, *popt)

    # --- Plot Ch 1 ---
    plt.plot(fine_x, fit_norm, "-", label=f"Ch {channel1} Fit", color=Colors.CB_CYCLE[0])
    plt.plot(x_axis, y_norm, ".", markersize=3, label=f"Ch {channel1} Data", color=Colors.CB_CYCLE[0])
    plt.errorbar(x_axis, y_norm, yerr=y_err_norm, ls="none", ecolor="k",
                 elinewidth=1, capsize=2)
    plt.axvline(x=ch1_xfrac, ymin=0, ymax=1, color='k', label=f'x{channel1} at {InterpadConfig.INTERPAD_FRACTION*100}% = {round(ch1_xfrac,2)}', ls = (0, (5, 10)), linewidth=0.6)

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

    # === get x at 50% ===
    ch2_x0 = popt[0]
    ch2_L = popt[2]
    ch2_k = popt[3]
    print(f"ch2_k: {ch2_k}, ch2_L: {ch2_L}, ch2_x0: {ch2_x0}")
    ch2_x0_uncertainty = math.sqrt(pcov[0][0])
    # === Compute x at INTERPAD_FRACTION of the sigmoid ===
    ch2_xfrac = ch2_x0 - (1/ch2_k) * numpy.log(1/(InterpadConfig.INTERPAD_FRACTION if ch2_L > 0 else (1 - InterpadConfig.INTERPAD_FRACTION)) - 1)
    # === Compute uncertainty on x at INTERPAD_FRACTION of the sigmoid ===
    # partial derivatives
    ch2_dx_dx0 = 1
    ch2_dx_dk  = numpy.log(1/InterpadConfig.INTERPAD_FRACTION - 1) / (ch2_k**2)
    # uncertainties
    ch2_sigma_x0 = numpy.sqrt(pcov[0,0])
    ch2_sigma_k  = numpy.sqrt(pcov[3,3])
    cov_x0k  = pcov[0,3]  # covariance x0-k
    # error propagation
    ch2_sigma_xfrac = numpy.sqrt( (ch2_dx_dx0*ch2_sigma_x0)**2 + (ch2_dx_dk*ch2_sigma_k)**2 + 2*ch2_dx_dx0*ch2_dx_dk*cov_x0k )

    # --- Fit curve ---
    fine_x = numpy.linspace(x_axis.min(), x_axis.max(), 500)
    fit_norm = sigmoid(fine_x, *popt)

    # --- Plot Ch 2 ---
    plt.plot(fine_x, fit_norm, "-", label=f"Ch {channel2} Fit", color=Colors.CB_CYCLE[1])
    plt.plot(x_axis, y_norm, ".", markersize=3, label=f"Ch {channel2} Data", color=Colors.CB_CYCLE[1])
    plt.errorbar(x_axis, y_norm, yerr=y_err_norm, ls="none", ecolor="k",
                 elinewidth=1, capsize=2)
    plt.axvline(x=ch2_xfrac, ymin=0, ymax=1, color='k', label=f'x{channel2} at {InterpadConfig.INTERPAD_FRACTION*100}% = {round(ch2_xfrac,2)}', ls = (0, (5, 10)), linewidth=0.6)


    # ============================
    # Sum of channels (no fit)
    # ============================
    # plt.plot(both_channels["x axis"], both_channels["y axis"], ".-",
    #          markersize=3, linewidth=1, color=Colors.CB_CYCLE[3],
    #          label=f"Ch {channel1}+{channel2} Data")
    # plt.errorbar(both_channels["x axis"], both_channels["y axis"],
    #              yerr=both_channels["y error"], ls="none",
    #              ecolor="k", elinewidth=1, capsize=2)

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

    return (abs(ch1_xfrac - ch2_xfrac), math.sqrt(ch1_sigma_xfrac**2 + ch2_sigma_xfrac**2))



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
            plt.plot(x_ax, y_ax, 'o', label=f"{key}", markersize=3, color=Colors.CB_CYCLE[color_index], linestyle="-", linewidth=1)
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
    save_results(result, "Interpad_distance")
    return None

def plot_time_resolution_interpad_region(datafile, positions, pdf):
    (chan1, chan2) = determine_active_channels(datafile)
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
                if math.isnan(amplitude) or amplitude > Filters.AMPLITUDE_THRESHOLD:
                    continue
                time_diff = (t_50_data[i,j,2, channel] - t_50_data[i,j,1, channel]) * 1e9
                if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                    continue
                peak_time = (t_90_data[i,j,1, channel] + 0.5 * time_over_90_data[i,j,1, channel]) * 1e9
                if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
                    continue
                if y[i] < InterpadConfig.INTERPAD_REGION_MIN or y[i] > InterpadConfig.INTERPAD_REGION_MAX: # only interpad region
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
            markersize=3, linewidth=1, color=Colors.CB_CYCLE[0],
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
                if filename not in final_plot:
                    final_plot[filename] = {}
                final_plot[filename][filename2] = plot_time_resolution_interpad_region(data_file, positions_file, pdf)
            plt.clf()
            plt.title(f"Time Resolution in Interpad Region vs bias Voltage, {filename}")
            plt.xlabel(r"x Position ($\mu$m)")
            plt.ylabel("Time Resolution (ns)")
            plt.gca().invert_xaxis()
            plt.tight_layout()
            colors_index = 0
            for key in final_plot[filename]:
                x_ax = final_plot[filename][key]["x axis"]
                y_ax = final_plot[filename][key]["y axis"]
                y_ax_err = final_plot[filename][key]["y error"]
                plt.plot(x_ax, y_ax, 'o-', label=f"{key}", markersize=3, color=Colors.CB_CYCLE[colors_index], linewidth=1)
                plt.errorbar(x_ax, y_ax, yerr = y_ax_err, ls='none', ecolor = 'k', elinewidth = 1, capsize = 2)
                colors_index += 1
            plt.ylim(bottom=InterpadConfig.INTERPAD_TIMING_SCALE[0], top=InterpadConfig.INTERPAD_TIMING_SCALE[1]) # ajust scale
            #plt.legend(loc='upper right', ncol = 1)
            plt.legend(loc='best', ncol = 1)
            # plt.legend(
            #     loc='center left',
            #     bbox_to_anchor=(1, 0.5),
            #     fontsize='small'
            # )
            # plt.tight_layout()
            fig = plt.gcf() # get current figure
            pdf.savefig(fig, dpi = 100, bbox_inches='tight')
        save_results(final_plot, "Timing_interpad_region")
    return None
