# SCRIPT FOR TIMING ANALYSIS

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
    with PdfPages(f"Output3.pdf") as pdf:
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