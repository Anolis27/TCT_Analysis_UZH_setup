# data_manager.py
from config import Paths, Colors, Filters
from scipy import stats
from scipy.optimize import curve_fit
import statistics
import sqlite3
import pandas
import numpy
import pickle
import math
import statistics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib import figure

def gaussian(x, mu, sig):
    return 1./(numpy.sqrt(2.*numpy.pi)*sig)*numpy.exp(-numpy.power((x - mu)/sig, 2.)/2)

def query_dataset(datafile):
    #print(f"Querying dataset...")
    #global n_position; global n_triggers; global n_channels
    connection = sqlite3.connect(datafile)
    query = "SELECT n_position FROM dataframe_table WHERE n_trigger = 0 and n_pulse = 1 and n_channel = 1"
    filtered_data = pandas.read_sql_query(query, connection)
    n_position = len(filtered_data)
    query = "SELECT n_trigger FROM dataframe_table WHERE n_position = 0 and n_pulse = 1 and n_channel = 1"
    filtered_data = pandas.read_sql_query(query, connection)
    n_triggers = len(filtered_data)
    query = "SELECT n_channel FROM dataframe_table WHERE n_position = 0 and n_pulse = 1 and n_trigger = 0"
    filtered_data = pandas.read_sql_query(query, connection)
    n_channels = len(filtered_data)
    connection.close()
    #print(f"{n_position} positions, {n_triggers} triggers and {n_channels} channels found")
    return n_position, n_triggers, n_channels

def get_positions(positions):
    # get the (x,y) positions from the saved data
    # print(f"Getting position data...")
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
            amplitudes.append(0)
        else:
            amplitudes.append(statistics.mean(result))
    return amplitudes

# Hardcoded to return channels 1 and 2 for now
def determine_active_channels(datafile):
##    n_position, n_triggers, n_channels = query_dataset(datafile)
##    result = {}; list_to_sort = []
##    for channel in range(1, n_channels + 1):
##        amplitudes = get_channel_amplitude(datafile, channel, pulse_no = 1, method = "median")
##        result[round(sum(amplitudes),3)] = channel
##        list_to_sort.append(round(sum(amplitudes),3))
##        list_to_sort = sorted(list_to_sort)
##    return tuple(sorted((result[list_to_sort[0]], result[list_to_sort[1]])))
    return (1,2)


def get_pad_positions(datafile, positions, channel): 
    # retrurns list of position indices that correspond to signal
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
    if not pad_position:
        print(f"[Channel {channel}] No non-zero amplitudes found -> no pad positions")
    return pad_position

def plot_pad_positions(datafile, positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    x, y = get_positions(positions)  # lists of positions in µm
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
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
            if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                continue
            peak_time = (t_90_data[i,j,1] + 0.5 * time_over_90_data[i,j,1]) * 1e9
            if peak_time < Filters.PEAK_TIME_MIN or peak_time > Filters.PEAK_TIME_MAX:
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
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
    n_position, n_triggers, n_channels = query_dataset(datafile)
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
            if math.isnan(amplitude1) or amplitude1 > Filters.AMPLITUDE_THRESHOLD:
                amplitude1 = 0
            if time_diff1 < Filters.TIME_DIFF_MIN or time_diff1 > Filters.TIME_DIFF_MAX:
                amplitude1 = 0
            if peak_time1 < Filters.PEAK_TIME_MIN or peak_time1 > Filters.PEAK_TIME_MAX:
                amplitude1 = 0
            if math.isnan(amplitude2) or amplitude2 > Filters.AMPLITUDE_THRESHOLD: # FOR SOME REASON WAS 0
                amplitude2 = 0
            if time_diff2 < Filters.TIME_DIFF_MIN or time_diff2 > Filters.TIME_DIFF_MAX:
                amplitude2 = 0
            if peak_time2 < Filters.PEAK_TIME_MIN or peak_time2 > Filters.PEAK_TIME_MAX:
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

        plt.title(f"Histogram amplitudes — y = {y_position}")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.legend()

        #plt.show()
    return result