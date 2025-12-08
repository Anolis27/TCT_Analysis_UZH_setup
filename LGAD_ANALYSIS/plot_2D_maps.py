import os
import sqlite3
import math
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

from config import Paths, Colors, Filters
from data_manager import (
    gaussian,
    query_dataset,
    get_positions,
    determine_active_channels
)

def get_sensorname_from_path(base_dir):
    return os.path.basename(os.path.dirname(base_dir))


def compute_amplitude_and_charge(datafile, positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    (x, y) = get_positions(positions)
    (active_ch1, active_ch2) = determine_active_channels(datafile)

    ampl = {active_ch1: [], active_ch2: []}
    charge = {active_ch1: [], active_ch2: []}

    connection = sqlite3.connect(datafile)

    for channel in (active_ch1, active_ch2):
        df = pd.read_sql(
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

        for i in range(n_position):
            A, C = [], []
            for j in range(n_triggers):
                amp = amplitude.get((i,j,1), np.nan)
                chg = collected.get((i,j,1), np.nan)

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

    # somme ch1 + ch2
    ampl["sum"] = [a + b for a,b in zip(ampl[active_ch1], ampl[active_ch2])]
    charge["sum"] = [a + b for a,b in zip(charge[active_ch1], charge[active_ch2])]

    return x, y, ampl, charge, active_ch1, active_ch2


def plot_heatmap(x, y, z, pdf, unit, title, max_v= 0):
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,10))

    data_frame = pd.DataFrame({'x': x, 'y': y, 'z': z})
    heatmap = data_frame.pivot_table(
        index='x', columns='y', values='z', aggfunc='mean'
    )

    sns.heatmap(
        heatmap, annot=False, cmap="plasma_r", vmax=max_v,
        square=True, ax=ax, yticklabels=5, xticklabels=5,
        cbar_kws={"shrink": 0.5}
    )

    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.title(title, fontsize=20)
    plt.xlabel(r"x ($\mu$m)", fontsize=18)
    plt.ylabel(r"y ($\mu$m)", fontsize=18)
    cbar = ax.collections[0].colorbar
    cbar.set_label(unit, fontsize=18)

    plt.tight_layout()
    pdf.savefig(fig)



# ============================================================
# 1) AMPLITUDE
# ============================================================
def plot_2d_amplitude(datafile, positions):

    base_dir = os.path.dirname(datafile)
    sensorname = get_sensorname_from_path(base_dir)

    x, y, ampl, _, ch1, ch2 = compute_amplitude_and_charge(datafile, positions)
    with PdfPages(f"2d_amps_{sensorname}.pdf") as pdf:
        plot_heatmap(x, y, ampl[ch1], pdf, "Amplitude [V]",
                    f"{sensorname} Ch{ch1}")

        plot_heatmap(x, y, ampl[ch2], pdf, "Amplitude [V]",
                    f"{sensorname} Ch{ch2}")

        plot_heatmap(x, y, ampl["sum"], pdf, "Amplitude [V]",
                    f"{sensorname} Ch{ch1}&{ch2}")
        pdf.close()


# ============================================================
# 2) COLLECTED CHARGE
# ============================================================
def plot_2d_charge(datafile, positions):

    base_dir = os.path.dirname(datafile)
    sensorname = get_sensorname_from_path(base_dir)


    x, y, _, charge, ch1, ch2 = compute_amplitude_and_charge(datafile, positions)

    with PdfPages(f"2d_charge_{sensorname}.pdf") as pdf:
        plot_heatmap(x, y, charge[ch1], pdf, "Collected charge [Vns]",
                     f"{sensorname} Ch{ch1}")

        plot_heatmap(x, y, charge[ch2], pdf, "Collected charge [Vns]",
                     f"{sensorname} Ch{ch2}")

        plot_heatmap(x, y, charge["sum"], pdf, "Collected charge [Vns]",
                     f"{sensorname} Ch{ch1}&{ch2}")
        pdf.close()


def compute_timing(datafile, positions):
    n_position, n_triggers, n_channels = query_dataset(datafile)
    (x, y) = get_positions(positions)
    (ch1, ch2) = determine_active_channels(datafile)

    times = {ch1: [], ch2: []}

    connection = sqlite3.connect(datafile)

    for channel in (ch1, ch2):
        df = pd.read_sql(
            f"""SELECT n_position,n_trigger,n_pulse, 
                    `t_90 (s)`, `Time over 90% (s)`,
                    `Amplitude (V)`, `Collected charge (V s)`, `t_50 (s)` 
                FROM dataframe_table 
                WHERE n_channel=={channel}""",
            connection
        )
        df.set_index(['n_position','n_trigger','n_pulse'], inplace=True)

        amplitude_data = df['Amplitude (V)']
        charge_data = df['Collected charge (V s)']
        t50 = df['t_50 (s)']

        for i in range(n_position):
            time_diffs = []
            for j in range(n_triggers):
                try:
                    time_diff = (t50[i,j,2] - t50[i,j,1]) * 1e9  # ns
                    if math.isnan(time_diff):
                        continue
                    amplitude = amplitude_data[i,j,1]
                    if math.isnan(amplitude):# or amplitude > -0.0:
                        continue
                    collected_charge = charge_data[i,j,1] * 1e9     
                    if math.isnan(collected_charge):# or collected_charge > 0:
                        continue
    
                    if time_diff < Filters.TIME_DIFF_MIN or time_diff > Filters.TIME_DIFF_MAX:
                        continue
                    time_diffs.append(time_diff)
                except KeyError:
                    continue  # some entries missing

            if len(time_diffs) < 10:
                times[channel].append(np.nan)
                continue

            mu = statistics.mean(time_diffs)
            std = statistics.stdev(time_diffs)

            counts, edges = np.histogram(
                time_diffs, bins=50, range=(mu-4*std, mu+4*std), density=True
            )
            centers = 0.5*(edges[:-1]+edges[1:])

            try:
                (mu_fit, sigma_fit), _ = curve_fit(
                    gaussian,
                    centers, counts, p0=[mu, std], maxfev=10000
                )
            except:
                sigma_fit = np.nan

            times[channel].append(sigma_fit)

    # combine ch1/ch2 : min resolution
    # times["sum"] = [
    #     min(a,b) if not (math.isnan(a) or math.isnan(b)) else (a if not math.isnan(a) else b)
    #     for a,b in zip(times[ch1], times[ch2])
    # ]
    times["sum"] = []
    for t1, t2 in zip(times[ch1], times[ch2]):
        if math.isnan(t1) and math.isnan(t2):
            times["sum"].append(np.nan)
        elif math.isnan(t1):
            times["sum"].append(t2)
        elif math.isnan(t2):
            times["sum"].append(t1)
        else:
            times["sum"].append(min(t1, t2))

    return x, y, times, ch1, ch2

# ============================================================
# 3) TIMING
# ============================================================
def plot_2d_timing(datafile, positions):

    base_dir = os.path.dirname(datafile)
    sensorname = get_sensorname_from_path(base_dir)
    vmax = 0.1

    x, y, times, ch1, ch2 = compute_timing(datafile, positions)
    
    with PdfPages(f"2d_timing_{sensorname}.pdf") as pdf:
        plot_heatmap(x, y, times[ch1], pdf, "Time [ns]",
                    f"{sensorname} Ch{ch1}", max_v=vmax)

        plot_heatmap(x, y, times[ch2], pdf, "Time [ns]",
                    f"{sensorname} Ch{ch2}", max_v=vmax)

        plot_heatmap(x, y, times["sum"], pdf, "Time resolution [ns]",
                    f"{sensorname} Ch{ch1}&{ch2}", max_v=vmax)
        pdf.close()