from data_manager import load_results
from config import InterpadConfig, Paths, Colors
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_saved_results(analysis):
    save_dir = os.path.join(Paths.SAVE_DIR, analysis)
    if not os.path.exists(save_dir):
        print(f"No saved results found for analysis: {analysis}")
        return
    
    # ============================================================
    # CASE 1 : ANALYSES Amplitude, Charge, Timing
    # ============================================================
    if analysis in ("Amplitude", "Collected_charge", "Timing"):
        plt.close()
        plt.figure(figsize=(8,6))
        linestyle_counter = 0
        color_counter = 0

        for filename in os.listdir(save_dir):
            if filename.endswith(".pkl"):
                filepath = os.path.join(save_dir, filename)
                (data_type, data) = load_results(filepath)
                sensor = filename.rstrip(".pkl")

                for channel, (voltages, means, stds) in data.items():
                    linestyle = "-" if linestyle_counter % 2 == 0 else "--"
                    plt.plot(
                        voltages, means,
                        marker="o", markersize=2,
                        linestyle=linestyle, linewidth=1,
                        color=Colors.CB_CYCLE[color_counter],
                        label=f"{sensor}, Ch {channel}"
                    )
                    plt.errorbar(
                        voltages, means, yerr=stds,
                        ls="none", ecolor="k",
                        elinewidth=1, capsize=2
                    )
                    linestyle_counter += 1

                color_counter += 1

        # Titles
        if analysis == "Amplitude":
            plt.title("Amplitude against Bias Voltage")
            plt.ylabel("Mean Amplitude (V)")
        elif analysis == "Collected_charge":
            plt.title("Collected Charge against Bias Voltage")
            plt.ylabel("Mean Collected Charge (Vns)")
        elif analysis == "Timing":
            plt.title("Jitter against Bias Voltage")
            plt.ylabel("Jitter (ns)")

        plt.xlabel("Bias Voltage (V)")
        plt.gca().invert_xaxis()
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{analysis}_saved_results.pdf", format='pdf', dpi=1200)
        #plt.show()
        return
    

    # ============================================================
    # CASE 2 : TIMING_INTERPAD_REGION
    # ============================================================
    if analysis == "Timing_interpad_region":
        pdf_path = f"{analysis}_saved_results.pdf"
        with PdfPages(pdf_path) as pdf:

            # iterate on sensors (directories)
            for sensor in sorted(os.listdir(save_dir)):
                sensor_dir = os.path.join(save_dir, sensor)
                if not os.path.isdir(sensor_dir):
                    continue

                plt.close()
                plt.figure(figsize=(8,6))
                color_index = 0

                # iterate on voltage files inside each sensor
                for file in sorted(os.listdir(sensor_dir), key=lambda x: int(x.rstrip("V.pkl"))):
                    if not file.endswith(".pkl"):
                        continue

                    filepath = os.path.join(sensor_dir, file)
                    voltage = file.replace(".pkl", "")

                    (data_type, data) = load_results(filepath)

                    # data = { "x axis": [...], "y axis": [...], "y error": [...] }
                    x = data["x axis"]
                    y = data["y axis"]
                    yerr = data["y error"]

                    plt.plot(
                        x, y,
                        "o-", markersize=3, linewidth=1,
                        color=Colors.CB_CYCLE[color_index],
                        label=f"{voltage}"
                    )
                    plt.errorbar(
                        x, y,
                        yerr=yerr,
                        ls="none", ecolor="k",
                        elinewidth=1, capsize=2
                    )

                    color_index += 1

                # Final formatting of the sensor plot
                plt.title(f"Jitter in Interpad Region — {sensor}")
                plt.xlabel("y Position (µm)")
                plt.ylabel("Jitter (ns)")
                plt.ylim(bottom=InterpadConfig.INTERPAD_TIMING_SCALE[0], top=InterpadConfig.INTERPAD_TIMING_SCALE[1]) # ajust scale
                plt.gca().invert_xaxis()
                plt.legend(loc="best")
                plt.tight_layout()

                # Save this figure into the PDF
                pdf.savefig()
        
        print(f"[OK] PDF saved: {pdf_path}")
        return
    
    if analysis == "Interpad_distance":
        pdf_path = f"{analysis}_saved_results.pdf"
        with PdfPages(pdf_path) as pdf:
            plt.close()
            plt.figure(figsize=(8,6))
            color_counter = 0

            for filename in os.listdir(save_dir):
                if filename.endswith(".pkl"):
                    sensor = filename.rstrip(".pkl")
                    filepath = os.path.join(save_dir, filename)
                    (data_type, data) = load_results(filepath)
                    
                    x_axis = data["x"]
                    y_axis = data["y"]
                    y_error = data["y_err"]
                    x_error = data["x_err"]
                    plt.plot(
                        x_axis, y_axis,
                        "o-", markersize=3, linewidth=1,
                        color=Colors.CB_CYCLE[color_counter],
                        label=f"{sensor}"
                    )
                    plt.errorbar(
                        x_axis, y_axis,
                        yerr=y_error,
                        xerr=x_error,
                        ls="none", ecolor="k",
                        elinewidth=1, capsize=2
                    )
                    color_counter += 1
            plt.title("Interpad Distance against Bias Voltage")
            plt.xlabel("Bias Voltage (V)")
            plt.gca().invert_xaxis()
            plt.ylabel("Interpad Distance (µm)")
            plt.legend(loc="best")
            plt.tight_layout()
            pdf.savefig()
        
        print(f"[OK] PDF saved: {pdf_path}")
        return
    
    # ============================================================
    # UNKNOWN ANALYSIS
    # ============================================================
    print(f"Unknown analysis type: {analysis}")
    return
