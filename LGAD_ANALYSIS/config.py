# config.py
import os

class Paths:
    DATA_ROOT = os.path.expanduser("C:/Users/mathi/Documents/UZH/LGAD_ANALYSIS/Data")
    PREFERRED_BASE_DIR = os.path.expanduser(
        "C:/Users/mathi/Documents/UZH/LGAD_ANALYSIS/Data/W5_V2_TW5/100V"
    )
    SAVE_DIR = "saved_results"

    BASE_DIR = None
    DATAFILE = None
    DATAFILE2 = None
    POSITIONS = None

    @classmethod
    def resolve(cls):
        if os.path.isdir(cls.PREFERRED_BASE_DIR):
            base_dir = cls.PREFERRED_BASE_DIR
        else:
            sensor_dirs = [
                d for d in os.listdir(cls.DATA_ROOT)
                if os.path.isdir(os.path.join(cls.DATA_ROOT, d))]
            if not sensor_dirs:
                raise FileNotFoundError(
                    f"No folder found in {cls.DATA_ROOT}")
            sensor_dir = os.path.join(cls.DATA_ROOT, sensor_dirs[0])
            print(f"[INFO] Sensor auto-detected : {sensor_dirs[0]}")
            voltage_dirs = sorted(
                d for d in os.listdir(sensor_dir)
                if os.path.isdir(os.path.join(sensor_dir, d)))
            if not voltage_dirs:
                raise FileNotFoundError(
                    f"No voltage folder found in {sensor_dir}")
            base_dir = os.path.join(sensor_dir, voltage_dirs[0])
            print(f"[INFO] Voltage auto-detected : {voltage_dirs[0]}")
        cls.BASE_DIR = base_dir
        cls.DATAFILE  = os.path.join(base_dir, "parsed_from_waveforms.sqlite")
        cls.DATAFILE2 = os.path.join(base_dir, "measured_data.sqlite")
        cls.POSITIONS = os.path.join(base_dir, "positions.pickle")


class Filters:  # key values for Ti-LGAD
    AMPLITUDE_THRESHOLD = -0.05  # V    -0.07
    TIME_DIFF_MIN = 98      # ns    98
    TIME_DIFF_MAX = 99.5     # ns   99.5
    PEAK_TIME_MIN = 4.5     # ns    4.5
    PEAK_TIME_MAX = 7     # ns    6.5

class InterpadConfig:
    INTERPAD_REGION_MIN = - 75 # um
    INTERPAD_REGION_MAX = 75    # um
    X_STEP = 5                 # um step size
    Y_POSITION_MID_PAD = 5     # um y position for plot_amplitude_along_y_axis and time_resolution_interpad_region
    INTERPAD_FRACTION = 0.9     # fraction of the sigmoid used for interpad distance calculation
    INTERPAD_TIMING_SCALE = (0.0, 0.1)  # y-axis limits for interpad timing plots (ns)
    POINTS_FROM_EDGE_STRIP_POSITION = 2  # number of points from edge not to consider (usually 2)
    
class Subplots:
    AMPLITUDE_ONE_PAD = True
    CHARGE_COLLECTION_ONE_PAD = True
    TIMING_ONE_PAD = True

class PlotsConfig:  # config for heatmap plots + time difference t50 histograms
    V_MAX_AMP=True
    V_MAX_CHARGE=True
    V_MAX_TIMING=True
    AMPLITUDE_V_MAX = 0
    CHARGE_COLLECTION_V_MAX = 0
    TIMING_V_MAX = 0.1
    TIME_DIFF_T50_N_BINS = 50
    TIME_DIFF_T50_BIN_MIN = 97.5
    TIME_DIFF_T50_BIN_MAX = 100

class Colors:
    CB_CYCLE = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
    '#984ea3', '#999999', '#e41a1c', '#dede00',
    '#56B4E9', 
    '#E69F00', 
    '#009E73', 
    '#F0E442', 
    '#0072B2', 
    '#CC79A7'  
]

    BASIC = [
        "Red","Green","Blue","Yellow","Cyan","Magenta","Orange","Purple",
        "Pink","Brown","Black","White","Gray","DarkRed","DarkGreen",
        "DarkBlue","LightGray","LightGreen","LightBlue","LightCoral"
    ]
