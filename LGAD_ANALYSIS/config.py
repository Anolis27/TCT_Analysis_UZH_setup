# config.py
import os

class Paths:
    BASE_DIR = os.path.expanduser("C:/Users/mathi/Documents/UZH/LGAD_ANALYSIS/Data/S2_V2_TW5/50V")
    DATAFILE = os.path.join(BASE_DIR, "parsed_from_waveforms.sqlite")
    DATAFILE2 = os.path.join(BASE_DIR, "measured_data.sqlite")
    POSITIONS = os.path.join(BASE_DIR, "positions.pickle")
    SAVE_DIR = "saved_results"

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

class Filters:  # key values for Ti-LGAD
    AMPLITUDE_THRESHOLD = -0.07  # V    -0.07
    TIME_DIFF_MIN = 98      # ns    98
    TIME_DIFF_MAX = 99.5     # ns   99.5
    PEAK_TIME_MIN = 4.5     # ns    4.5
    PEAK_TIME_MAX = 7     # ns    6.5

class Subplots:
    AMPLITUDE_ONE_PAD = True
    CHARGE_COLLECTION_ONE_PAD = True
    TIMING_ONE_PAD = True

class InterpadConfig:
    INTERPAD_REGION_MIN = -75  # um
    INTERPAD_REGION_MAX = 75    # um
    INTERPAD_FRACTION = 0.9     # fraction of the sigmoid used for interpad distance calculation
    INTERPAD_TIMING_SCALE = (0.01, 0.085)  # y-axis limits for interpad timing plots (ns)