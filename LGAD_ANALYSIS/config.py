# config.py
import os

class Paths:
    BASE_DIR = os.path.expanduser("C:/Users/mathi/Documents/UZH/LGAD_ANALYSIS/Data/V2_TW5/50V")
    DATAFILE = os.path.join(BASE_DIR, "parsed_from_waveforms.sqlite")
    DATAFILE2 = os.path.join(BASE_DIR, "measured_data.sqlite")
    POSITIONS = os.path.join(BASE_DIR, "positions.pickle")

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

class Filters:
    AMPLITUDE_THRESHOLD = -0.07  # V
    TIME_DIFF_MIN = 98      # ns
    TIME_DIFF_MAX = 99.5     # ns
    PEAK_TIME_MIN = 4.5     # ns
    PEAK_TIME_MAX = 6.5     # ns
    INTERPAD_REGION_MIN = -75  # um
    INTERPAD_REGION_MAX = 75    # um
