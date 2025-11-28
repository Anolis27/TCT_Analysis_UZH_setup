# config.py
import os

class Paths:
    BASE_DIR = os.path.expanduser("C:/Users/mathi/Documents/UZH/Data/V1_TW5/100V")
    DATAFILE = os.path.join(BASE_DIR, "parsed_from_waveforms.sqlite")
    DATAFILE2 = os.path.join(BASE_DIR, "measured_data.sqlite")
    POSITIONS = os.path.join(BASE_DIR, "positions.pickle")

class Colors:
    CB_CYCLE = [
        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
        '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'
    ]

    BASIC = [
        "Red","Green","Blue","Yellow","Cyan","Magenta","Orange","Purple",
        "Pink","Brown","Black","White","Gray","DarkRed","DarkGreen",
        "DarkBlue","LightGray","LightGreen","LightBlue","LightCoral"
    ]

class Filters:
    AMPLITUDE_THRESHOLD = -0.07  # V
    TIME_DIFF_MIN = 98
    TIME_DIFF_MAX = 99.5
    PEAK_TIME_MIN = 4.5
    PEAK_TIME_MAX = 6.5
