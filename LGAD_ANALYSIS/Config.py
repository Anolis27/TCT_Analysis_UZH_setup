# config.py
import os

# Define global configuration parameters

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'] #color blind friendly colors
colors = ["Red","Green","Blue","Yellow","Cyan","Magenta","Orange","Purple","Pink","Brown","Black","White","Gray","DarkRed","DarkGreen","DarkBlue","LightGray","LightGreen","LightBlue","LightCoral"]

test_directory = os.path.expanduser("C:/Users/mathi/Documents/UZH/Data/V1_TW5/100V")
test_datafile  = os.path.abspath(f"{test_directory}/parsed_from_waveforms.sqlite")
test_datafile2 = os.path.abspath(f"{test_directory}/measured_data.sqlite")
test_positions = os.path.abspath(f"{test_directory}/positions.pickle")

######## FILTERS GLOBAL PARAMETERS ########
AMPLITUDE_THRESHOLD = -0.07  # V
TIME_DIFF_MIN = 98         # s
TIME_DIFF_MAX = 99.5       # s
PEAK_TIME_MIN = 4.5       # s
PEAK_TIME_MAX = 6.5       # s
###########################################