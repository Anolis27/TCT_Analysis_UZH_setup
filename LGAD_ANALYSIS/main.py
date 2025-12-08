# main SCRIPT
from plot_saved_results import plot_saved_results
from config import Paths, Colors, Filters, Subplots, InterpadConfig
from amplitude import *
from timing import *
from charge_collection import *
from data_manager import *
from interpad import *
from plot_2D_maps import *
import time



def main():
    print(f"Script started")
    start_time = time.time()

    ####### DEBUGGING #######
    # active_channels = determine_active_channels(Paths.DATAFILE)
    # print(f"Active channels: {active_channels}")                     # WORKING
    #plot_pad_positions(Paths.DATAFILE, Paths.POSITIONS)              # WORKING
    #plot_pad_position_everything()                             # WORKING
    #plot_sensor_strip_positions(Paths.DATAFILE, Paths.POSITIONS)  # WORKING
    #########################

    ####### AMPLITUDE ANALYSIS #######
    #plot_amplitude_everything()                                      # WORKING
    #plot_saved_results("Amplitude")                                   # WORKING
    #plot_2d_amplitude(Paths.DATAFILE, Paths.POSITIONS)              # WORKING   but needs merge 1pdf
    ##################################

    ####### CHARGE COLLECTION ANALYSIS #######
    #plot_collected_charge_everything()                              # WORKING
    #plot_saved_results("Collected_charge")                            # WORKING
    #plot_2d_charge(Paths.DATAFILE, Paths.POSITIONS)                 # WORKING   but needs merge 1pdf
    ##########################################

    ####### TIMING ANALYSIS #######
    #plot_time_resolution_everything()                             # WORKING
    #plot_2d_timing(Paths.DATAFILE, Paths.POSITIONS)                # WORKING   but needs merge 1pdf
    #plot_saved_results("Timing")                                   # WORKING
    ###############################

    ####### INTERPAD ANALYSIS #######
    #plot_interpad_distance_against_bias_voltage_v2()               # WORKING
    #plot_saved_results("Interpad_distance")                        # WORKING
    #plot_time_resolution_interpad_region_everything()                # WORKING
    #plot_saved_results("Timing_interpad_region")                     # WORKING
    #################################


    time_taken = round(time.time() - start_time)

    minutes = time_taken // 60
    seconds = time_taken - 60*minutes 
    print(f"--- Runtime: {minutes} minutes {seconds} seconds ---")

if __name__ == "__main__":
    main()
    