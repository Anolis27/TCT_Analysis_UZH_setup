# main SCRIPT
from config import Paths, Colors, Filters
from amplitude import *
from timing import *
from charge_collection import *
from data_manager import *
from interpad import *
import time



def main():
    print(f"Script started")
    start_time = time.time()

    ####### DEBUGGING #######
    # active_channels = determine_active_channels(Paths.DATAFILE)
    # print(f"Active channels: {active_channels}")
    #########################

    ####### AMPLITUDE ANALYSIS #######
    # plot_amplitude(Paths.DATAFILE, Paths.POSITIONS)                # NOT WORKING
    # plot_amplitude_everything()                                      # WORKING

    ##################################

    ####### CHARGE COLLECTION ANALYSIS #######
    # plot_collected_charge_everything()                              # WORKING

    ##########################################

    ####### TIMING ANALYSIS #######
    # plot_time_resolution_everything()                             # WORKING
    ###############################

    ####### INTERPAD ANALYSIS #######
    plot_interpad_distance_against_bias_voltage_v2()               # NOT WORKING
    #################################


    time_taken = round(time.time() - start_time)

    minutes = time_taken // 60
    seconds = time_taken - 60*minutes 
    print(f"--- Runtime: {minutes} minutes {seconds} seconds ---")

if __name__ == "__main__":
    main()
    