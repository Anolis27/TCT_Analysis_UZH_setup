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
    Paths.resolve() # If BASEDIR is not found, it is resolved automatically
    start_time = time.time()

    ####### DEBUGGING #######
    # active_channels = determine_active_channels(Paths.DATAFILE)
    # print(f"Active channels: {active_channels}")                   
    #plot_pad_positions(Paths.DATAFILE, Paths.POSITIONS)           
    #plot_pad_position_everything()                           
    #plot_sensor_strip_positions(Paths.DATAFILE, Paths.POSITIONS)
    #########################

    ####### FILTER CONFIG ######                     
    #plot_amplitude_against_t_peak(Paths.DATAFILE, time_variable = "t_90 (s)") 
    #plot_amplitude(Paths.DATAFILE)
    #plot_time_difference_t50(Paths.DATAFILE)
    ############################

    ####### AMPLITUDE ANALYSIS #######
    #plot_noise(Paths.DATAFILE)  
    #plot_amplitude_everything()                                 
    #plot_saved_results("Amplitude")                               
    #plot_2d_amplitude(Paths.DATAFILE, Paths.POSITIONS)             
    #plot_amplitude_along_y_axis(Paths.DATAFILE, Paths.POSITIONS)  
    ##################################

    ####### CHARGE COLLECTION ANALYSIS #######
    #plot_collected_charge_everything()                          
    #plot_saved_results("Collected_charge")                       
    #plot_2d_charge(Paths.DATAFILE, Paths.POSITIONS)               
    #plot_collected_charge(Paths.DATAFILE)                         
    ##########################################

    #plot_2d_amplitude_charge_everything()

    ####### TIMING ANALYSIS #######
    #plot_time_resolution_everything()                       
    #plot_saved_results("Timing")                               
    #plot_2d_timing(Paths.DATAFILE, Paths.POSITIONS)                
    #plot_2d_timing_everything() 
    ###############################

    ####### INTERPAD ANALYSIS #######
    #plot_interpad_distance_against_bias_voltage_v2()            
    #plot_saved_results("Interpad_distance")                      
    #plot_time_resolution_interpad_region_everything()               
    #plot_saved_results("Timing_interpad_region")                     
    #################################

    ######## save data processing #######
    #merge_saved_results("saved_results/Collected_charge/W9_V2_TW1_lowV.pkl", "saved_results/Collected_charge/W9_V2_TW1_highV.pkl", "saved_results/Collected_charge/W9_V2_TW1_merged.pkl" ) # NOT WORKING


    time_taken = round(time.time() - start_time)

    minutes = time_taken // 60
    seconds = time_taken - 60*minutes 
    print(f"--- Runtime: {minutes} minutes {seconds} seconds ---")

if __name__ == "__main__":
    main()
    