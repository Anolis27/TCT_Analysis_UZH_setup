# main SCRIPT
from config import Paths, Colors, Filters
from amplitude import *
from timing import *
from charge_collection import *
from interpad import *
import time



def main():
    print(f"Script started")
    start_time = time.time()

    ####### AMPLITUDE ANALYSIS #######
    plot_amplitude(Paths.DATAFILE, Paths.POSITIONS)

    ##################################

    ####### CHARGE COLLECTION ANALYSIS #######

    ##########################################

    ####### TIMING ANALYSIS #######

    ###############################

    ####### INTERPAD ANALYSIS #######

    #################################


    time_taken = round(time.time() - start_time)

    minutes = time_taken // 60
    seconds = time_taken - 60*minutes 
    print(f"--- Runtime: {minutes} minutes {seconds} seconds ---")