# main SCRIPT
import config as cfg
import data_manager as dm
import amplitude as amp
import charge_collection as cc
import interpad as ipd

import time

def main():
    print(f"Script started")
    start_time = time.time()



    time_taken = round(time.time() - start_time)

    minutes = time_taken // 60
    seconds = time_taken - 60*minutes 
    print(f"--- Runtime: {minutes} minutes {seconds} seconds ---")