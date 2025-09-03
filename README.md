# water-dysh

This repository contains python scripts to generate waterfall plots of GBT spectral line data using the python package `dysh`. I wrote this program with the intent of using it to look at the dynamic spectrum of the data to look at the presence of RFI over the duration of the observation. It has the ability to modify the data in several different ways before plotting:

- `raw_data` -- this plots the data exactly as it is in the `sdfits` file.

- `median_subtract` -- this calculates the median spectrum over the course of a scan and subtracts it from each subintegration

- `calibrate_Ta` -- this follows the standard GBT calibration method using the noise diode temperature ($T_{cal}$) and system temperature ($T_{sys}$) to convert the data from counts to Jy. This method of calibration also has the option to flag and remove RFI that is broadband (> 5 MHz wide) and on short timescales (~few subintegrations). 



## RFI flagging/removal

This method exploits the high time variability of some RFI signals to allow us to substitute data from other integrations to remove RFI from the calOFF integrations before calibrating. This allows us to calibrate the strength of RFI signals "through" themselves. If unmodified, these signals would create artifacts in the calibrated data. 

In order to utilize this RFI flagging/removal, you must create a `band_allocations.py` file 

    touch band_allocations.py

and fill it with a dictionary containing the known RFI bands (in GHz) you would like to check. An example of the contents and dictionary format is below: 

    band_allocation_ghz_dict = {"none":{},
                                'example_band':{'first_channel':[1.23, 1.25],
                                                'second_channel':[1.42, 1.43]
                                               }
                               }

The first keys are to label sets of bands that you want grouped together. The second set of keys (within the first dictionary) correspond to the names and boundaries of the channels you would like to specify. This allows you to add as many distinct groups of frequency bounds as you would like, and keep them distinct from each other.

## running this program

These python files are written as library files, and executing them as they are will not produce anything. They will need to be imported and called in an additional python file. An example python file is below: 

    # driver.py

    from GBT_waterfall import * 
    from dysh.fits.gbtfitsload import GBTFITSLoad, GBTOffline

    if __name__ == "__main__":
        outdir = "path/to/save/"
        session_IDs = ["project_code_1", "project_code_2"]

        for session_ID in session_IDs:
            # sdf = GBTFITSLoad(path_to_file) # for data not in /home/sdfits
            sdf = GBTOffline(session_ID) # for data in /home/sdfits
            
            GBT_waterfall(sdf, 
                          session_ID, 
                          fmin_GHz=1.0, 
                          fmax_GHz=2.0, 
                          band_allocation="example_band", 
                          channels=["first_channel", "second_channel"],
                          outdir=outdir, 
                          cal_type="calibrate_Ta", 
                          replace_RFI=True, 
                          n_SD=1, 
                          debug=False)

