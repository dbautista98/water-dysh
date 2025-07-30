"""
This code uses dysh to read and generate a waterfall plot of GBT data. 
These functions are meant to be imported to another file where the list of 
data to be read in can be manually set, as well as any input/output directories
and plot windows. 
"""

try:
    from band_allocations import band_allocation_ghz_dict
except:
    band_allocation_ghz_dict = {"none":{}}
import calibration
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation,SkyCoord
from astropy import units as u
import astropy.units as u
from datetime import datetime
import numpy as np
import os

band_options = list(band_allocation_ghz_dict.keys())

def which_band_allocation():
    """
    prints the list of available band allocations to be used
    """
    print("the available band allocation keys are:")
    for band in band_options:
        print("\t", band)

# a dictionary to map from sdfits metadata to polarization
polnum_to_pol = {
                1: 'I',
                2: 'Q',
                3: 'U',
                4: 'V',
                -1: 'RR',
                -2: 'LL',
                -3: 'RL',
                -4: 'LR',
                -5: 'XX',
                -6: 'YY',
                -7: 'XY',
                -8: 'YX'}

calibration_type = {"raw_data":calibration.raw_data,
                   "median_subtract":calibration.median_subtract,
                   "calibrate_Ta":calibration.calibrate_Ta,
                   }

calibration_options = list(calibration_type.keys())

def which_calibration():
    """
    prints the list of available calibration types to be used
    """
    print("the available calibration options are:")
    for band in calibration_options:
        print("\t", band)

def plot_band_allocations(ax, freq, band_allocation="none", show_label=True):
    """
    Overlays the band allocations and adds a label at the top of the figure

    Arguments:
    ----------------
    ax : matplotlib.axes._axes.Axes
        the specific subplot object that will be annotated 
    freq : numpy.ndarray
        the frequency axis of the given data
    band_allocation : str
        the key to identify which set of band allocations to 
        overlay on the plot. Running which_band_allocation() will 
        show the available options 
    show_label : bool
        a flag controlling whether or not to include a text label at the 
        top of the figure 
    """
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
    ylim_chan_label = ylim[1] + 0.01*(ylim[1] - ylim[0])

    for i,nc in enumerate(list(band_allocation_ghz_dict[band_allocation].keys())):

        sat_dl_nu_ghz0 = band_allocation_ghz_dict[band_allocation][nc][0]
        if (freq.min()) <= sat_dl_nu_ghz0 <= (freq.max()):

            ax.vlines(sat_dl_nu_ghz0,ylim[0],ylim[1],ls='--',color='k',alpha=0.5)

        sat_dl_nu_ghz1 = band_allocation_ghz_dict[band_allocation][nc][1] 

        if (freq.min()) <= sat_dl_nu_ghz1 <= (freq.max()):
            ax.vlines(sat_dl_nu_ghz1,ylim[0],ylim[1],ls='--',color='k',alpha=0.5)

        band_width = np.abs(sat_dl_nu_ghz1 - sat_dl_nu_ghz0)
        text_x = sat_dl_nu_ghz0 + 0.1*band_width

        if (freq.min()) <= text_x <= freq.max()-0.5*band_width and show_label:
            ax.text(text_x,ylim_chan_label,nc,fontsize=10)

    return

def check_dir(outpath):
    """
    checks for the existence of a directory and if it 
    does not exist, it will generate the directory

    Arguments:
    ----------------
    outpath : str
        the filepath to the directory whose existence is to be verified
    """
    if os.path.exists(outpath):
        pass
    else:
        os.mkdir(outpath)

def get_metadata(tpsb, i=0):
    """
    retrieves the pointing and timestamp metadata for a given scan or set of scans

    Arguments:
    ----------------
    tpsb : dysh.spectra.scan.ScanBlock
        total power scan block that contains all the relevant data for an observation
    i : int
        index of the tpsb object. Defaults to zero. This is used to index through individual scans

    Returns:
    ---------------- 
    az_values : list
        A list containing the azimuth metadata for each integration of a scan
    el_values : list
        A list containing the elevation metadata for each integration of a scan
    timestamps : list 
        A list containing the timestamp metadata for each integration of a scan
    """
    timestamps = []
    az_values = []
    el_values = []

    all_medadata = tpsb[i].meta

    # checking the coordinate system 
    coord1 = all_medadata[0]['CTYPE2']
    coord2 = all_medadata[0]['CTYPE3']

    # ensure that the code can handle the coordinate system
    assert coord1 in ["AZ", "RA"] and coord2 in ["EL", "DEC"], "Congratulations you found a coordinate system that Dan didn't account for! Please tell him :)"

    for subint_num in range(len(all_medadata)):
        this_subint_metadata = all_medadata[subint_num]
        az_values.append(this_subint_metadata["CRVAL2"])
        el_values.append(this_subint_metadata["CRVAL3"])
        timestamps.append(this_subint_metadata["DATE-OBS"])

    # convert RA/DEC to AZ/EL 
    if coord1 == "RA" and coord2 == "DEC":
        GBT = EarthLocation.of_site('Green Bank Telescope')
        coords = SkyCoord(ra=np.array(az_values)*u.deg, dec=np.array(el_values)*u.deg, obstime=timestamps, frame="icrs", location=GBT)
        az_values = list(coords.altaz.az.deg)
        el_values = list(coords.altaz.alt.deg)

    return az_values, el_values, timestamps

def frequency_cut(freq, ts_no_spur, fmin_GHz=0, fmax_GHz=1e99):
    """
    Option to apply a frequency mask to the data. This function is used as an intermediate
    helper function when plotting the data.

    Arguments:
    ----------------
    freq : numpy.ndarray
        the frequency axis of the given data
    ts_no_spur_median_subtracted : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
    fmin_GHz : float
        minimum frequency that will be plotted. The default is 0 GHz. 
    fmax_GHz : float
        maximum frequency that will be plotted. The default is 1e99 GHz. 

    Returns:
    ---------------- 
    freq : numpy.ndarray
        The masked frequency axis of the given data
    ts_no_spur_median_subtracted : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
        nchan is now the number of frequency channels that fall within the 
        specified boundaries. 
    """
    ts_no_spur = np.ma.copy(ts_no_spur)
    freq = np.ma.copy(freq)
    freq_mask = np.where((freq >= fmin_GHz) & (freq <= fmax_GHz))
    ts_no_spur = ts_no_spur[::, freq_mask][::, 0, ::]
    freq = freq[freq_mask]
    return freq, ts_no_spur

def GBT_waterfall(sdf, session_ID, fmin_GHz=0, fmax_GHz=1e99, band_allocation="none", channels=[], cal_type="median_subtract", scale="linear", outdir="./", plot_type="png", replace_RFI=False, n_SD=1, debug=False):
    """
    Generates a waterfall plot of the given data. The data can be restricted in frequency 
    with the fmin_GHz, fmax_GHz parameters. There are also the option to specify the band 
    allocation to plot as an overlay. 

    Arguments:
    ----------------
    sdf : dysh.fits.gbtfitsload.GBTOffline OR dysh.fits.gbtfitsload.GBTFITSLoad
        a Dysh object holding the loaded data of a GBT observation
    session_ID : str
        session ID for an observation. This is used to identify the observation
        as well as generate the directory structure for saving the plots 
    fmin_GHz : float
        minimum frequency that will be plotted. The default is 0 GHz. 
    fmax_GHz : float
        maximum frequency that will be plotted. The default is 1e99 GHz. 
    band_allocation : str
        the key to identify which set of band allocations to 
        overlay on the plot. Running which_band_allocation() will 
        show the available options 
    cal_type : str
        label to identify what operations were done to scale the data. 
        changing this label allows the user to change which calibration
        type is used on the data. Running which_calibration() will show 
        the available options
    scale : str
        the option to change the scaling of the data. It can either be 
        linear or log scaled. The default is linear
    outdir : str
        filepath to where the generated plots will be saved
    plot_type : str
        the ablity to specify whether to save the plot as a pdf or png. 
        The default is to save as a png
    debug : bool
        A flag to generate log csv and plots. These outputs will be saved to the 
        current working directory
    """
    assert cal_type in calibration_options, "the available calibration options are %s"%calibration_options
    assert plot_type in ["png", "pdf"], "the plot_type options are: ['png', 'pdf']"
    assert scale in ["linear", "log"], "the scale options are: ['linear', 'log']"
    assert band_allocation in band_options, "the available band_allocation options are %s"%band_options
    assert fmin_GHz < fmax_GHz, "warning: fmin is greater than fmax"

    calibration_kwargs = {"replace_RFI":replace_RFI, 
                          "n_SD":n_SD,
                          "band_allocation":band_allocation,
                          "channels":channels,
                          "debug":debug}

    # ensure that the output directory structure exists
    check_dir(outdir)
    outdir = f"{outdir}/{session_ID}/"
    check_dir(outdir)

    summary_df = sdf.summary()

    # switch between uniform or granular plotting logic
    if_difference = np.diff(summary_df["# IF"].values)
    pl_difference = np.diff(summary_df["# POL"].values)
    fd_difference = np.diff(summary_df["# FEED"].values)
    all_diffs = np.hstack([if_difference, pl_difference, fd_difference])

    # check if all scans were performed with the same setup
    if np.any(all_diffs != 0):
        # individually handle each scan
        single_scan_waterfall(sdf, fmin_GHz=fmin_GHz, fmax_GHz=fmax_GHz, cal_type=cal_type, scale=scale, outdir=outdir, plot_type=plot_type, **calibration_kwargs)
    else:
        # all scans were performed with the same setup
        uniform_waterfalls(sdf, fmin_GHz=fmin_GHz, fmax_GHz=fmax_GHz, cal_type=cal_type, scale=scale, outdir=outdir, plot_type=plot_type, **calibration_kwargs)

def uniform_waterfalls(sdf, fmin_GHz=0, fmax_GHz=1e99, cal_type="median_subtract", scale="linear", outdir="./", plot_type="png", **kwargs):
    """
    A helper function called from GBT_waterfall to handle the loop logic for observations in which 
    all scans were performed with the same number of polarizatoins, IF windows, and feeds. 
    For a detailed description of the arguments, see the documentation for GBT_waterfalls
    """
    summary_df = sdf.summary()
    scans = summary_df["SCAN"].values
    scans.sort()

    plnums = np.arange(summary_df["# POL"].values[0])
    ifnums = np.arange(summary_df["# IF"].values[0])
    fdnums = np.arange(summary_df["# FEED"].values[0])

    for fdnum in fdnums:
        for plnum in plnums:
            for ifnum in ifnums:
                tpsb = sdf.gettp(scan=scans,ifnum=ifnum,plnum=plnum,fdnum=fdnum)
                for i in range(len(scans)):
                    # calibrate the data here
                    freq, ts_no_spur, unit = calibration_type[cal_type](sdf, tpsb, i, **kwargs)
                    az_values, el_values, timestamps = get_metadata(tpsb, i=i)

                    # pipe relevant metadata into kwarg dictionary
                    kwargs["filename"]= sdf.filename
                    kwargs["scan"] = tpsb[i].scan
                    kwargs["pl"] = polnum_to_pol[tpsb[i].meta[0]["CRVAL4"]]
                    kwargs["ifnum"] = tpsb[i].ifnum
                    kwargs["fdnum"] = tpsb[i].fdnum
                    kwargs["df_kHz"] = np.round(np.abs(tpsb[i].meta[0]["CDELT1"])/1000, 3)
                    kwargs["rcvr"] = tpsb[i].meta[0]["FRONTEND"]
                    kwargs["time_delta"] = datetime.strptime(timestamps[1], "%Y-%m-%dT%H:%M:%S.%f") - datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%f")
                    kwargs["dt"] = np.round(kwargs["time_delta"].total_seconds(), 3)
                    kwargs["az_values"] = az_values
                    kwargs["el_values"] = el_values
                    kwargs["timestamps"] = timestamps
                    kwargs["unit"] = unit

                    # generate the waterfall 
                    plot_waterfall(freq, ts_no_spur, fmin_GHz=fmin_GHz, fmax_GHz=fmax_GHz, cal_type=cal_type, scale=scale, outdir=outdir, plot_type=plot_type, **kwargs)

def single_scan_waterfall(sdf, fmin_GHz=0, fmax_GHz=1e99, cal_type="median_subtract", scale="linear", outdir="./", plot_type="png", **kwargs):
    """
    A helper function called from GBT_waterfall to handle the loop logic for observations in which 
    there are scans with differing numbers of polarizations or IF windows or feeds
    For a detailed description of the arguments, see the documentation for GBT_waterfalls
    """
    summary_df = sdf.summary()
    scans = summary_df["SCAN"].values
    scans.sort()

    # for each scan, pull the number of feeds, polarizations, and IF windows
    # there is no reason to assume that they will be the same for all scans in a session
    for i, this_scan in enumerate(scans):
        fdnums = np.arange(summary_df[summary_df["SCAN"] == this_scan]["# FEED"].iloc[0])
        for fdnum in fdnums:
            plnums = np.arange(summary_df[summary_df["SCAN"] == this_scan]["# POL"].iloc[0])
            for plnum in plnums:
                ifnums = np.arange(summary_df[summary_df["SCAN"] == this_scan]["# IF"].iloc[0])
                for ifnum in ifnums:
                    tpsb = sdf.gettp(scan=[this_scan],ifnum=ifnum,plnum=plnum,fdnum=fdnum) 
                    # calibrate the data here
                    freq, ts_no_spur, unit = calibration_type[cal_type](sdf, tpsb, i, **kwargs)
                    az_values, el_values, timestamps = get_metadata(tpsb, i=i)

                    # pipe relevant metadata into kwarg dictionary
                    kwargs["filename"]= sdf.filename
                    kwargs["scan"] = tpsb[i].scan
                    kwargs["pl"] = polnum_to_pol[tpsb[i].meta[0]["CRVAL4"]]
                    kwargs["ifnum"] = tpsb[i].ifnum
                    kwargs["fdnum"] = tpsb[i].fdnum
                    kwargs["df_kHz"] = np.round(np.abs(tpsb[i].meta[0]["CDELT1"])/1000, 3)
                    kwargs["rcvr"] = tpsb[i].meta[0]["FRONTEND"]
                    kwargs["time_delta"] = datetime.strptime(timestamps[1], "%Y-%m-%dT%H:%M:%S.%f") - datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%f")
                    kwargs["dt"] = np.round(kwargs["time_delta"].total_seconds(), 3)
                    kwargs["az_values"] = az_values
                    kwargs["el_values"] = el_values
                    kwargs["timestamps"] = timestamps
                    kwargs["unit"] = unit

                    # generate waterfall 
                    plot_waterfall(freq, ts_no_spur, fmin_GHz=fmin_GHz, fmax_GHz=fmax_GHz, cal_type=cal_type, scale=scale, outdir=outdir, plot_type=plot_type, **kwargs)

def plot_waterfall(freq, timeseries_grid, fmin_GHz=0, fmax_GHz=1e99, cal_type="median_subtract", scale="linear", outdir="./", plot_type="png", **kwargs):
    """
    A helper function that generates and annotates a waterfall plot from GBT sdfits data. 
    This function is called from within other functions but can be called on its own. 
    
    Arguments:
    ----------------
    freq : numpy.ndarray
        the frequency axis of the given data to be plotted. It should have shape (nchan,)
    timeseries_grid : numpy.ma.MaskedArray
        The time series data of the scan to be plotted. It has shape (n_int, nchan)
    fmin_GHz : float
        minimum frequency that will be plotted. The default is 0 GHz. 
    fmax_GHz : float
        maximum frequency that will be plotted. The default is 1e99 GHz. 
    cal_type : str
        label to identify what operations were done to scale the data. 
        changing this label allows the user to change which calibration
        type is used on the data. Running which_calibration() will show 
        the available options
    scale : str
        the option to change the scaling of the data. It can either be 
        linear or log scaled. The default is linear
    outdir : str
        filepath to where the generated plots will be saved
    plot_type : str
        the ablity to specify whether to save the plot as a pdf or png. 
        The default is to save as a png
    """
    assert len(freq) == timeseries_grid.shape[1], f"input data dimensions do not match: {len(freq)}, {timeseries_grid.shape[1]}"

    # kwarg retrieval
    band_allocation = kwargs.get("band_allocation", "none")
    replace_RFI = kwargs.get("replace_RFI", False)
    n_SD = kwargs.get("n_SD", 1)

    # previously dysh object metadata, now from kwargs
    filename = kwargs.get("filename", "unknown_project") 
    scan = kwargs.get("scan", -999) 
    pl = kwargs.get("pl", -999)
    ifn = kwargs.get("ifnum", -999)
    fd = kwargs.get("fdnum", -999)
    df_kHz = kwargs.get("df_kHz", -999)
    rcvr = kwargs.get("rcvr", "unknown")
    time_delta = kwargs.get("time_delta", -999) 
    dt = kwargs.get("dt", -999)
    unit = kwargs.get("unit", "unknown")
    az_values = kwargs.get("az_values", -999*np.ones(len(timeseries_grid)))
    el_values = kwargs.get("el_values", -999*np.ones(len(timeseries_grid)))
    timestamps = kwargs.get("timestamps", len(timeseries_grid)*["-999"])

    if replace_RFI:
        rfi_flag_filename = f"RFI_flag_n-SD_{n_SD}_"
    else:
        rfi_flag_filename = ""

    if np.any( freq < fmax_GHz) and np.any( freq > fmin_GHz):
        print(f"plotting: {os.path.basename(filename).replace('.raw.vegas', '')} scan = {scan} ifnum = {ifn} plnum = {pl} fdnum = {fd}")

        freq, timeseries_grid = frequency_cut(freq, timeseries_grid, fmin_GHz=fmin_GHz, fmax_GHz=fmax_GHz)
        extent = [freq[0], freq[-1], 0, len(timeseries_grid)]

        flux = np.ma.mean(timeseries_grid, axis=0)
        max_val = np.nanmax(timeseries_grid)
        y = np.arange(len(timeseries_grid))
        data_sd = np.nanstd(timeseries_grid)
        data_mean = np.ma.median(timeseries_grid)
        vmax = data_mean + 2*data_sd
        vmin = data_mean - 2*data_sd

        plt.close("all")
        time_series = np.nanmean(timeseries_grid, axis=1)
        fig = plt.figure(figsize=(10,12), layout="constrained")
        gs = fig.add_gridspec(2,2, hspace=0.02, wspace=0.03, width_ratios=[3,1], height_ratios=[1,3])
        (ax1, ax2), (ax3, ax4) = gs.subplots(sharex="col", sharey="row")

        #ax1
        ax1.set_title(f"{filename}\nrcvr: {rcvr}\npeak power: {np.round(max_val, 2)} {unit}\nScan {scan}\npolarization {pl}\nifnum {ifn}\nfdnum {fd}\ndt = {dt} s\ndf = {df_kHz} kHz\n")
        ax1.plot(freq, flux, color="black", linewidth=1)
        ax1.set_yscale(scale)
        ax1.set_ylim(np.nanmin(flux) - 0.05 * (np.nanmax(flux) - np.nanmin(flux)), np.nanmax(flux) + 0.25*(np.nanmax(flux) - np.nanmin(flux)))
        ax1.set_ylabel(f"average power\n[{unit}]")
        plot_band_allocations(ax1, freq, band_allocation=band_allocation)

        #ax2
        ax2.set_visible(not ax2)

        #ax3
        wf = ax3.imshow(timeseries_grid, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, origin="lower")
        ax3.set_xlabel("Frequency [GHz]")
        ax3.set_ylabel("timestamp [UTC]\npointing (AZ, EL)")
        fig.colorbar(wf, ax=ax3, label=f'power [{unit}]', orientation="horizontal")
        plot_band_allocations(ax3, freq, band_allocation=band_allocation, show_label=False)

        pointing_coords = []
        for j in range(len(az_values)):
            pointing_coords.append(f"({np.round(az_values[j], 2)}, {np.round(el_values[j], 2)})")

        all_labels = []
        for i in range(len(timestamps)):
            all_labels.append(timestamps[i] + "\n" + pointing_coords[i])

        # update y-tick labels 
        if len(ax3.get_yticks()) > len(all_labels):
            ax3.set_yticks(np.arange(len(all_labels)) + 1)
            ax3.set_yticklabels(np.arange(len(all_labels)) + 1)

            integration_indices = []
            for index in range(len(ax3.get_yticklabels())):
                integration_indices.append(int(ax3.get_yticklabels()[index].get_text()))

            new_labels = []
            new_labels.append(all_labels[0])
            for index in integration_indices[1:]:
                new_labels.append(all_labels[index-1])

            yticks = ax3.get_yticks()
            ax3.set_yticks(yticks)
            ax3.set_yticklabels(new_labels)
        else:
            integration_indices = np.linspace(0, len(timeseries_grid), num=len(ax3.get_yticklabels()), dtype=int)
            new_labels = []
            new_labels.append(all_labels[0])
            for index in integration_indices[1:]:
                new_labels.append(all_labels[index-1])

            ax3.set_yticks(integration_indices)
            ax3.set_yticklabels(new_labels)

        #ax4
        ax4.plot(time_series, y + 0.5, color="black", linewidth=1)
        ax4.set_xscale(scale)
        ax4.set_xlabel(f"\naverage power per\nfrequency channel\n[{unit}]")

        ax1.set_xlim(np.min(freq), np.max(freq))
        ax3.set_xlim(np.min(freq), np.max(freq))
        plt.savefig(f"{outdir}/{os.path.basename(filename)}_waterfall_ifnum_{ifn}_scan_{scan}_plnum_{pl}_fdnum_{fd}_caltype_{cal_type}_{rfi_flag_filename}metadata.{plot_type}", bbox_inches="tight", transparent=False)
        plt.close("all")
