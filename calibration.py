import numpy as np
import astropy.units as u


def raw_data(sdf, tpsb, i=0):
    """
    returns the raw counts data for a given scan (indexed by `i`)

    Arguments:
    ----------------
    sdf : dysh.fits.gbtfitsload.GBTOffline
        a Dysh object holding the loaded data of a GBT observation
    tpsb : dysh.spectra.scan.ScanBlock
        total power scan block that contains all the relevant data for an observation
    i : int
        index of the tpsb object. Defaults to zero. This is used to index through individual scans

    Returns:
    ----------------
    flux : numpy.ma.MaskedArray
        masked array containing the average spectrum of the given data
    freq : numpy.ndarray
        the frequency axis of the given data
    ts_no_spur : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan)
    """
    timeseries = tpsb[i]._calibrated # variables that start with an underscore will be replaced in fututre versions of dysh
    average_spect = tpsb[i].timeaverage()
    flux = np.ma.masked_where(average_spect.mask, average_spect.data)
    freq = average_spect.spectral_axis.to(u.GHz).value
    ts_no_spur = np.ma.masked_where(timeseries.mask, timeseries.data)
    return flux, freq, ts_no_spur

def median_subtract(sdf, tpsb, i=0):
    """
    returns the median subtracted data for a given scan (indexed by `i`). 
    This data has units of "counts". The median spectrum is calculated for 
    each frequency channel independently to reduce time-varying RFI artifacts
    showing up in the final data. Note that this method is not immune to very
    dense RFI regions, where there may not be an integration that is free of RFI.

    Arguments:
    ----------------
    sdf : dysh.fits.gbtfitsload.GBTOffline
        a Dysh object holding the loaded data of a GBT observation
    tpsb : dysh.spectra.scan.ScanBlock
        total power scan block that contains all the relevant data for an observation
    i : int
        index of the tpsb object. Defaults to zero. This is used to index through individual scans

    Returns:
    ----------------
    flux : numpy.ma.MaskedArray
        masked array containing the average spectrum of the given data. This is the average 
        raw data minus the median spectrum 
    freq : numpy.ndarray
        the frequency axis of the given data
    ts_no_spur_median_subtracted : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
        This data grid has had the median spectrum subtracted from each integration
    """
    timeseries = tpsb[i]._calibrated # variables that start with an underscore will be replaced in fututre versions of dysh
    average_spect = tpsb[i].timeaverage()
    freq = average_spect.spectral_axis.to(u.GHz).value
    ts_no_spur = np.ma.masked_where(timeseries.mask, timeseries.data)
    median_spectrum = np.ma.median(ts_no_spur, axis=0)
    flux = np.mean(ts_no_spur - median_spectrum, axis=0)
    ts_no_spur_median_subtracted = ts_no_spur - median_spectrum
    return flux, freq, ts_no_spur_median_subtracted

def get_spectrum_and_freq(sdf, i=0, calstate=True, scan=[1], ifnum=0, plnum=0, fdnum=0):
    """
    A helper function to return only the spectrum and frequency axis from the given
    scan metadata
    """
    tpsb = sdf.gettp(scan=scan,ifnum=ifnum,plnum=plnum,fdnum=fdnum,cal=calstate)
    timeseries = tpsb[i]._calibrated
    average_spect = tpsb[i].timeaverage()
    freq = average_spect.spectral_axis.to(u.GHz).value
    ts_no_spur = np.ma.masked_where(timeseries.mask, timeseries.data)
    tsys = np.array(list(tpsb[0].meta[i]["TSYS"] for i in range(len(tpsb[0].meta))))
    return freq, ts_no_spur, tsys

def replace_bad_integrations(ts_grid):
    """
    A function to identify integrations that have a strong presence of 
    broadband RFI (signals on the order of 5-250 MHz) and replace them 
    with data from neighboring integrations that do not have strong RFI
    """
    return ts_grid

# add docstring
def calibrate_scan(sdf, tpsb, i=0, scan=[1], fdnum=0, plnum=0, ifnum=0, replace_RFI=False):
    """
    This is the standard calibration method, following the process outlined  in 
    https://www.gb.nrao.edu/GBT/DA/gbtidl/gbtidl_calibration.pdf
    """
    # read in the data, keeping both a calon and caloff set
    freq,  cal_ts, tsys = get_spectrum_and_freq(sdf, calstate=True, scan=scan, ifnum=ifnum, plnum=plnum, fdnum=fdnum)
    freq, nocal_ts, tsys = get_spectrum_and_freq(sdf, calstate=False, scan=scan, ifnum=ifnum, plnum=plnum, fdnum=fdnum)
    assert cal_ts.shape == nocal_ts.shape, "data shapes do not match: %s vs %s" %(cal_ts.shape, nocal_ts.shape)
    # replace the bad integrations from the off data
    if replace_RFI:
        nocal_ts = replace_bad_integrations(nocal_ts)
    else:
        nocal_ts = nocal_ts

    # tsys
    print(f"tsys shape: {tsys.shape}")
    print(f"ts shape: {cal_ts.shape}")

    Ta = tsys[:, np.newaxis] * ((cal_ts - nocal_ts) / nocal_ts)
    flux = np.ma.mean(Ta, axis=0)

    return flux, freq, Ta
