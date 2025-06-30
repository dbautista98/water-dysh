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
    average_spect : dysh.spectra.spectrum.Spectrum
        The spectrum object that contains all the metadata relevant to the scan. 
        This is mainly used for extracting metadata for plotting
    """
    timeseries = tpsb[i]._calibrated # variables that start with an underscore will be replaced in fututre versions of dysh
    average_spect = tpsb[i].timeaverage()
    flux = np.ma.masked_where(average_spect.mask, average_spect.data)
    freq = average_spect.spectral_axis.to(u.GHz).value
    ts_no_spur = np.ma.masked_where(timeseries.mask, timeseries.data)
    return flux, freq, ts_no_spur, average_spect

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
    average_spect : dysh.spectra.spectrum.Spectrum
        The spectrum object that contains all the metadata relevant to the scan. 
        This is mainly used for extracting metadata for plotting
    """
    timeseries = tpsb[i]._calibrated # variables that start with an underscore will be replaced in fututre versions of dysh
    average_spect = tpsb[i].timeaverage()
    freq = average_spect.spectral_axis.to(u.GHz).value
    ts_no_spur = np.ma.masked_where(timeseries.mask, timeseries.data)
    median_spectrum = np.ma.median(ts_no_spur, axis=0)
    flux = np.mean(ts_no_spur - median_spectrum, axis=0)
    ts_no_spur_median_subtracted = ts_no_spur - median_spectrum
    return flux, freq, ts_no_spur_median_subtracted, average_spect
