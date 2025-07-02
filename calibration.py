import numpy as np
import astropy.units as u
try:
    from band_allocations import band_allocation_ghz_dict
except:
    band_allocation_ghz_dict = {"none":{}}


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
    A helper function to return the spectrum and frequency axis from the given
    scan metadata, as well as the Tsys for each integration

    Arguments:
    ----------------
    sdf : dysh.fits.gbtfitsload.GBTOffline OR dysh.fits.gbtfitsload.GBTFITSLoad
        a Dysh object holding the loaded data of a GBT observation
    i : int
        index of the tpsb object. Defaults to zero. This is used to index 
        through individual scans
    calstate : bool
        Boolean value telling whether the noise diode state is on (True) or off (False)
        Only data with the specified state will be returned 
    scan : list
        the list of scan numbers to return data for. Typically passed in from  
        the sdf summary
    ifnum : int
        The IF window to return data for. This can range from 0 - N, and can be found in
        in the sdf summary 
    plnum : int
        The polarization state to return data for. Typically this is {0,1}, but can go higer.
        This information can be found in the sdf summary
    fdnum : int
        The feed number to return data from. This will be 0 for all single beam receivers. 
        For multibeam receivers, this number can be higher. This information can be found 
        in the sdf summary. 

    Returns:
    ----------------
    freq : numpy.ndarray
        the frequency axis of the given data
    ts_no_spur_median_subtracted : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
        This data grid has had the median spectrum subtracted from each integration
    tcal : float
        The known contribution to the system temperature when the noise diode(s)
        are active
    tsys : numpy.ndarray
        The system temperature for each subintegration of the observation. The system
        temperature is a sum of all the temperature contributions through the system
    """
    tpsb = sdf.gettp(scan=scan,ifnum=ifnum,plnum=plnum,fdnum=fdnum,cal=calstate)
    timeseries = tpsb[i]._calibrated
    average_spect = tpsb[i].timeaverage()
    freq = average_spect.spectral_axis.to(u.GHz).value
    ts_no_spur = np.ma.masked_where(timeseries.mask, timeseries.data)
    tsys = np.array(list(tpsb[0].meta[i]["TSYS"] for i in range(len(tpsb[0].meta))))
    return freq, ts_no_spur, tsys

def replace_bad_integrations(freq, ts_grid, n_SD=1, band_allocation="none", channels=[]):
    """
    A function to identify integrations that have a strong presence of 
    broadband RFI (signals on the order of 5-250 MHz) and replace them 
    with data from neighboring integrations that do not have strong RFI

    The method for cleaning RFI will be sigma clipping the integrations 
    with strong RFI and replacing them with data from integrations 
    with less RFI. 

    This process can be iterated several times to clean out lower level
    signals on subsequent passes 

    Arguments:
    ----------------
    freq : numpy.ndarray
        the frequency axis of the given data
    ts_grid : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
    n_SD : float
        the number of standard deviations above the median spectrum averaged 
        total power to consider an integration to have RFI. Integrations exceeding
        this criteria will be flagged for replacement with data containing less RFI.
    band_allocation : str
        the key to identify which set of known band allocations to use for RFI flagging
    channels : list of str
        A list of channel names to loop over. These channel names coorrespond to a 
        list containing a pair of [lower, upper] frequency bounds in units of GHz

    Returns:
    ----------------
    ts_grid : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
        This array will contain data that has been cleaned of RFI in the flagged
        integrations. 
    """
    timeseries_grid = np.ma.copy(ts_grid)
    # identify indices of integrations that have strong RFI
    bad_indices = flag_RFI_channels(freq, timeseries_grid, n_SD, band_allocation=band_allocation, channels=channels)
    print(f"replacing {len(bad_indices)} integrations ({np.round(len(bad_indices)/len(timeseries_grid)*100, 2)}%) with cleaner data")

    for indx in bad_indices:
        # identify upper and lower indices of integrations that have
        #   less (ideally none) RFI 
        lower, upper = get_good_neighbor(indx, bad_indices, len(ts_grid))

        replacement_spectrum = select_replacement_spectrum(ts_grid, lower, upper)
        ts_grid[indx] = replacement_spectrum

    return ts_grid

def get_good_neighbor(bad_index, all_bad_indices, data_length):
    """
    Identify the indices of the closest neighboring integrations that are not flagged
    for strong RFI. The nearest (in time) non-flagged integrations preceeding and following
    the flagged integration will be selected. If there is no clean integration preceeding the 
    flagged integration, the function will search for another clean spectrum after the chosen
    following scan. Likewise, an additional preceeding scan will be chosen if no non-flagged
    integration can be found following the flagged integration

    Arguments:
    ----------------
    bad_index : int
        The index of a subintegration that has been flagged as containing
        excessive amounts of RFI
    all_bad_indices : numpy.ndarray
        A list of indices corresponding to the flagged subintegrations
    data_length : int
        the total number of subintegrations in the data being cleaned

    Returns:
    ----------------
    lower_index : int
        The index of the closest (in time) integration that does not contain 
        an excessive amount of RFI, which preceeds the flagged integration
    upper_index : int
        The index of the closest (in time) integration that does not contain 
        an excessive amount of RFI, which follows the flagged integration
    """
    lower_index = bad_index
    upper_index = bad_index

    while lower_index in all_bad_indices and lower_index >= 0:
        lower_index -= 1

    while upper_index in all_bad_indices and upper_index < data_length - 1:
        upper_index += 1

    # is this block still needed if I have the following code? 
    if lower_index < 0 and upper_index >= data_length:
        print(f"lower index: {lower_index}, upper index: {upper_index}, number of integrations: {data_length}")
        raise Exception("data out of range. Tell Dan he needs to fix this.")

    elif lower_index < 0:
        lower_index = upper_index + 1
        while lower_index in all_bad_indices and lower_index < data_length - 1:
            lower_index += 1
    elif upper_index >= data_length:
        upper_index = lower_index - 1
        while upper_index in all_bad_indices and upper_index >= 0:
            upper_index -= 1

    return lower_index, upper_index

def channel_timeseries(target_freq_GHz, freq_axis, timeseries_grid):
    """
    Selects all time steps at the specified frequency channel to provide the 
    intensity as a function of time. 

    Arguments:
    ----------------
    target_freq_GHz : float
        Target frequency (in units of GHz) to extract time-series information for. 
    freq_axis : numpy.ndarray
        the frequency axis of the given data, in units of GHz
    timeseries_grid : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 

    Returns:
    ----------------
    timeseries : numpy.ma.MaskedArray
        The intensity of the specified frequency channel over the course
        of the scan. It will have shape (n_int,)
    """
    assert timeseries_grid.shape[1] == len(freq_axis), f"Data shapes do not match: grid channels: {timeseries_grid.shape[1]}, frequency axis channels: {len(freq_axis)}"
    assert np.min(freq_axis) <= target_freq_GHz <= np.max(freq_axis), "target frequency not in range"

    target_index = np.argmin(np.abs(freq_axis - target_freq_GHz))
    return timeseries_grid[::, target_index]

def chunk_timeseries(low_f, high_f, freq_axis, timeseries_grid):
    """
    Selects the specified frequency range and returns the average intensity as a function
    of time. 

    Arguments:
    ----------------
    low_f : float
        The lower frequency bound (in units of GHz) to check for RFI
    high_f : float
        The upper frequency bound (in units of GHz) to check for RFI
    freq_axis : numpy.ndarray
        the frequency axis of the given data, in units of GHz
    timeseries_grid : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
    
    Returns:
    ----------------
    timeseries : numpy.ma.MaskedArray
        The average intensity over the specified frequency range for the course
        of the scan. It will have shape (n_int,)
    """
    assert timeseries_grid.shape[1] == len(freq_axis), f"Data shapes do not match: grid channels: {timeseries_grid.shape[1]}, frequency axis channels: {len(freq_axis)}"
    assert np.min(freq_axis) <= low_f and  np.max(freq_axis) >= high_f, "target frequency range exceeds data range"

    mask = np.where((low_f <= freq_axis) & (freq_axis <= high_f))
    selected_data = timeseries_grid[::, mask][::, 0, ::]
    return np.ma.mean(selected_data, axis=1)

def find_RFI_integrations(time_series, n_SD=1):
    """
    Takes a time-series of data and returns the indices that are at least `n_SD` above or
    below the median value over the range. 

    Arguments:
    ----------------
    timeseries : numpy.ma.MaskedArray
        The average intensity over a specified frequency range for the course
        of the scan. It will have shape (n_int,)
    n_SD : float
        the number of standard deviations above the median spectrum averaged 
        total power to consider an integration to have RFI. Integrations exceeding
        this criteria will be flagged for replacement with data containing less RFI.

    Returns:
    ----------------
    bad_indices : numpy.ndarray
        A list of indices corresponding to the flagged subintegrations
    """
    # Identify which integrations are affected by strong RFI 
    median_power = np.median(time_series)
    sd = np.sqrt(np.sum( (time_series - median_power)**2 ) / len(time_series))
    bad_indices = np.where((median_power - n_SD*sd >= time_series) | (time_series >= median_power + n_SD*sd))
    return bad_indices[0]

def select_replacement_spectrum(timeseries_grid, good_lower, good_upper):
    """
    This function will take the indices of the neighboring good data
    and combine them to create a spectrum with less RFI that can be 
    used to calibrate the original CALON/CALOFF spectrum

    Arguments:
    ----------------
    timeseries_grid : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
    good_lower : int
        The index of the closest (in time) integration that does not contain 
        an excessive amount of RFI, which preceeds the flagged integration
    good_upper : int
        The index of the closest (in time) integration that does not contain 
        an excessive amount of RFI, which follows the flagged integration

    Returns:
    ----------------
    best_replacement : numpy.ma.MaskedArray
        A spectrum synthesized from the spectra indexed by the provided lower and upper 
        indices. It will have shape (nchan,)
    """
    best_replacement = np.ma.min([timeseries_grid[good_lower], timeseries_grid[good_upper]], axis=0)

    return best_replacement

def flag_RFI_channels(freq, timeseries_grid, sd_threshold, band_allocation="none", channels=[]):
    """
    Takes a set of known frequency allocations and a time-series of spectra and
    checks each frequency range for the presence of strong RFI. This list of flagged
    integrations is combined across all channels and a list of all unique flagged indices
    is returned

    Arguments:
    ----------------
    freq : numpy.ndarray
        the frequency axis of the given data, in units of GHz
    timeseries_grid : numpy.ma.MaskedArray
        The time series data of the scan block. It has shape (n_int, nchan). 
    sd_threshold : : float
        the number of standard deviations above the median spectrum averaged 
        total power to consider an integration to have RFI. Integrations exceeding
        this criteria will be flagged for replacement with data containing less RFI.
    band_allocation : str
        the key to identify which set of known band allocations to use for RFI flagging
    channels : list of str
        A list of channel names to loop over. These channel names coorrespond to a 
        list containing a pair of [lower, upper] frequency bounds in units of GHz

    Returns:
    ----------------
    all_flagged : numpy.ndarray
        A list of unique indices corresponding to the flagged subintegrations across 
        all checked channels
    """
    all_flagged = []
    for chan in channels:
        low_f, high_f = band_allocation_ghz_dict[band_allocation][chan]
        time_slice = chunk_timeseries(low_f, high_f, freq, timeseries_grid)
        bad_indices = find_RFI_integrations(time_slice, N=sd_threshold)
        all_flagged.append(bad_indices)
    all_flagged = np.unique(np.hstack(all_flagged))
    return all_flagged

def calibrate_Ta(sdf, tpsb, i=0, replace_RFI=False, n_SD=1, band_allocation="none", channels=[]):
    """
    Take a scan block and use noise diode and system temperature data to calibrate 
    to antenna temperature (Ta). This is a modification of standard calibration method, 
    the initial process is outlined in 
    https://www.gb.nrao.edu/GBT/DA/gbtidl/gbtidl_calibration.pdf

    There is also the option to pass a list of band allocations and a flagging threshold
    to search for, flag, and clean integrations that have strong RFI in the reference data. 
    This is useful when dealing with strong RFI that passes through the telescope beam on 
    short timescales, such that there is a significant change in received intensity when 
    switching from CALOFF to CALON or vice versa. 

    Arguments:
    ----------------
    sdf : dysh.fits.gbtfitsload.GBTOffline
        a Dysh object holding the loaded data of a GBT observation
    tpsb : dysh.spectra.scan.ScanBlock
        total power scan block that contains all the relevant data for an observation
    i : int
        index of the tpsb object. Defaults to zero. This is used to index through individual scans
    replace_RFI : bool
        A flag specifying whether to clean integrations with strong RFI 
    n_SD : float
        the number of standard deviations above the median spectrum averaged 
        total power to consider an integration to have RFI. Integrations exceeding
        this criteria will be flagged for replacement with data containing less RFI.
    band_allocation : str
        the key to identify which set of known band allocations to use for RFI flagging
    channels : list of str
        A list of channel names to loop over. These channel names coorrespond to a 
        list containing a pair of [lower, upper] frequency bounds in units of GHz

    Returns:
    ----------------
    flux : numpy.ma.MaskedArray
        masked array containing the average spectrum of the given data
    freq : numpy.ndarray
        the frequency axis of the given data
    Ta : numpy.ma.MaskedArray
        The calibrated time series data of the scan block. It has shape (n_int, nchan)
        and has units of antenna temperature [Kelvin]
    """
    # pull scan metadata from tpsb object
    scan = tpsb[i].meta[0]["SCAN"]
    ifnum = tpsb[i].meta[0]["IFNUM"]
    plnum = tpsb[i].meta[0]["PLNUM"]
    fdnum = tpsb[i].meta[0]["FDNUM"]

    # read in the data, keeping both a calon and caloff set
    freq,  cal_ts, tsys = get_spectrum_and_freq(sdf, calstate=True, scan=scan, ifnum=ifnum, plnum=plnum, fdnum=fdnum)
    freq, nocal_ts, tsys = get_spectrum_and_freq(sdf, calstate=False, scan=scan, ifnum=ifnum, plnum=plnum, fdnum=fdnum)
    assert cal_ts.shape == nocal_ts.shape, "data shapes do not match: %s vs %s" %(cal_ts.shape, nocal_ts.shape)
    
    # replace the bad integrations from the off data
    if replace_RFI:
        # this will need to be changed to correctly calibrate the data
        cal_ts = np.copy(nocal_ts)
        nocal_ts = replace_bad_integrations(freq, nocal_ts, n_SD=n_SD, band_allocation=band_allocation, channels=channels)

    Ta = tsys[:, np.newaxis] * ((cal_ts - nocal_ts) / nocal_ts)
    flux = np.ma.mean(Ta, axis=0)

    return flux, freq, Ta
