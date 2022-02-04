import logging
import numpy as np
from igorwriter import IgorWave
from typing import List, Optional, Union, Tuple, Callable, Any, Dict
import lmfit as lm
from scipy.interpolate import interp1d

def save_to_igor_itx(file_path: str, xs: List[np.ndarray], datas: List[np.ndarray], names: List[str],
                     ys: Optional[List[np.ndarray]] = None,
                     x_labels: Optional[Union[str, List[str]]] = None,
                     y_labels: Optional[Union[str, List[str]]] = None):
    """Save data to a .itx file which can be dropped into Igor"""

    def check_axis_linear(arr: np.ndarray, axis: str, name: str, current_waves: list) -> bool:
        if arr.shape[-1] > 1 and not np.all(np.isclose(np.diff(arr), np.diff(arr)[0])):
            logging.warning(f"{file_path}: Igor doesn't support a non-linear {axis}-axis. Saving as separate wave")
            axis_wave = IgorWave(arr, name=name + f'_{axis}')
            current_waves.append(axis_wave)
            return False
        else:
            return True

    if x_labels is None or isinstance(x_labels, str):
        x_labels = [x_labels] * len(datas)
    if y_labels is None or isinstance(y_labels, str):
        y_labels = [y_labels] * len(datas)
    if ys is None:
        ys = [None] * len(datas)
    assert all([len(datas) == len(list_) for list_ in [xs, names, x_labels, y_labels]])

    waves = []
    for x, y, data, name, x_label, y_label in zip(xs, ys, datas, names, x_labels, y_labels):
        wave = IgorWave(data, name=name)
        if x is not None:
            if check_axis_linear(x, 'x', name, waves):
                wave.set_dimscale('x', x[0], np.mean(np.diff(x)), units=x_label)
        if y is not None:
            if check_axis_linear(y, 'y', name, waves):
                wave.set_dimscale('y', y[0], np.mean(np.diff(y)), units=y_label)
        elif y_label is not None:
            wave.set_datascale(y_label)
        waves.append(wave)

    with open(file_path, 'w') as fp:
        for wave in waves:
            wave.save_itx(fp, image=True)  # Image = True hopefully makes np and igor match in x/y

            
def convert_to_4_setpoint_AW(aw: np.ndarray):
    """
    Takes a single AW (which may include ramping steps) and returns an AW with 4 setpoints only (0, +, 0, -)
    Args:
        aw (np.ndarray):

    Returns:
        np.ndarray: AW with only 4 setpoints but same length as original
    """
    aw = np.asanyarray(aw)
    assert aw.ndim == 2
    full_len = np.sum(aw[1])
    assert full_len % 4 == 0
    new_aw = np.ndarray((2, 4), np.float32)

    # split Setpoints/lens into 4 sections
    for i, aw_chunk in enumerate(np.reshape(aw.swapaxes(0, 1), (4, -1, 2)).swapaxes(1, 2)):
        sp = aw_chunk[0, -1]  # Last value of chunk (assuming each setpoint handles it's own ramp)
        length = np.sum(aw_chunk[1])
        new_aw[0, i] = sp
        new_aw[1, i] = length
    return new_aw

            
            
def single_wave_masks(arbitrary_wave):
    """
    Generate mask waves for each part of the arbitrary wave applied during scan
    """
    aw = arbitrary_wave
    lens = aw[1].astype(int)
    masks = np.zeros((len(lens), np.sum(lens)), dtype=np.float16)
    for i, m in enumerate(masks):
        s = np.sum(lens[:i])
        m[s:s+lens[i]] = 1
        m[np.where(m == 0)] = np.nan
    return masks


def center_data(x: np.ndarray, data: np.ndarray, centers: Union[List[float], np.ndarray],
                method: str = 'linear', return_x: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Centers data onto x_array. x is required to at least have the same spacing as original x to calculate relative
    difference between rows of data based on center values.

    Args:
        return_x (bool): Whether to return the new x_array as well as centered data
        method (str): Specifies the kind of interpolation as a string
            (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’)
        x (np.ndarray): x_array of original data
        data (np.ndarray): data to center
        centers (Union(list, np.ndarray)): Centers of data in real units of x

    Returns:
        np.ndarray, [np.ndarray]: Array of data with np.nan anywhere outside of interpolation, and optionally new
        x_array where average center has been subtracted
    """
    data = np.atleast_2d(data)
    centers = np.asarray(centers)
    avg_center = np.average(centers)
    nx = np.linspace(x[0] - avg_center, x[-1] - avg_center, data.shape[-1])
    ndata = []
    for row, center in zip(data, centers):
        interper = interp1d(x - center, row, kind=method, assume_sorted=False, bounds_error=False)
        ndata.append(interper(nx))
    ndata = np.array(ndata)
    if return_x is True:
        return ndata, nx
    else:
        return ndata


def mean_data(x: np.ndarray, data: np.ndarray, centers: Union[List[float], np.ndarray],
              method: str = 'linear', return_x: bool = False, return_std: bool = False,
              nan_policy: str = 'omit') -> \
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Centers data and then calculates mean and optionally standard deviation from mean
    Args:
        x (np.ndarray):
        data (np.ndarray):
        centers (np.ndarray):
        method (str):
        return_x (bool):
        return_std (bool):
        nan_policy (str): 'omit' to leave NaNs in any column that has > 1 NaN, 'ignore' to do np.nanmean(...)
    Returns:
        np.ndarray, [np.ndarray]: data averaged along axis 0, optionally the centered x and/or the standard deviation of mean
    """
    temp_centered = center_data(x, data, centers, method, return_x=return_x)
    if return_x:
        centered, x = temp_centered
    else:
        centered = temp_centered
        x = None

    if nan_policy == 'omit':
        averaged = np.mean(centered, axis=0)
    elif nan_policy == 'ignore':
        averaged = np.nanmean(centered, axis=0)
    else:
        raise ValueError(f'got {nan_policy} for nan_policy. Must be "omit" or "ignore"')

    ret = [averaged]
    if return_x:
        ret.append(x)
    if return_std:
        ret.append(np.nanstd(data, axis=0))
    if len(ret) == 1:
        ret = ret[0]
    return ret


def bin_data(data: np.ndarray, bin_x: int = 1, bin_y: int = 1, bin_z: int = 1, stdev: bool = False) -> \
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Bins up to 3D data in x then y then z. If bin_y == 1 then it will only bin in x direction (similar for z)
    )

    Args:
        data (np.ndarray): 1D, 2D or 3D data to bin in x and or y axis and or z axis
        bin_x (): Bin size in x
        bin_y (): Bin size in y
        bin_z (): Bin size in z
        stdev (bool): If True will additionally return the standard deviation of each bin
    Returns:

    """
    ndim = data.ndim
    data = np.array(data, ndmin=3)  # Force 3D so same function works for 1, 2, 3D
    original_shape = data.shape
    num_z, num_y, num_x = [np.floor(s / b).astype(int) for s, b in zip(data.shape, [bin_z, bin_y, bin_x])]
    # ^^ Floor so e.g. num_x*bin_x does not exceed len x
    chop_z, chop_y, chop_x = [s - n * b for s, n, b in zip(data.shape, [num_z, num_y, num_x], [bin_z, bin_y, bin_x])]
    # ^^ How much needs to be chopped off in total to make it a nice round number
    data = data[
           np.floor(chop_z / 2).astype(int): original_shape[0] - np.ceil(chop_z / 2).astype(int),
           np.floor(chop_y / 2).astype(int): original_shape[1] - np.ceil(chop_y / 2).astype(int),
           np.floor(chop_x / 2).astype(int): original_shape[2] - np.ceil(chop_x / 2).astype(int)
           ]

    data = data.reshape(num_z, bin_z, num_y, bin_y, num_x, bin_x)  # Break up into bin sections
    data = np.moveaxis(data, [1, 3, 5], [-3, -2, -1])  # Put all parts we want to average over at the end
    data = data.reshape((num_z, num_y, num_x, -1))  # Combine all parts that want to be averaged
    if stdev:
        std = data.std(axis=-1)
    else:
        std = None
    data = data.mean(axis=-1)

    if ndim == 3:
        pass
    elif ndim == 2:
        data = data[0]
    elif ndim == 1:
        data = data[0, 0]

    if stdev:
        if ndim == 3:
            pass
        elif ndim == 2:
            std = std[0]
        elif ndim == 1:
            std = std[0, 0]
        return data, std
    else:
        return data
    
    
def i_sense(x, mid, theta, amp, lin, const):
    """ Simple weakly coupled charge transition shape """
    arg = (x - mid) / (2 * theta)
    return -amp / 2 * np.tanh(arg) + lin * (x - mid) + const


def calculate_fit(x: np.ndarray, data: np.ndarray, params: lm.Parameters, func: Callable[[Any], float],
                  auto_bin=True, min_bins=1000,
                  method: str = 'leastsq',
                  ) -> lm.model.ModelResult:
    """
    Calculates fit on data (Note: assumes that 'x' is the independent variable in fit_func)
    Args:
        x (np.ndarray): x_array (Note: fit_func should have variable with name 'x')
        data (np.ndarray): Data to fit
        params (lm.Parameters): Initial parameters for fit
        func (Callable): Function to fit to
        auto_bin (bool): if True will bin data into >= min_bins
        min_bins: How many bins to use for binning (actual num will lie between min_bins >= actual > min_bins*1.5)

    Returns:
        (lm.model.ModelResult): Fit result
    """
    model = lm.model.Model(func)
    
    if auto_bin and data.shape[-1] > min_bins*2:  # between 1-2x min_bins won't actually end up binning
        bin_size = int(np.floor(data.shape[-1] / min_bins))  # Will end up with >= self.AUTO_BIN_SIZE pts
        x, data = [bin_data(arr, bin_x=bin_size) for arr in [x, data]]
        
    try:
        fit = model.fit(data.astype(np.float32), params, x=x.astype(np.float32), nan_policy='omit', method=method)
        if fit.covar is None and fit.success is True:  # Failed to calculate uncertainties even though fit
            # was successful
            logging.warning(f'Uncertainties failed')
        elif fit.success is False:
            logging.warning(f'Fit failed')
    except TypeError as e:
        logging.error(f'{e} while fitting')
        fit = None
    return fit


