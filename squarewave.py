"""All the functions for processing I_sense data into the various steps of square wave heated data"""
from dataclasses import dataclass, field

import lmfit as lm
import numpy as np
from common import full_wave_masks, center_data, calculate_fit, i_sense
from typing import Union, Optional, List
import logging


@dataclass
class SquareProcessedData:
    """Class to store all calculated data"""
    # Data that will be calculated
    x: np.ndarray = field(default=None, repr=False)  # x_array with length of num_steps (for cycled, averaged, entropy)
    chunked: np.ndarray = field(default=None, repr=False)  # Data broken in to chunks based on AWG (just plot
    # raw_data on orig_x_array)
    setpoint_averaged: np.ndarray = field(default=None, repr=False)  # Setpoints averaged only
    setpoint_averaged_x: np.ndarray = field(default=None, repr=False)  # x_array for setpoints averaged only
    cycled: np.ndarray = field(default=None, repr=False)  # setpoint averaged and then cycles averaged data
    entropy_signal: np.ndarray = field(default=None, repr=False)  # 2D Entropy signal data

    centers_used: np.ndarray = None  # Center values used when averaging data
    averaged: np.ndarray = field(default=None, repr=False)  # setpoint averaged, cycle_avg, then averaged in y
    average_entropy_signal: np.ndarray = field(default=None, repr=False)  # Averaged Entropy signal
        
    def transition_part(self, which: str = 'cold', data: np.ndarray = None) -> np.ndarray:
        """Return specific part of data which has already been separated into the 4 square wave parts
        e.g. data.shape = (4, ...)
        """
        if data is None:
            data = self.averaged
        parts = get_transition_parts(which)
        return np.nanmean(data[parts, :], axis=0)
    
    
def process_per_row_parts(x: np.ndarray, data: np.ndarray,
                          arbitrary_wave: np.ndarray, num_steps: int,
                          setpoint_start: int,
                          num_cycles=1) -> SquareProcessedData:
    """
    Separate data into relevant parts based on the square wave heating applied (no averaging of 2D data as that
    requires centering data first)

    Args:
        x (): x-array of data
        data (): i_sense data as recorded (i.e. normal 1D or 2D)
        arbitrary_wave (): 2D wave that contains info about setpoint values and durations of square wave (should have exactly 4 setpoints)
        num_steps (): No. DAC steps in sweep
        setpoint_start (): What sample to start
        num_cycles (): Number of full square wave cycles per DAC step

    Returns:
        (SquareProcessedData): Partially filled Output (missing averaged parts)

    """
    output = SquareProcessedData()
    
    # Calculate true x_array (num_steps)
    output.x = np.linspace(x[0], x[-1], num_steps)

    # Get chunked data (setpoints, ylen, numsteps, numcycles, splen)
    full_wave_masks_ = full_wave_masks(arbitrary_wave, num_steps, num_cycles)
    setpoint_length = data.shape[-1]/num_steps/num_cycles/4  # For now this is always fixed for equal setpoint lengths
    output.chunked = chunk_data(data, full_wave_masks=full_wave_masks_, setpoint_lengths=[setpoint_length]*4,
                                num_steps=num_steps, num_cycles=num_cycles)

    # Average setpoints of data ([ylen], setpoints, numsteps, numcycles)
    output.setpoint_averaged = average_setpoints(output.chunked, start_index=setpoint_start,
                                                 fin_index=None)
    output.setpoint_averaged_x = np.linspace(x[0], x[-1],
                                             num_steps * num_cycles)

    # Averaged cycles ([ylen], setpoints, numsteps)
    output.cycled = average_cycles(output.setpoint_averaged, start_cycle=None, fin_cycle=None)  # Not using this at the moment

    # Per Row Entropy signal
    output.entropy_signal = entropy_signal(np.moveaxis(output.cycled, 1, 0))  # Moving setpoint axis to be first

    return output


def process_avg_parts(partial_output: SquareProcessedData, centers: np.ndarray) -> SquareProcessedData:
    """
    Calculate the average i_sense and entropy signals after centering data based on cold transition fits
    
    Args:
        partial_output ():
        centers (): The center positions to use for averaging. If None, data will be centered with a default transition fit

    Returns:
        (SquareProcessedData): Filled output (i.e. including averaged data and ent
    """
    out = partial_output
    # Center and average 2D data or skip for 1D
    
    out.x, out.averaged, out.centers_used = average_2D(out.x,
                                                       out.cycled,
                                                       centers=centers,
                                                       avg_nans=False)

    # Avg Entropy signal
    out.average_entropy_signal = entropy_signal(out.averaged)

    return out


def chunk_data(data, full_wave_masks: np.ndarray, setpoint_lengths: List[int], num_steps: int, num_cycles: int) -> List[
    np.ndarray]:
    """
    Breaks up data into chunks which make more sense for square wave heating datasets.
    Args:
        data (np.ndarray): 1D or 2D data (full data to match original x_array).
            Note: will return with y dim regardless of 1D or 2D

    Returns:
        List[np.ndarray]: Data broken up into chunks [setpoints, np.ndarray(ylen, num_steps, num_cycles, sp_len)].
            NOTE: Has to be a list returned and not a ndarray because sp_len may vary per steps
            NOTE: This is the only step where setpoints should come first, once sp_len binned it should be ylen first
    """
    masks = full_wave_masks
    zs = []
    for mask, sp_len in zip(masks, setpoint_lengths):
        sp_len = int(sp_len)
        z = np.atleast_2d(data)  # Always assume 2D data
        zm = z * mask  # Mask data
        zm = zm[~np.isnan(zm)]  # remove blanks
        zm = zm.reshape(z.shape[0], num_steps, num_cycles, sp_len)
        zs.append(zm)
    return zs


def average_setpoints(chunked_data, start_index=None, fin_index=None):
    """ Averages last index of AWG data passed in from index s to f.

    Args:
        chunked_data (List[np.ndarray]): List of datas chunked nicely for AWG data.
            dimensions (num_setpoints_per_cycle, (len(y), num_steps, num_cycles, sp_len))
        start_index (Union(int, None)): Start index to average in each setpoint chunk
        fin_index (Union(int, None)): Final index to average to in each setpoint chunk (can be negative)

    Returns:
        np.ndarray: Array of zs with averaged last dimension. ([ylen], setpoints, num_steps, num_cycles)
        Can be an array here because will always have 1 value per
        averaged chunk of data (i.e. can't have different last dimension any more)
    """

    assert np.all([arr.ndim == 4 for arr in chunked_data])  # Assumes [setpoints, (ylen, num_steps, num_cycles, sp_len)]
    nz = []
    for z in chunked_data:
        z = np.moveaxis(z, -1, 0)  # move sp_len to first axis to make mean nicer
        nz.append(np.mean(z[start_index:fin_index], axis=0))

    # nz = [np.mean(z[:, :, :, start_index:fin_index], axis=3) for z in chunked_data]  # Average the last dimension
    nz = np.moveaxis(np.array(nz), 0, 1)  # So that ylen is first now
    # (ylen, setpoins, num_steps, num_cycles)

    if nz.shape[0] == 1:  # Remove ylen dimension if len == 1
        nz = np.squeeze(nz, axis=0)
    return np.array(nz)


def average_cycles(binned_data, start_cycle=None, fin_cycle=None):
    """
    Average values from cycles from start_cycle to fin_cycle
    Args:
        binned_data (np.ndarray): Binned AWG data with shape ([ylen], setpoints, num_steps, num_cycles)
        start_cycle (Union(int, None)): Cycle to start averaging from
        fin_cycle (Union(int, None)): Cycle to finish averaging on (can be negative to count backwards)

    Returns:
        np.ndarray: Averaged data with shape ([ylen], setpoints, num_steps)

    """
    # [y], setpoints, numsteps, cycles
    data = np.array(binned_data, ndmin=4)  # [y], setpoints, numsteps, cycles
    averaged = np.mean(np.moveaxis(data, -1, 0)[start_cycle:fin_cycle], axis=0)
    if averaged.shape[0] == 1:  # Return 1D or 2D depending on y_len
        averaged = np.squeeze(averaged, axis=0)
    return averaged


def average_2D(x: np.ndarray, data: np.ndarray, centers: Optional[np.ndarray] = None, avg_nans: bool = False):
    """
    Averages data in y direction (aligns first if centers passed, otherwise blind average)
    Args:
        x (np.ndarray): Original x_array for data
        data (np.ndarray): Data after binning and cycle averaging. Shape ([ylen], setpoints, num_steps)
        centers (Optional[np.ndarray]): Optional center positions to use instead of standard automatic transition fits
        avg_nans (bool): Whether to average data which includes NaNs (useful for two part entropy scans)
    Returns:
        Tuple[np.ndarray, np.ndarray]: New x_array, averaged_data (shape (setpoints, num_steps))
    """
    if data.ndim == 3:
        if centers is None:
            centers = [0]*data.shape[0]
        nzs = []
        nxs = []
        for z in np.moveaxis(data, 1, 0):  # For each of v0_0, vP, v0_1, vM
            nz, nx = center_data(x, z, centers, return_x=True)
            nzs.append(nz)
            nxs.append(nx)
        assert (nxs[0] == nxs).all()  # Should all have the same x_array
        ndata = np.array(nzs)
        if avg_nans is True:
            ndata = np.nanmean(ndata, axis=1)  # Average centered data
        else:
            ndata = np.mean(ndata, axis=1)  # Average centered data
        nx = nxs[0]
    else:
        nx = x
        ndata = data
        logging.info(f'Data passed in was {data.ndim - 1}D (not 2D), same values returned')
    return nx, ndata, centers


def calculate_centers(data: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate center of transition for 1D or 2D data

    Args:
        data (): 1D or 2D transition data
        x: x-array of data (defaults to index value)

    Returns:
        center values
    """
    data = np.atleast_2d(data)
    if x is None:
        x = np.arange(data.shape[-1])

    params = lm.Parameters()
    params.add_many(
        ('mid', 0, True, None, None, None, None),
        ('theta', 5, True, 0.01, None, None, None),
        ('amp', 1, True, 0, None, None, None),
        ('lin', 0.001, True, 0, None, None, None),
        ('const', 5, True, None, None, None, None)
    )

    z0s = data[:, (0, 2)]
    z0_avg_per_row = np.mean(z0s, axis=1)  # Average together both 0 heating parts

    fits = [calculate_fit(x, z, params, func=i_sense, auto_bin=True) for z in z0_avg_per_row]
    centers = np.array([f.params['mid'].value if f else np.nan for f in fits])
    if np.any([fit is None for fit in fits]):  # Not able to do transition fits for some reason
        logging.warning(f'{np.sum([1 if fit is None else 0 for fit in fits])} of {len(fits)} transition fits failed')
        centers[np.isnan(centers)] = np.nanmean(centers)  # Replace with average center value, probably a good guess
    return centers


def entropy_signal(data: np.ndarray) -> np.ndarray:
    """
    Calculates equivalent of second harmonic from data with v0_0, vP, v0_1, vM as first dimension
    Note: Data should be aligned for same x_array before doing this
    Args:
        data (np.ndarray): Data with first dimension corresponding to v0_0, vP, v0_1, vM. Can be any dimensions for rest

    Returns:
        np.ndarray: Entropy signal array with same shape as data minus the first axis

    """
    assert data.shape[0] == 4
    entropy_data = -1 * (np.mean(data[(1, 3),], axis=0) - np.mean(data[(0, 2),], axis=0))
    return entropy_data


def get_transition_parts(part: str) -> Union[tuple, int]:
    if isinstance(part, str):
        part = part.lower()
        if part == 'cold':
            parts = (0, 2)
        elif part == 'hot':
            parts = (1, 3)
        elif part == 'vp':
            parts = (1,)
        elif part == 'vm':
            parts = (3,)
        else:
            raise ValueError(f'{part} not recognized. Should be in ["hot", "cold", "vp", "vm"]')
    elif isinstance(part, int):
        parts = part
    else:
        raise ValueError(f'{part} not recognized. Should be in ["hot", "cold", "vp", "vm"]')
    return parts


def get_transition_part(data: np.ndarray, part: Union[str, int]) -> np.ndarray:
    """
    Returns the specified part of I_sense data (i.e. for square wave heating analysis)
    Args:
        data (): I_sense data where axis [-2] has shape 4 (i.e. split into the separate parts of square wave)
        part (): Which part out of 'cold', 'hot', 0, 1, 2, 3 to return

    Returns:

    """
    assert data.shape[-2] == 4  # If not 4, then it isn't square wave transition data

    parts = get_transition_parts(part=part)

    data = np.take(data, parts, axis=-2)
    data = np.mean(data, axis=-2)
    return data


if __name__ == "__main__":
    print('Succesfully loaded')