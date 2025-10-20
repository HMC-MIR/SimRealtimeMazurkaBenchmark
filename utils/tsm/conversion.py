# This file contains functions for converting from an alignmentpath to a TSM path
import warnings
import numpy as np
import utils.constants as constants
from numba import njit

@njit(cache=True)
def sigmoid(x):
    """
    Compute the sigmoid function for a given input x. sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))

def _get_alpha_numba(
    path,
    i,
    history=100,
    default_alpha=1.0,
    max_timewarp_factor=2.0,
    x=np.array([], dtype=np.float64),
    sum_x=-1.0,  # -1.0 means not computed yet
    sum_x2=-1.0,  # -1.0 means not computed yet
):
    """
    Calculate the alpha (playback speed factor) value based on the path and desired history window.
    Optionally uses precomputed x, sum_x, and sum_x2 for efficiency.

    Args:
        path (array): The current alignment path as a numpy array.
        i (int): The current index in the path for which to calculate the alpha value.
        history (int): The number of previous locations to consider for calculating the alpha value.
        default_alpha (float): The default alpha value to return if not enough history is available.
        max_timewarp_factor (float): Maximum time warp factor for TSM.
        x (array, optional): Precomputed x values for the current history window.
        sum_x (float, optional): Precomputed sum of x values.
        sum_x2 (float, optional): Precomputed sum of squares of x values.

    Returns:
        float: The calculated alpha value.
    """
    if i + 1 < history:
        return default_alpha

    # Get the last `history` locations
    if x.size == 0:
        x = np.arange(i - history + 1, i + 1, dtype=np.float64)  # query
    y = np.array(path[i - history + 1 : i + 1], dtype=np.float64)  # ref

    # Calculate sums for least squares formula
    sum_x, sum_y, sum_xy, sum_x2 = _calc_sums(x, y, sum_x, sum_x2)

    # Calculate alpha
    alpha = _calc_alpha(
        sum_x, sum_x2, sum_y, sum_xy, history, max_timewarp_factor, default_alpha
    )
    return alpha


@njit
def _calc_sums(x, y, sum_x, sum_x2):
    """helper function to calculate sums"""
    if sum_x == -1.0:
        sum_x = np.sum(x)
    if sum_x2 == -1.0:
        sum_x2 = np.sum(x * x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    return sum_x, sum_y, sum_xy, sum_x2


@njit
def _calc_alpha(
    sum_x, sum_x2, sum_y, sum_xy, history, max_timewarp_factor, default_alpha
):
    """helper function to calculate alpha"""
    denominator = history * sum_x2 - sum_x * sum_x

    # Handle edge case where denominator is zero (all x values are the same)
    if abs(denominator) < 1e-10:
        return default_alpha

    # Calculate slope using manual least squares formula
    # For line y = mx + c, slope m = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    m = (history * sum_xy - sum_x * sum_y) / denominator
    alpha = 1.0 / m if m != 0 else default_alpha  # alpha is inverse of slope

    # Clip m to be within a reasonable range
    alpha = max(1.0 / max_timewarp_factor, min(max_timewarp_factor, alpha))

    return alpha


@njit(cache=True)
def calc_scale_from_deviation(deviation, sensitivity, max_scale):
    """Calculate the alpha adjustment scale from the deviation between current TSM position and predicted alignment position.

    Args:
        deviation (float): The deviation between current TSM position and predicted alignment position.
        sensitivity (float): The sensitivity of the scale. Higher sensitivity means more aggressive scaling and potentially more unstable.
        max_scale (float): The maximum scale factor for alpha adjustment.

    Returns:
        float: The scale factor for alpha adjustment.
    """

    # Map deviation to a scaling factor
    normalized_deviation = sensitivity * deviation

    # use sigmoid to map deviation to a scale factor in (0, 1)
    dev_scale = sigmoid(normalized_deviation)

    # Rescale to [-1, 1]
    scale = -1 + 2 * dev_scale

    # Rescale to [-max_scale, max_scale]
    scale = max_scale**scale

    return scale


class AlphaCache:
    """Cache for alpha values.

    This class is used to cache the alpha values for the runtime alignment.

    Args:
        history (int): The number of previous locations to consider for calculating the alpha value.
    """

    def __init__(self, history=constants.DEFAULT_ALPHA_LOOKBACK):
        """Initialize the cache.

        Args:
            history (int): The number of previous locations to consider for calculating the alpha value.
        """
        self.history = history
        self.x = np.arange(-history + 1, 1, dtype=np.float64)  # query
        self.sum_x = np.sum(self.x)
        self.sum_x2 = np.sum(self.x * self.x)



def to_tsm_path(alignment_path, ref_length=None, query_length=None, lag=0):
    """Converts raw alignment path to TSM path

    Args:
        alignment_path (np.ndarray): Alignment path of shape (2, N), where the first row is the query indices and the second row is the reference indices
        ref_length (int): Length of reference, in feature frames
        query_length (int): Length of query, in feature frames
        lag (int): Lag to apply to the TSM path
    Returns:
        np.ndarray: TSM path in frames
    """
    tsm_path = [0.0]

    # resample alignment path so that there is one reference frame per query frame
    if query_length is None:
        query_length = alignment_path[0][-1]
    alignment_path = np.interp(
        np.arange(query_length), alignment_path[0], alignment_path[1]
    )
    alignment_path = alignment_path.astype(int)

    # initialize TSM
    if ref_length is None:
        ref_length = alignment_path[-1]
    xh = ref_length * constants.DEFAULT_HOP_LENGTH
    pos = 0  # initial TSM position

    # intiailize alignment parameters
    _alpha_cache = AlphaCache(history=constants.DEFAULT_ALPHA_LOOKBACK)
    current_alpha = 1.0
    base_alpha = 1.0
    applied_alpha = 1.0
    query_alpha_history = []
    alpha_history = []
    lag_frames = int(lag / 1000 * constants.DEFAULT_SR / constants.DEFAULT_HOP_LENGTH)
    lag_samples = int(lag / 1000 * constants.DEFAULT_SR)

    for i, ref_frame in enumerate(alignment_path):
        # update the alpha value
        if (i + 1) % constants.DEFAULT_ALPHA_UPDATE_FREQUENCY == 0:
            new_alpha = _get_alpha_numba(
                alignment_path,
                i - lag_frames,
                history=constants.DEFAULT_ALPHA_LOOKBACK,
                default_alpha=1.0,
                max_timewarp_factor=constants.DEFAULT_MAX_TIMEWARP_FACTOR,
                x=_alpha_cache.x,
                sum_x=_alpha_cache.sum_x,
                sum_x2=_alpha_cache.sum_x2,
            )
            if (
                abs(new_alpha - current_alpha) > 0.01
            ):  # only update if the change is greater than 1%
                current_alpha = new_alpha
                base_alpha = current_alpha

        # adjust alpha value based on deviation
        elif (i + 1) % constants.DEFAULT_ALPHA_ADJUST_FREQUENCY == 0:
            curr_frame = int(round((pos - lag_samples) / constants.DEFAULT_HOP_LENGTH))
            latest_frame = max(0, int(i - lag_frames))  # in feature frames
            deviation = curr_frame - (alignment_path[latest_frame] - alignment_path[0])
            scale = calc_scale_from_deviation(
                deviation,
                constants.DEFAULT_ALPHA_ADJUST_SENSITIVITY,
                constants.DEFAULT_ALPHA_ADJUST_MAX_SCALE,
            )
            current_alpha = base_alpha * scale
            current_alpha = np.clip(
                current_alpha,
                1 / constants.DEFAULT_MAX_TIMEWARP_FACTOR,
                constants.DEFAULT_MAX_TIMEWARP_FACTOR,
            )
        query_alpha_history.append(current_alpha)

        # update TSM if not at end of reference
        if pos < xh:
            ref_frame = int(pos / constants.DEFAULT_HOP_LENGTH)

            # search for query frame
            query_frame = 0  # default to first frame
            search_start = min(i - 1, 100)
            for j in range(i - 1, max(-1, i - search_start - 1), -1):
                if alignment_path[j] <= ref_frame:
                    query_frame = j
                    break
            else:  # if no query frame is found, use the last query frame
                query_frame = i - 1
            query_frame = max(0, min(query_frame, i - 1))

            # update TSM alpha
            try:
                applied_alpha = query_alpha_history[query_frame]
            except IndexError:
                if len(query_alpha_history) > 0:
                    applied_alpha = query_alpha_history[-1]
                    warnings.warn(
                        f"No alpha history for query frame {query_frame}, using last alpha"
                    )
                else:
                    applied_alpha = 1.0
                    warnings.warn("No alphas have been computed yet, using 1.0")

            # perform simulated TSM update
            Ha = constants.DEFAULT_HOP_LENGTH / applied_alpha
            pos += Ha
            
            tsm_path.append(pos / constants.DEFAULT_HOP_LENGTH)
            alpha_history.append(applied_alpha)

        else:  # at end of reference
            print(f"End of reference reached after frame {i}. pos: {pos}, Ha: {Ha}")
            break

    # make into 2xN array with query and reference indices
    tsm_path = np.array([np.arange(len(tsm_path)), tsm_path])
    return tsm_path