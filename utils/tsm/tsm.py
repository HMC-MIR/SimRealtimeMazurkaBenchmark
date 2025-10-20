# standard library imports
import threading
import time
import warnings

# custom imports
import pyaudio
import numpy as np
import librosa as lb
from scipy.signal import medfilt
from numba import njit
import utils.constants as constants
from utils.cache_utils import generate_cache_key, load_cache, store_cache


@njit(cache=True)
def calc_sum_squared_window(window, hop_length):
    """
    Calculates the denominator term for computing synthesis frames.

    Inputs
    window: array specifying the window used in FFT analysis
    hop_length: the synthesis hop size in samples

    Returns an array specifying the normalization factor.
    """
    assert (
        len(window) % hop_length == 0
    ), "Hop length does not divide the window evenly."

    numShifts = len(window) // hop_length
    den = np.zeros_like(window)
    i_times_hop_length = 0
    for i in range(numShifts):
        den += np.roll(np.square(window), i_times_hop_length)
        i_times_hop_length += hop_length

    return den


@njit(cache=True)
def construct_full_stft(S):
    """
    Constructs the full STFT matrix from the half-spectrum.

    Args:
        S (np.ndarray): The STFT half-spectrum.

    Returns:
        np.ndarray: The full STFT matrix.
    """
    fft_size = (S.shape[0] - 1) * 2
    Sfull = np.zeros((fft_size, S.shape[1]), dtype=np.complex64)
    Sfull[0 : S.shape[0], :] = S
    Sfull[S.shape[0] :, :] = np.conj(np.flipud(S[1 : fft_size // 2, :]))
    return Sfull


def compute_inverse_ffts(Sfull):
    """
    Computes inverse FFTs for all frames.

    Args:
        Sfull (np.ndarray): The full STFT matrix.

    Returns:
        np.ndarray: The time-domain frames.
    """
    frames = np.zeros_like(Sfull)
    for i in range(frames.shape[1]):
        frames[:, i] = np.fft.ifft(Sfull[:, i])
    return np.real(frames)  # remove imaginary components due to numerical roundoff


@njit(cache=True)
def apply_window_and_normalize(frames, window, den):
    """
    Applies window and normalization to frames.

    Args:
        frames (np.ndarray): The time-domain frames.
        window (np.ndarray): The window function.
        den (np.ndarray): The normalization denominator.

    Returns:
        np.ndarray: The windowed and normalized frames.
    """
    return frames * window.reshape((-1, 1)) / den.reshape((-1, 1))


@njit(cache=True)
def reconstruct_signal(frames, hop_length, L):
    """
    Reconstructs the signal from overlapping frames.

    Args:
        frames (np.ndarray): The windowed and normalized frames.
        hop_length (int): The hop length for the STFT.
        L (int): The length of the signal.

    Returns:
        np.ndarray: The reconstructed signal.
    """
    y = np.zeros(hop_length * (frames.shape[1] - 1) + L)
    offset = 0
    for i in range(frames.shape[1]):
        y[offset : offset + L] += frames[:, i]
        offset += hop_length
    return y


def invert_stft(S, hop_length, den, window, L):
    """
    Inverts the STFT of a signal.

    Args:
        S (np.ndarray): The STFT of the signal.
        hop_length (int): The hop length for the STFT.
        den (np.ndarray): The denominator for the STFT.
        window (np.ndarray): The window for the STFT.
        L (int): The length of the signal.

    Returns:
        np.ndarray: The reconstructed signal.
    """
    # construct full stft matrix
    Sfull = construct_full_stft(S)

    # compute inverse FFTs
    frames = compute_inverse_ffts(Sfull)

    # synthesis frames
    frames = apply_window_and_normalize(frames, window, den)

    # reconstruction
    y = reconstruct_signal(frames, hop_length, L)

    return y


@njit(cache=True)
def float2pcm_int16(sig):
    """
    Converts a floating point signal to int16 PCM format.

    Args:
        sig (np.ndarray): The signal to convert.

    Returns:
        np.ndarray: The converted signal as int16.
    """
    # assert sig <= 1 and sig >= -1, "Data must be normalized between -1.0 and 1.0"
    sig = np.asarray(sig)
    abs_max = 32767  # 2^15 - 1 for int16
    return (sig * abs_max).clip(-32768, 32767).astype(np.int16)


def float2pcm(sig, dtype="int16"):
    """
    Converts a floating point signal to PCM format.

    Args:
        sig (np.ndarray): The signal to convert.
        dtype (str): The data type to convert to.

    Returns:
        np.ndarray: The converted signal.
    """
    if dtype == "int16":
        return float2pcm_int16(sig)
    else:
        # Fallback for other dtypes
        sig = np.asarray(sig)
        dtype = np.dtype(dtype)
        i = np.iinfo(dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


@njit(cache=True)
def hanning_window(n):
    """
    Creates a Hanning window of length n.

    Args:
        n (int): Window length

    Returns:
        np.ndarray: Hanning window
    """
    window = np.zeros(n)
    for i in range(n):
        window[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (n - 1)))
    return window


@njit(cache=True)
def ola(ratio, L, pos, xp, L_ola, Ha_ola, Hs_ola):
    """
    Overlap-add processing.
    Supports fractional analysis hop by rounding indices at access time.
    """
    ola_y = np.zeros(L)
    hanning_win = hanning_window(L_ola)
    offset = 0
    for i in range(ratio):
        # Round computed analysis position to nearest sample for indexing
        ola_start = int(np.rint(pos + (Ha_ola * i)))
        ola_end = ola_start + L_ola
        if ola_end <= len(xp) and ola_start >= 0:
            ola_win = xp[ola_start:ola_end]
            ola_win_synth = ola_win * hanning_win
            ola_y[offset : offset + L_ola] += ola_win_synth
        offset += Hs_ola

    return ola_y


@njit(cache=True)
def create_harmonic_percussive_masks(Yh, Yp):
    """
    Creates harmonic and percussive masks based on filtered components.

    Args:
        Yh (np.ndarray): Harmonic filtered magnitude
        Yp (np.ndarray): Percussive filtered magnitude

    Returns:
        tuple: (Mh, Mp) - harmonic and percussive masks
    """
    Mh = Yh > Yp
    Mp = np.logical_not(Mh)
    return Mh, Mp


@njit(cache=True)
def apply_masks_to_stft(X, Mh, Mp):
    """
    Applies harmonic and percussive masks to STFT.

    Args:
        X (np.ndarray): Complex STFT
        Mh (np.ndarray): Harmonic mask
        Mp (np.ndarray): Percussive mask

    Returns:
        tuple: (Xh, Xp) - harmonic and percussive STFT components
    """
    Xh = X * Mh
    Xp = X * Mp
    return Xh, Xp


@njit(cache=True)
def normalize_audio_signal(x):
    """
    Normalizes audio signal to prevent clipping.

    Args:
        x (np.ndarray): Audio signal

    Returns:
        np.ndarray: Normalized audio signal
    """
    max_val = np.max(np.abs(x))
    if max_val > 1.0:
        return x / max_val
    return x


def compute_phase_vocoder_frame(
    xh_segment, window, omega_nom, prev_fft, prev_phase, Ha, sr, Hs
):
    """
    Computes phase vocoder frame for harmonic component.

    Args:
        xh_segment (np.ndarray): Harmonic audio segment
        window (np.ndarray): Window function
        omega_nom (np.ndarray): Nominal frequencies
        prev_fft (np.ndarray): Previous FFT frame
        prev_phase (np.ndarray): Previous phase
        Ha (int): Analysis hop size
        sr (int): Sample rate
        Hs (int): Synthesis hop size

    Returns:
        tuple: (pv_frame_mod, new_prev_phase, new_prev_fft) - processed frame and updated state
    """
    # Apply window
    pv_win = xh_segment * window

    # Compute FFT
    S = np.fft.rfft(pv_win)
    magnitude = np.abs(S)

    new_prev_phase = prev_phase.copy()
    new_prev_fft = S.copy()

    if prev_fft is not None:
        # Phase unwrapping
        dphi = np.angle(S) - np.angle(prev_fft)
        dphi = dphi - omega_nom * (Ha / sr)
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

        # Instantaneous frequency
        w_if = omega_nom + dphi * (sr / Ha)
        new_prev_phase += w_if * (Hs / sr)

    # Reconstruct signal
    X_mod = magnitude * np.exp(1j * new_prev_phase)
    pv_frame_mod = np.fft.irfft(X_mod)

    return pv_frame_mod, new_prev_phase, new_prev_fft


@njit(cache=True)
def update_output_buffer(output_buffer, pv_frame_mod, window, den, Hs):
    """
    Updates the output buffer with new frame.

    Args:
        output_buffer (np.ndarray): Current output buffer
        pv_frame_mod (np.ndarray): Phase vocoder frame
        window (np.ndarray): Window function
        den (np.ndarray): Normalization denominator
        Hs (int): Synthesis hop size

    Returns:
        np.ndarray: Updated output buffer
    """
    # Shift buffer
    output_buffer[:-Hs] = output_buffer[Hs:]
    output_buffer[-Hs:] = 0

    # Add new frame
    output_buffer += pv_frame_mod * (window / den)

    return output_buffer


@njit(cache=True)
def clip_output_buffer(output_buffer):
    """
    Clips output buffer to valid range.

    Args:
        output_buffer (np.ndarray): Output buffer

    Returns:
        np.ndarray: Clipped output buffer
    """
    return np.clip(output_buffer, -32768, 32767)


@njit(cache=True)
def extract_output_chunk(output_buffer, Hs):
    """
    Extracts output chunk for streaming.

    Args:
        output_buffer (np.ndarray): Output buffer
        Hs (int): Synthesis hop size

    Returns:
        np.ndarray: Output chunk
    """
    return output_buffer[:Hs]


class TSM:
    """
    TSM class implements the hybrid realtime tsm algorithm
    """

    def __init__(
        self,
        chunk: int = constants.DEFAULT_CHUNK_SIZE_TSM,
        L: int = 2048,
        L_ola: int = 256,
        sr: int = constants.DEFAULT_SR,
        audio_data: np.ndarray = None,
        recording: bool = False,
        playback: bool = True,
        alpha: float = 1.0,
        cache_dir: str = "cache",
        save_cache: bool = True,
    ):
        """
        Initialize the TSM class

        Args:
            chunk (int): Chunk size for processing. Default is 2048.
            L (int): Window length for STFT. Default is 2048.
            L_ola (int): Overlap-add window length. Default is 256.
            sr (int): Sample rate. Default is 22050.
            audio_data (np.ndarray): Input audio data to be processed.
            recording (bool): Whether to record output. Default is False.
            playback (bool): Whether to play output. Default is True.
            alpha (float): Time stretch factor. Default is 1.0.
            cache_dir (str): Directory to save/load cached harmonic-percussive separation. Default is None.
            save_cache (bool): Whether to save computed separation to cache. Default is True.
        """

        # set parameters
        self.chunk = chunk
        self.L = L
        self.L_ola = L_ola
        self.Hs = L // 4
        self.Hs_ola = L_ola // 2
        self.alpha = alpha
        self.window = hanning_window(L)
        self.output_buffer = np.zeros(L)
        self.prev_fft = None
        self.prev_phase = np.zeros(L // 2 + 1)

        self.recording = recording
        self.playback = playback

        self.audio_data = audio_data
        self.sr = sr

        # timing
        self.tsm_elapsed_t = 0
        self.stream_end_t = 0

        # threading
        self.stop_event = threading.Event()

        # set omega_nom and den
        self.omega_nom = np.arange(L // 2 + 1) * 2 * np.pi * self.sr / L
        self.den = calc_sum_squared_window(self.window, self.Hs)

        # caching
        self.cache_dir = cache_dir
        self.save_cache = save_cache
        self.cache_key = generate_cache_key(self.audio_data)
        self.xh = load_cache(self.cache_dir, self.cache_key, "xh.pkl")
        self.xp = load_cache(self.cache_dir, self.cache_key, "xp.pkl")
        if self.xh is None or self.xp is None:
            self.harmonic_percussive_separation(x=audio_data)

        # normalize xh and xp
        self.xh = normalize_audio_signal(self.xh)
        self.xp = normalize_audio_signal(self.xp)

        # convert to int16
        self.xh = float2pcm(self.xh).astype(np.int16)
        self.xp = float2pcm(self.xp).astype(np.int16)

    def harmonic_percussive_separation(self, x, hop_length=512, lh=6, lp=6):
        """
        Separates the harmonic and percussive components of the input signal.

        Args:
            x (np.ndarray): Input audio data to be processed.
            hop_length (int, optional): The hop length for the STFT. Defaults to 512.
            lh (int, optional): Harmonic filter size. Defaults to 6.
            lp (int, optional): Percussive filter size. Defaults to 6.
        """
        # get stft of x
        X = lb.core.stft(
            x, n_fft=self.L, hop_length=hop_length, window=self.window, center=False
        )

        # get magnitude of X
        Y = np.abs(X)

        # apply median filter to Y using JIT-compatible function
        Yh = medfilt(Y, (1, 2 * lh + 1))
        Yp = medfilt(Y, (2 * lp + 1, 1))

        # get harmonic and percussive masks
        Mh, Mp = create_harmonic_percussive_masks(Yh, Yp)

        # get harmonic and percussive components
        Xh, Xp = apply_masks_to_stft(X, Mh, Mp)

        # invert stft of Xh and Xp
        self.xh = invert_stft(Xh, hop_length, self.den, self.window, self.L)
        self.xp = invert_stft(Xp, hop_length, self.den, self.window, self.L)

        # save raw separation results to cache (before normalization and conversion)
        if self.save_cache:
            store_cache(self.cache_dir, self.cache_key, self.xh, "xh.pkl")
            store_cache(self.cache_dir, self.cache_key, self.xp, "xp.pkl")

    def update_stream(self, pos: float, stream: pyaudio.Stream):
        """
        Updates the stream with the current position and stream.

        Args:
            pos (int): The current position in the stream.
            stream (pyaudio.Stream): The stream to update.
        """
        start_t = time.perf_counter()

        # Update alpha from current alignment (keep fractional hop sizes)
        Ha = self.Hs / self.alpha
        Ha_ola = self.Hs_ola / self.alpha
        orch_out = []

        # Phase Vocoder processing
        # Round position to nearest sample for indexing
        offset_idx = int(np.rint(pos))
        if offset_idx + self.L > len(self.xh):
            # handle edge case where pos + self.L is out of bounds
            # we take the remaining samples from the end of the xh array and pad with zeros
            xh_segment = np.concatenate(
                (self.xh[offset_idx:], np.zeros(self.L - (len(self.xh) - offset_idx)))
            )
        else:
            xh_segment = self.xh[offset_idx : offset_idx + self.L]
        pv_frame_mod, self.prev_phase, self.prev_fft = compute_phase_vocoder_frame(
            xh_segment,
            self.window,
            self.omega_nom,
            self.prev_fft,
            self.prev_phase,
            Ha,
            self.sr,
            self.Hs,
        )

        # Update output buffer
        self.output_buffer = update_output_buffer(
            self.output_buffer, pv_frame_mod, self.window, self.den, self.Hs
        )

        # OLA processing
        ratio = self.Hs // self.Hs_ola
        ola_y = ola(ratio, self.L, pos, self.xp, self.L_ola, Ha_ola, self.Hs_ola)
        self.output_buffer += ola_y

        # Clip and output
        self.output_buffer = clip_output_buffer(self.output_buffer)

        orch_out = extract_output_chunk(self.output_buffer, self.Hs)

        elapsed_time = time.perf_counter() - start_t
        self.tsm_elapsed_t += elapsed_time

        return Ha, orch_out

    def __del__(self):
        """Cleanup method"""
        if hasattr(self, "stop_event"):
            self.stop_event.set()


def online_tsm(sr, audio_data, alphas, hop):
    """Performs online tsm on given audio data and provided alphas.

    Args:
        sr (int): audio frame rate
        audio_data (array-like): the audio to be time scale modified
        alphas (array-like): the alphas to be applied to the audio data. Should have one alpha for every TSM window
        hop (int): the hop size for TSM windows

    Returns:
        _type_: _description_
    """
    # TODO: allocate buffer based on maximum alpha
    max_alpha = 3  # TODO: change this!
    audio_modified = np.empty(len(audio_data) * max_alpha)

    # set up TSM module
    tsm = TSM(
        sr=sr,
        audio_data=audio_data,
        recording=True,
        playback=False,
    )

    # initalize variables
    out_pos = 0  # location in the output audio
    tsm_pos = 0.0  # location in the original audio (allow fractional samples)
    actual_alphas = []  # list of actual alphas applied
    actual_times = []  # list of actual reference times at which the alphas are applied

    while tsm_pos <= len(tsm.xh):
        # find query frame corresponding to reference frame and update alpha
        try:
            idx = int(tsm_pos / hop)
            tsm.alpha = alphas[idx]
        except IndexError:
            warnings.warn(
                f"IndexError: idx={idx} is greater than the length of alphas={len(alphas)}, using last alpha value"
            )
            tsm.alpha = alphas[-1]

        actual_alphas.append(tsm.alpha)
        actual_times.append(tsm_pos / sr)

        # perform TSM update
        Ha, tsm_out = tsm.update_stream(tsm_pos, stream=None)

        # modify output audio
        audio_size = len(tsm_out)
        audio_modified[out_pos : out_pos + audio_size] = tsm_out

        # advance time markings
        tsm_pos += Ha
        out_pos += audio_size

    # trim buffer
    audio_modified = audio_modified[:out_pos]

    return audio_modified, actual_alphas, actual_times
