from tqdm.notebook import tqdm
import numpy as np
import librosa as lb
import soundfile as sf
from hmc_mir import tsm_tools
# from eval_tools import getScenarioIds
from system_utils import get_piano_reference_boundaries
import utils.constants as constants
import os

def getModRefVQuery(EXP_ROOT_DIR, scenarios_dir, lag=0):
    '''Inputs: 
    EXP_ROOT_DIR: directory of the experiment
    mode_id: id of the mode
    scenarios_dir: directory of the scenarios
    Outputs:
    Saves the stereo mix of the modified reference and the query audio in 
    the directory of the experiment based on mode and system
    '''
    scenarios = os.listdir(scenarios_dir)
    for scenario_id in tqdm(scenarios):
        try:
            if lag == 0:
                alignment = np.load(f'{EXP_ROOT_DIR}/{scenario_id}/tsm.npy')
            else:
                alignment = np.load(f'{EXP_ROOT_DIR}/{scenario_id}/tsm_lag{lag}.npy')
            alignment = np.flipud(alignment)
            alignment[0] -= alignment[0][0]
            p_ref = f'scenarios/{scenario_id}/ref.wav'
            y, _ = lb.load(p_ref, sr=constants.DEFAULT_SR)
            # start_time, end_time = get_piano_reference_boundaries(f'scenarios/continuous/{scenario_id}')
            y_mod = tsm_tools.tsmvar_hybrid(y, alignment, constants.DEFAULT_SR)
            
            # Create stereo mix: left channel = y_mod, right channel = piano audio
            p_piano = f'scenarios/{scenario_id}/query.wav'
            y_piano, _ = lb.load(p_piano, sr=constants.DEFAULT_SR)
            # Make sure both channels have the same length
            min_length = min(len(y_mod), len(y_piano))
            y_mod_trimmed = y_mod[:min_length]
            y_piano_trimmed = y_piano[:min_length]

            # Create stereo array (left = y_mod, right = y_piano)
            stereo_mix = np.array([y_mod_trimmed, y_piano_trimmed]).T

            # Save the stereo mix
            if lag==0:
                output_file = f'{EXP_ROOT_DIR}/{scenario_id}/y_modvsp.wav'
            else:
                output_file = f'{EXP_ROOT_DIR}/{scenario_id}/y_modvsp_lag{lag}.wav'
            sf.write(output_file, stereo_mix, constants.DEFAULT_SR, subtype='PCM_16')
        except FileNotFoundError:
            print(f'{scenario_id} does not exist')
            continue
    print('Stereo mixes of ref_mod and query completed and saved!')