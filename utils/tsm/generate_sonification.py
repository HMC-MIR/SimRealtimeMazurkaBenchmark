from tqdm.notebook import tqdm
import numpy as np
import librosa as lb
import soundfile as sf
from hmc_mir import tsm_tools
from eval_tools import getScenarioIds
from system_utils import get_piano_reference_boundaries
import utils.constants as constants

def getModRefVQuery(EXP_ROOT_DIR, modes, mode_id, scenarios_dir):
    '''Inputs: 
    EXP_ROOT_DIR: directory of the experiment
    mode_id: id of the mode
    scenarios_dir: directory of the scenarios
    Outputs:
    Saves the stereo mix of the modified reference and the query audio in 
    the directory of the experiment based on mode and system
    '''
    for scenario_id in tqdm(getScenarioIds(scenarios_dir)):
        try:
            alignment = np.load(f'{EXP_ROOT_DIR}/{scenario_id}/tsm.npy')
            alignment = np.flipud(alignment)
            alignment[0] -= alignment[0][0]
            p_ref = f'scenarios/{modes[mode_id]}/{scenario_id}/pref.wav'
            y, _ = lb.load(p_ref, sr=constants.DEFAULT_SR)
            start_time, end_time = get_piano_reference_boundaries(f'scenarios/continuous/{scenario_id}')
            y_mod = tsm_tools.tsmvar_hybrid(y[int(start_time*constants.DEFAULT_SR):int(end_time*constants.DEFAULT_SR)], alignment, constants.DEFAULT_SR)
            
            # Create stereo mix: left channel = y_mod, right channel = piano audio
            p_piano = f'scenarios/{modes[mode_id]}/{scenario_id}/p.wav'
            y_piano, _ = lb.load(p_piano, sr=constants.DEFAULT_SR)
            # Make sure both channels have the same length
            min_length = min(len(y_mod), len(y_piano))
            y_mod_trimmed = y_mod[:min_length]
            y_piano_trimmed = y_piano[:min_length]

            # Create stereo array (left = y_mod, right = y_piano)
            stereo_mix = np.array([y_mod_trimmed, y_piano_trimmed]).T

            # Save the stereo mix
            output_file = f'{EXP_ROOT_DIR}/{scenario_id}/y_modvsp.wav'
            sf.write(output_file, stereo_mix, constants.DEFAULT_SR, subtype='PCM_16')
        except FileNotFoundError:
            print(f'{scenario_id} does not exist')
            continue
    print('Stereo mixes of ref_mod and query completed and saved!')