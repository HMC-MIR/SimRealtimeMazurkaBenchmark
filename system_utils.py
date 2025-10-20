# %% [markdown]
# # System Utilities

# %% [markdown]
# This notebook contains various utility functions that are used frequently in the notebooks that implement alignment systems.

# %%
import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt

# %%
AUDIO_SUMMARY_FILE = 'cfg_files/AudioDataSummary.csv'
EVAL_MEASURES_FILE = 'annot/eval.measures'

# %%
def verify_scenario_dir(indir):
    '''
    Verifies that the specified scenario directory has the required files.
    
    Inputs
    indir: the scenario directory to verify
    '''
    assert os.path.exists(indir), f'{indir} does not exist'
    assert os.path.exists(f'{indir}/p.wav'), f'{indir}/p.wav does not exist'
    assert os.path.exists(f'{indir}/o.wav'), f'{indir}/o.wav does not exist'
    assert os.path.exists(f'{indir}/po.wav'), f'{indir}/po.wav does not exist'

# %%
def get_orchestra_start_end_times(scenario_dir):
    '''
    Returns the timestamps of when the orchestra begins playing and ends playing in the orchestra recording.
    This information is specified in the audio data summary file (only for orchestra recordings).
    
    Inputs
    scenario_dir: the scenario directory to process
    
    Outputs
    orchStart: the timestamp in the orchestra recording where the orchestra begins playing, specified in seconds
    orchEnd: the timestamp in the orchestra recording where the orchestra playing ends, specified in seconds
    '''
    
    # get orchestra recording basename
    scenario_info_file = f'{scenario_dir}/scenario.info'
    si = get_scenario_info(scenario_info_file)
    orch_id = os.path.splitext(os.path.basename(si['o']))[0] # e.g. rach2_mov1_O1
        
    # get start & end timestamps
    d = get_audio_summary_info()
    o_basename = f'{orch_id}.wav'
    if d[o_basename] is None:
        raise Exception(f'Global start and end timestamps not found for {o_basename}')
    orchStart, orchEnd = d[o_basename]
   
    return orchStart, orchEnd

# %%
def get_orchestra_query_boundaries(scenario_dir):
    '''
    Determines the start and end time in the orchestra recording where the query is located.
    
    Inputs
    scenario_dir: the directory containing the scenario information
    
    Returns the query start and end times in the orchestra recording, specified in seconds
    '''
    
    info_file = f'{scenario_dir}/scenario.info'
    assert os.path.exists(info_file)
    
    d = get_scenario_info(info_file)
    orch_start_sec = d['oStart']
    orch_end_sec = d['oEnd']
    
    return orch_start_sec, orch_end_sec

# %%
def get_piano_query_boundaries(scenario_dir):
    '''
    Determines the start and end time in the piano recording where the query is located.
    
    Inputs
    scenario_dir: the directory containing the scenario information
    
    Returns the query start and end times in the piano recording, specified in seconds
    '''
    
    info_file = f'{scenario_dir}/scenario.info'
    assert os.path.exists(info_file)
    
    d = get_scenario_info(info_file)
    p_start_sec = d['pStart']
    p_end_sec = d['pEnd']
    
    return p_start_sec, p_end_sec

# %%
def get_piano_reference_boundaries(scenario_dir):
    '''
    Determines the start and end time in the piano recording where the piano reference is located.
    
    Inputs
    scenario_dir: the directory containing the scenario information
    
    Returns the piano reference start and end times in the piano recording, specified in seconds
    '''
    
    info_file = f'{scenario_dir}/scenario.info'
    assert os.path.exists(info_file)
    
    d = get_scenario_info(info_file)
    p_start_sec = d['prefStart']
    p_end_sec = d['prefEnd']
    
    return p_start_sec, p_end_sec

# %%
def get_scenario_info(infile):
    '''
    Parses a scenarios summary (multiple) or info (single) file and returns the information as a dictionary.
    
    Inputs
    infile: filepath specifying the scenarios.summary or scenario.info file to parse
    
    If a summary file is specified, returns a nested dictionary whose primary key is the scenario 
    id (e.g. 's1'), and whose secondary key is one of the following:
    
    'p': filepath of piano only recording
    'o': filepath of orchestra only recording
    'po': filepath of the full mix recording
    'measStart': measure start
    'measEnd': measure end
    'pStart': timestamp of query start in P recording
    'pEnd': timestamp of query end in P recording
    'oStart': timestamp of query start in O recording
    'oEnd': timestamp of query end in O recording
    'prefStart': timestamp of piano reference start in P recording
    'prefEnd': timestamp of piano reference end in P recording
    If an info file is specified, returns a dictionary with the key-value pairs listed above, as well as:
    
    'scenario_id': the scenario id
    '''
    d = {}
    with open(infile) as f:
        for line in f:
            parts = line.strip().split()
            sid = parts[0]
            d[sid] = {}
            d[sid]['p'] = parts[1]
            d[sid]['o'] = parts[2]
            d[sid]['po'] = parts[3]
            d[sid]['measStart'] = int(parts[4])
            d[sid]['measEnd'] = int(parts[5])
            d[sid]['pStart'] = float(parts[6])
            d[sid]['pEnd'] = float(parts[7])
            d[sid]['oStart'] = float(parts[8])
            d[sid]['oEnd'] = float(parts[9])
            d[sid]['prefStart'] = float(parts[10])
            d[sid]['prefEnd'] = float(parts[11])
            
    if len(d) == 1: # info file
        d[sid]['scenario_id'] = sid
        return d[sid]
    else: # summary file
        return d

# %%
def get_audio_summary_info():
    '''
    Parses the information in AudioDataSummary.csv and returns the information in a dictionary.
    '''
    df = pd.read_csv(AUDIO_SUMMARY_FILE)
    d = {}
    for bname, t in zip(df['id'],df['timestamps']):
        if pd.isna(t):
            d[bname] = None
        else:
            parts = t.split('-') # e.g. '30.2-676.0'
            assert len(parts) == 2
            tStart, tEnd = float(parts[0]), float(parts[1])
            d[bname] = (tStart, tEnd)
    return d

# %%
def get_eval_measure_set(mov_id):
    '''
    Determines the set of the measures in the movement that will be evaluated.
    
    Inputs
    mov_id: id for the concerto movement of interest, e.g. rach2_mov1
    
    Returns a python set containing the measure indices that should be evaluated.
    '''
    d = {} # measure indices to evaluate
    with open(EVAL_MEASURES_FILE) as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == mov_id:
                for i in range(1, len(parts)):
                    startMeasure, endMeasure = parts[i].split('-')
                    startMeasure = int(startMeasure)
                    endMeasure = int(endMeasure)
                    for j in range(startMeasure, endMeasure+1):
                        d[j] = 1            
    return set(d.keys())

# %%
def visualize_alignments(system_names, scenario_id, alignment_type):
    '''Graphs the alignment between two files

    Inputs:
    alignment: a list of 2xN numpy array containing the alignment between two files
    labels: a list of strings containing the labels for each alignment
    '''

    alignment_0 = np.load(f'experiments/{system_names[0]}/{scenario_id}/{alignment_type}.npy')
    fig_ratio = alignment_0[0][-1] / alignment_0[1][-1]
    plt.figure(figsize=(10,10*fig_ratio))

    for system_name in system_names:
        alignment = filter_vertical_and_horizontal_segments(np.load(f'experiments/{system_name}/{scenario_id}/{alignment_type}.npy'))
        plt.plot(alignment[1], alignment[0])
    plt.xlabel('Source Time')
    plt.ylabel('Aligned Time')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend(system_names)

# %%
def filter_vertical_and_horizontal_segments(align):
    '''
    Filters out vertical and horizontal segments from an alignment.

    Inputs
    align: a 2xN numpy array containing the alignment between two files
    
    Returns a 2xM numpy array containing the filtered alignment between two files
    '''
    new_align = []

    for i in range(0, align.shape[1]-1):
        x = align[0, i]
        y = align[1, i]
        x_next = align[0, i+1]
        y_next = align[1, i+1]

        # new_align.append(dydx)

        if x != x_next and y != y_next:
            new_align.append([x, y])
    new_align.append([align[0, -1], align[1, -1]])

    return np.array(new_align).T


