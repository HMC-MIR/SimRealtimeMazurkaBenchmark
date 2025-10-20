import numpy as np
import system_utils
import pickle
import pandas as pd
import os

def parseAnnotationFile(annotfile):
    '''
    Parses a beat annotation file.
    
    Inputs
    annotfile: filepath of the beat annotation file
    
    Returns a dictionary whose key is the measure number and whose value is the corresponding timestamp.
    '''
    df = pd.read_csv(annotfile, sep=',')
    timestamps = np.array(df['start'])
    measure_nums = np.array(df['measure'])
    d = {}
    for (t, m) in zip(timestamps, measure_nums):
        d[m] = t
    return d

def getEvalMeasureSet(scenarioInfo):
    '''
    Gets the list of measure indices at which to evaluate.
    
    Inputs
    scenarioInfo: the scenario.info file for the scenario of interest
    
    Returns a sorted list of measure numbers.
    '''
    d = system_utils.get_scenario_info(scenarioInfo)
    o_basename = os.path.splitext(os.path.basename(d['o']))[0] # e.g. rach2_mov1_O1
    mov_id = '_'.join(o_basename.split('_')[0:2]) # e.g. rach2_mov1
    measureSet = system_utils.get_eval_measure_set(mov_id)
    return measureSet

def getGroundTruthTimestamps(annotfile1, annotfile2, scenarioInfo):
    '''
    Parses two beat annotation files and returns a list of the corresponding ground truth timestamps.
    
    Inputs
    annotfile1: the first beat annotation file
    annotfile2: the second beat annotation file
    scenarioInfo: path to the scenario.info file
    
    Outputs
    eval_pts: an Nx2 array of specifying the ground truth timestamps for N measures
    overlap_measures: an array containing the list of evaluated measures, sorted in increasing order
    '''
    
    # parse annotation files
    gt1 = parseAnnotationFile(annotfile1)
    gt2 = parseAnnotationFile(annotfile2)
   
    # measures to evaluate
    allEvalMeasures = getEvalMeasureSet(scenarioInfo)
    eval_measures = sorted(set(gt1).intersection(set(gt2)).intersection(allEvalMeasures))

    # construct (t1, t2) ground truth timestamps
    eval_pts = []
    for m in eval_measures:
        eval_pts.append((gt1[m], gt2[m]))
    
    return np.array(eval_pts), np.array(eval_measures)

def calcAlignErrors_single(hypfile, annotfile1, annotfile2, scenarioInfo, frames=False):
    '''
    Calculates the alignment errors for a single hypothesis file.
    
    Inputs
    hypfile: a .npy file containing the estimated alignment
    annotfile1: the beat annotation file for the piano recording
    annotfile2: the beat annotation file for the orchestra recording
    scenarioInfo: path to the scenario.info file
    
    Outputs
    err: the alignment errors in the estimated alignment
    measNums: the measure numbers that are evaluated
    '''
    gt, measNums = getGroundTruthTimestamps(annotfile1, annotfile2, scenarioInfo) # ground truth
    if gt.shape[0] == 0:
        print(f'No measures to evaluate in {hypfile}')
        return None, None
    # check if hypfile exists
    if not os.path.exists(hypfile):
        print(f'{hypfile} does not exist')
        return None, None
    hypalign = np.load(hypfile) # piano-orchestra predicted alignment in sec
    if frames:
        hypalign = hypalign / (22050/512)
    pred = np.interp(gt[:,0], hypalign[0,:], hypalign[1,:])
    err = pred - gt[:,1]
    return err, measNums

def getScenarioIds(scenarios_dir):
    '''
    Gets a list of scenario ids in a given scenarios/ directory.
    
    Inputs
    scenarios_dir: directory containing scenarios information
    
    Returns a list of scenario ids, sorted in increasing order.
    '''
    summary_file = f'{scenarios_dir}/scenarios.summary'
    d = system_utils.get_scenario_info(summary_file)
    return list(d.keys())

def calcAlignErrors_batch(exp_dir, scenarios_dir, out_dir, hypFileExt = '', tsm = False, lag = 0):
    '''
    Calculates the alignment errors for all scenarios in an experiment directory.
    
    Inputs
    exp_dir: the experiment directory to evaluate
    scenarios_dir: the directory containing the scenarios information
    out_dir: the directory to save outputs and figures to
    hypFileExt: a string specifying an extension to the hypothesis file name (used for evaluating
        systems at different iterations)
    tsm: if True, the hypothesis file is a TSM path file
    lag: lag to apply to the TSM path
    '''
    # evaluate all scenarios
    d = {}
    for scenario_id in getScenarioIds(scenarios_dir):
        if tsm:
            if lag == 0:
                hypFile = f'{exp_dir}/{scenario_id}/tsm{hypFileExt}.npy'
            else: # if lag is not 0, the hypothesis file is a TSM path file with the lag
                hypFile = f'{exp_dir}/{scenario_id}/tsm_lag{lag}{hypFileExt}.npy'
        else:
            hypFile = f'{exp_dir}/{scenario_id}/hyp{hypFileExt}.npy'
        pianoAnnot = f'{scenarios_dir}/{scenario_id}/p.beats'
        orchAnnot = f'{scenarios_dir}/{scenario_id}/o.beats'
        scenarioInfo = f'{scenarios_dir}/{scenario_id}/scenario.info'
        errs, measNums = calcAlignErrors_single(hypFile, pianoAnnot, orchAnnot, scenarioInfo)
        if errs is not None:
            d[scenario_id] = (errs, measNums) # key: scenario_id, value: (errors, measureNums)
        
    # save
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfile = f'{out_dir}/errs.pkl'
    pickle.dump(d, open(outfile, 'wb'))