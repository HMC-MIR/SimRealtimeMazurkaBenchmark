import logging
import os

import numpy as np
import pickle

def read_start_times(annotfile):
    start_times = []
    with open(annotfile, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('%'):
                continue
            fields = line.split()
            if len(fields) < 1:
                continue
            try:
                start_time = float(fields[0])
                start_times.append(start_time)
            except ValueError:
                continue
    return np.array(start_times)

def getGroundTruthTimestamps(query_annot_file, ref_annot_file):
    """
    Get the ground truth timestamps from the annotation files.
    """
    # Read annotation file, extract start times as numpy array
    gt_query = read_start_times(query_annot_file)
    gt_ref = read_start_times(ref_annot_file)
    gt = np.stack([gt_query, gt_ref], axis = 1)
    return gt


def eval_alignment_single(hypfile, query_annot_file, ref_annot_file, logger: logging.Logger = None):
    '''Returns the signed alignment error at each annotated beat, or None if the
    scenario could not be scored.'''
    # check if hypfile exists
    if not os.path.exists(hypfile):
        if logger:
            logger.warning(f'{hypfile} does not exist')
        else:
            print(f'{hypfile} does not exist')
        return None

    # check annotation files
    if not os.path.exists(query_annot_file):
        if logger:
            logger.warning(f'{query_annot_file} does not exist')
        else:
            print(f'{query_annot_file} does not exist')
        return None
        
    if not os.path.exists(ref_annot_file):
        if logger:
            logger.warning(f'{ref_annot_file} does not exist')
        else:
            print(f'{ref_annot_file} does not exist')
        return None

    gt = getGroundTruthTimestamps(query_annot_file, ref_annot_file)
    if gt.shape[0] == 0:
        if logger:
            logger.warning(f'No measures to evaluate in {hypfile}')
        else:
            print(f'No measures to evaluate in {hypfile}')
        return None
    
    hypalign = np.load(hypfile)
    
    try:
        pred = np.interp(gt[:,0], hypalign[0,:], hypalign[1,:])
        err = pred - gt[:,1]
        return err
    except Exception as e:
        if logger:
            logger.error(f"Error evaluating {hypfile}: {e}")
        else:
            print(f"Error evaluating {hypfile}: {e}")
        return None

def eval_alignment_batch(exp_dir, scenarios_dir, out_dir, logger: logging.Logger = None):
    # evaluate all scenarios
    d = {}
    
    if not os.path.exists(scenarios_dir):
        msg = f"Scenarios directory not found: {scenarios_dir}"
        if logger: logger.error(msg)
        else: print(msg)
        return

    scenario_ids = sorted(os.listdir(scenarios_dir))
    if logger:
        logger.info(f"evaluating {len(scenario_ids)} scenarios from {exp_dir}")
    
    success_count = 0
    fail_count = 0
    
    for scenario_id in scenario_ids:
        # Skip if not a directory
        if not os.path.isdir(os.path.join(scenarios_dir, scenario_id)):
            continue
            
        hypFile = f'{exp_dir}/{scenario_id}/hyp.npy'
            
        query_annot_file = f'{scenarios_dir}/{scenario_id}/query.beats'
        ref_annot_file = f'{scenarios_dir}/{scenario_id}/ref.beats'
        
        err = eval_alignment_single(hypFile, query_annot_file, ref_annot_file, logger=logger)
        
        if err is not None:
            d[scenario_id] = err
            success_count += 1
        else:
            fail_count += 1
    
    # save
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfile = f'{out_dir}/errs.pkl'
    pickle.dump(d, open(outfile, 'wb'))
    
    if logger:
        logger.info(f"Evaluation complete. Success: {success_count}, Failed: {fail_count}. Results saved to {outfile}")
    else:
        print(f"Evaluation complete. Success: {success_count}, Failed: {fail_count}. Results saved to {outfile}")