import os
import subprocess

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matchmaker_worker.py')
DEFAULT_ENV_NAME = 'matchmaker'


def verify_matchmaker_installation(python_path=None):
    '''
    Verifies that the matchmaker conda environment is available.

    Inputs
    python_path: path to the python interpreter of the matchmaker env. If None, uses
                 $MATCHMAKER_PYTHON, then a sibling env of the active conda env.

    Returns the path to the python interpreter.
    '''
    if python_path is None:
        python_path = os.environ.get('MATCHMAKER_PYTHON')

    if python_path is None:
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            candidate = os.path.join(os.path.dirname(conda_prefix), DEFAULT_ENV_NAME, 'bin', 'python')
            if os.path.exists(candidate):
                python_path = candidate

    if python_path is None or not os.path.exists(python_path):
        raise RuntimeError(
            f'Could not locate the {DEFAULT_ENV_NAME} environment. '
            'Set MATCHMAKER_PYTHON to its python interpreter.'
        )
    if not os.path.exists(WORKER_PATH):
        raise RuntimeError(f'Worker script not found at {WORKER_PATH}')

    return python_path


def run_matchmaker_alignment(ref_feat_path, query_feat_path, out_file, method, sr, hop_length,
                             window_size, distance_metric, step_size=None, ref_start_sec=0.0,
                             python_path=None):
    '''
    Runs a MatchMaker OLTW baseline in the matchmaker conda env and saves the alignment.

    Inputs
    ref_feat_path: filepath to the reference feature .npy file
    query_feat_path: filepath to the query feature .npy file
    out_file: filepath where the estimated alignment (2xN, in seconds) is saved
    method: 'dixon' or 'arzt'
    sr: sample rate
    hop_length: hop length in samples
    window_size: search window in seconds
    distance_metric: 'cosine' or 'euclidean'
    step_size: max reference frames advanced per query frame (arzt only)
    ref_start_sec: offset to chop off the reference and add back to the result
    python_path: path to the matchmaker env python interpreter
    '''
    python_cmd = verify_matchmaker_installation(python_path)

    cmd = [
        python_cmd, WORKER_PATH,
        '--ref-feat', ref_feat_path,
        '--query-feat', query_feat_path,
        '--out', out_file,
        '--method', method,
        '--sr', str(sr),
        '--hop-length', str(hop_length),
        '--window-size', str(window_size),
        '--distance-metric', distance_metric,
        '--ref-start-sec', str(ref_start_sec),
    ]
    if step_size is not None:
        cmd += ['--step-size', str(step_size)]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'MatchMaker worker failed for {query_feat_path}:\n{result.stderr}')
