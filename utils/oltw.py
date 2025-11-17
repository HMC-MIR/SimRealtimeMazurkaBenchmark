import subprocess
import os
from shutil import which
import re
import numpy as np
import system_utils

def verify_oltw_installation(jar_path):
    '''Verifies that all tools needed to run OLTW are present
    
    Inputs
    jar_path: Path to the PerformanceMatcher.jar file
    '''
    # Prefer conda Java if available (usually has the right version)
    java_cmd = None
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_java = os.path.join(conda_prefix, 'bin', 'java')
        if os.path.exists(conda_java):
            java_cmd = conda_java
    
    # Fall back to system Java
    if java_cmd is None:
        java_cmd = which('java')
    
    assert java_cmd is not None, 'Java is not installed or not in PATH. Please install Java runtime environment.'
    assert os.path.exists(jar_path), f'PerformanceMatcher.jar not found at {jar_path}. Please ensure the JAR file is in the correct location.'
    
    # Check Java version - PerformanceMatcher.jar requires Java 21 (class file version 65.0)
    try:
        result = subprocess.run([java_cmd, '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT, 
                               timeout=5,
                               text=True)
        version_output = result.stdout + result.stderr if result.stderr else result.stdout
        
        # Check if Java 21 or higher
        import re
        version_match = re.search(r'version "(\d+)\.', version_output)
        if version_match:
            java_version = int(version_match.group(1))
            if java_version < 21:
                raise RuntimeError(
                    f'Java version {java_version} is too old. PerformanceMatcher.jar requires Java 21 or higher. '
                    f'Current Java: {java_cmd}\nVersion output: {version_output}'
                )
        else:
            print(f'Warning: Could not parse Java version. Output: {version_output}')
    except subprocess.TimeoutExpired:
        pass  # If it times out, that's okay
    except Exception as e:
        if 'too old' in str(e):
            raise
        print(f'Warning: Could not verify Java version: {e}')
    
    # Try to run java -jar to verify the JAR is valid
    try:
        result = subprocess.run([java_cmd, '-jar', jar_path], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               timeout=5)
    except subprocess.TimeoutExpired:
        pass  # If it runs, that's fine
    except Exception as e:
        raise RuntimeError(f'Could not run PerformanceMatcher.jar: {e}')
    
    return java_cmd

def parse_oltw_alignment(infile):
    '''
    Parses the OLTW alignment text output file.
    
    Inputs
    infile: filepath to the OLTW alignment text output file
    
    Returns a list of tuples (query_frame, ref_frame) in frame indices.
    OLTW outputs query frames as 1-based indices and reference frames as 0-based indices
    relative to the chopped reference audio.
    '''
    alignment_data = []
    with open(infile, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in lines:
        if line.startswith("ALIGNMENT"):
            parts = line.split(" ")
            if len(parts) >= 3:
                # Parse query index (remove trailing comma if present)
                query_idx = int(parts[1].rstrip(','))
                # Parse reference index
                ref_idx = int(parts[2])
                alignment_data.append((query_idx, ref_idx))
    
    return alignment_data

def online_processing(scenario_dir, out_dir, hop_length, jar_path=None):
    '''
    Carries out online processing using the OLTW algorithm.
    
    Inputs
    scenario_dir: The scenario directory to process
    out_dir: The directory to put results, intermediate files, and logging info
    hop_length: The hop length in samples (default 512 for OLTW)
    jar_path: Path to PerformanceMatcher.jar. If None, will look in match/PerformanceMatcher.jar

    This function will compute and save the predicted alignment in the output directory in a file hyp.npy
    The alignment is saved in seconds, with reference times relative to the full reference audio.
    '''
    
    # Set default jar path if not provided
    if jar_path is None:
        # Try to find the jar file relative to current directory
        possible_paths = [
            'match/PerformanceMatcher.jar',
            '../match/PerformanceMatcher.jar',
            os.path.join(os.path.dirname(os.getcwd()), 'match', 'PerformanceMatcher.jar')
        ]
        jar_path = None
        for path in possible_paths:
            if os.path.exists(path):
                jar_path = path
                break
        if jar_path is None:
            raise FileNotFoundError('Could not find PerformanceMatcher.jar. Please specify jar_path.')
    
    # Verify installation
    verify_oltw_installation(jar_path)
    
    # Set up file paths
    query_path = os.path.join(scenario_dir, 'query.wav')
    ref_path = os.path.join(scenario_dir, 'ref.wav')
    alignment_output_path = os.path.join(out_dir, 'oltw_alignment.txt')
    
    # Run OLTW algorithm
    cmd = [
        'java', '-jar', jar_path,
        '-b', '-q', '-G', '-D', '--use-chroma-map',
        query_path, ref_path
    ]
    
    with open(alignment_output_path, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    
    if result.returncode != 0:
        print(f'Warning: OLTW returned non-zero exit code. Check {alignment_output_path} for details.')
    
    # Parse alignment output
    alignment_data = parse_oltw_alignment(alignment_output_path)
    
    if len(alignment_data) == 0:
        raise RuntimeError(f'No alignment data found in {alignment_output_path}')
    
    # Convert frame indices to seconds and adjust reference offset
    hop_sec = hop_length / 22050.0  # Assuming 22050 Hz sample rate
    
    # Convert to seconds: query frames are 1-based, reference frames are 0-based relative to chopped audio
    # We subtract 1 from query frames to convert to 0-based, then multiply by hop_sec
    # For reference, we multiply by hop_sec to get time in full reference
    alignment_seconds = []
    for query_frame, ref_frame in alignment_data:
        query_sec = (query_frame - 1) * hop_sec  # Convert 1-based to 0-based, then to seconds
        ref_sec = ref_frame * hop_sec
        alignment_seconds.append((query_sec, ref_sec))
    
    # Convert to numpy array and transpose to match expected format (2 x N)
    alignment_array = np.array(alignment_seconds).T
    
    # Save alignment
    np.save(os.path.join(out_dir, 'hyp.npy'), alignment_array)
    
    # Clean up intermediate files
    if os.path.exists(alignment_output_path):
        os.remove(alignment_output_path)
    
    return
