import subprocess
import pickle
import tempfile
import os
import numpy as np

def extract_match_features(wav_path, output_path=None):
    """
    Extracts features from an audio file using the MATCH Java tool.

    Args:
        wav_path (str): Path to the input WAV file.
        output_path (str, optional): Path to save the extracted features.
    Returns:
        list: Extracted features as strings.
    """
    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as tmp_file:
        temp_output = tmp_file.name

    cmd = [
        "xvfb-run",
        "java",
        "-cp",
        "match_feat_src",
        "at.ofai.music.match.PerformanceMatcher",
        "-b", "-q", "-D",
        wav_path,
        wav_path  # Dummy second argument
    ]

    # Run the command and redirect stdout to the temp file
    with open(temp_output, 'w') as out_f:
        subprocess.run(cmd, stdout=out_f, check=True)

    # Read and process the output as floats
    with open(temp_output, 'r') as f:
        lines = f.readlines()
        
        # the features are in the even lines
        # each line is a string of comma-separated floats
        # convert the string to a list of floats
        # note that the last character of each line is a comma and a newline character
        features = []
        for i, line in enumerate(lines):
            if i % 2 == 0:
                features.append(list(map(float, line.strip()[:-1].split(","))))
        
    # reshape the features into a 2D array
    features = np.array(features).T

    # Optionally save to pickle
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)

    # Clean up temp file
    os.remove(temp_output)

    return features