"""
Worker for the MatchMaker OLTW baselines. Runs inside the `matchmaker` conda env.

pymatchmaker pins numpy<2, so it cannot share a process with the benchmark env.
This script is invoked as a subprocess and communicates only through .npy files.
Nothing in this directory may be named matchmaker.py: it would shadow the installed
package, since the worker's own directory is first on sys.path.

MatchMaker's frame-level followers are score followers, but their position axis is
just an array of floats. Setting it to reference timestamps makes them report
reference seconds, which is what the benchmark's hyp.npy expects.
"""

import argparse
import importlib.metadata

import numpy as np
from matchmaker.dp import OnlineTimeWarpingArztFrame, OnlineTimeWarpingDixonFrame

# Row order of alignment_path flipped between 0.3.0 and upstream main.
EXPECTED_VERSION = '0.3.0'

FOLLOWERS = {'dixon': OnlineTimeWarpingDixonFrame, 'arzt': OnlineTimeWarpingArztFrame}


def verify_version():
    """Checks that the installed pymatchmaker matches the version this worker assumes."""
    version = importlib.metadata.version('pymatchmaker')
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f'Expected pymatchmaker {EXPECTED_VERSION}, found {version}. '
            'The alignment_path row order is version dependent; re-verify before changing this.'
        )
    return version


def align(ref_feat, query_feat, method, frame_rate, window_size, step_size, distance_metric):
    """
    Aligns query features against reference features using a MatchMaker OLTW follower.

    Inputs
    ref_feat: reference features, shape D x N
    query_feat: query features, shape D x M
    method: 'dixon' or 'arzt'
    frame_rate: feature frames per second
    window_size: search window in seconds
    step_size: max reference frames advanced per query frame (arzt only)
    distance_metric: 'cosine' or 'euclidean'

    Returns a 2xN array indicating the estimated alignment in seconds.
    """
    ref = np.ascontiguousarray(ref_feat.T, dtype=np.float32)
    query = np.ascontiguousarray(query_feat.T, dtype=np.float32)

    # Position axis is reference time, so get_current_position() returns seconds.
    ref_secs = np.arange(ref.shape[0]) / frame_rate

    kwargs = {
        'reference_features': ref,
        'score_positions': ref_secs,
        'ref_frame_to_beat': ref_secs,
        'frame_rate': frame_rate,
        'window_size': window_size,
        # dixon passes the metric to scipy, arzt to its own cythonized metrics
        'distance_func': distance_metric if method == 'dixon' else distance_metric.capitalize(),
    }
    if method == 'arzt':
        kwargs['step_size'] = step_size

    follower = FOLLOWERS[method](**kwargs)
    for t in range(query.shape[0]):
        follower(query[t], t / frame_rate)
        if not follower.is_still_following():
            break

    # pymatchmaker 0.3.0 returns (reference, query); the benchmark expects (query, reference)
    wp = np.asarray(follower.alignment_path, dtype=float)[::-1]

    query_dur = (query.shape[0] - 1) / frame_rate
    if wp.shape[1] > 0 and wp[0].max() > query_dur + 1.0:
        raise RuntimeError('Query times exceed the query duration; check the alignment_path row order.')

    return wp


def main():
    parser = argparse.ArgumentParser(description='Run a MatchMaker OLTW baseline on precomputed features.')
    parser.add_argument('--ref-feat', required=True, help='reference feature .npy file (D x N)')
    parser.add_argument('--query-feat', required=True, help='query feature .npy file (D x M)')
    parser.add_argument('--out', required=True, help='output hyp.npy file')
    parser.add_argument('--method', required=True, choices=sorted(FOLLOWERS))
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--hop-length', type=int, default=512)
    parser.add_argument('--window-size', type=float, default=10.0, help='search window in seconds')
    parser.add_argument('--step-size', type=int, default=3, help='arzt only')
    parser.add_argument('--distance-metric', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--ref-start-sec', type=float, default=0.0,
                        help='chop this much off the front of the reference and add it back to the result')
    args = parser.parse_args()

    verify_version()

    hop_sec = args.hop_length / args.sr
    ref_feat = np.load(args.ref_feat)[:, int(args.ref_start_sec / hop_sec):]
    query_feat = np.load(args.query_feat)

    wp = align(ref_feat, query_feat, args.method, args.sr / args.hop_length,
               args.window_size, args.step_size, args.distance_metric)
    wp[1, :] += args.ref_start_sec

    np.save(args.out, wp)


if __name__ == '__main__':
    main()
