import pretty_midi
import numpy as np
import sys
from eval_midi import prepare_quantization, bin_quantize


def get_onset_feature_vector(midi_path):
    # warning: all tracks are used in evaluation
    midi = pretty_midi.PrettyMIDI(midi_path)
    quantize_boundaries = prepare_quantization(midi)
    results = np.zeros(len(quantize_boundaries) + 1, dtype=np.int64)
    for ins in midi.instruments:
        for note in ins.notes:
            start = bin_quantize(note.start, quantize_boundaries)
            results[start] += 1

    return results


def compare_feature(est_feature, ref_feature, ord=2):
    if len(est_feature) != len(ref_feature):
        print(
            f"Warning: different length in subbeats for ref ({len(ref_feature)}) & est ({len(est_feature)}) midi"
        )
    # assuming 4/4 songs here, and 16 samples form a feature vector
    # pad to 16k samples for nearest integer k
    length = ((max(len(est_feature), len(ref_feature)) - 1) // 16 + 1) * 16
    est_feature = np.pad(est_feature, (0, length - len(est_feature)))
    ref_feature = np.pad(ref_feature, (0, length - len(ref_feature)))
    return np.linalg.norm(
        est_feature.reshape((-1, 16)) - ref_feature.reshape((-1, 16)), ord=ord, axis=1
    ).mean()


if __name__ == "__main__":
    # compare framewise onset count and perform average L2 distance

    est_feature = get_onset_feature_vector(sys.argv[1])
    ref_feature = get_onset_feature_vector(sys.argv[2])
    print("Average texture vector distance:", compare_feature(est_feature, ref_feature))
