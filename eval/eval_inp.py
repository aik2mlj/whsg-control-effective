import pretty_midi
import numpy as np


def prepare_quantization(midi, beat_div=4):
    # get decision boundaries for quantization
    # beat_div: the subdivision of beat intervals
    # beat_div = 4 means a 16-note quantization assuming a 4/4 meter (4 beats/bar * 4 subbeats/beat)
    beat = midi.get_beats()
    subbeat = np.interp(
        np.arange(len(beat) * beat_div) / beat_div, np.arange(len(beat)), beat
    )
    return (subbeat[1:] + subbeat[:-1]) / 2


def bin_quantize(time, boundaries):
    # quantize a time step according to decision boundaries
    return np.searchsorted(boundaries, time)


def get_statistics(midi_paths):
    # warning: all tracks are used in evaluation
    if not isinstance(midi_paths, list) and not isinstance(midi_paths, tuple):
        midi_paths = [midi_paths]
    pitch_freq = {}
    duration_freq = {}
    for midi_path in midi_paths:
        midi = pretty_midi.PrettyMIDI(midi_path)
        quantize_boundaries = prepare_quantization(midi)
        for ins in midi.instruments:
            for note in ins.notes:
                pitch = note.pitch
                duration = bin_quantize(note.end, quantize_boundaries) - bin_quantize(
                    note.start, quantize_boundaries
                )
                if duration <= 0:
                    duration = 1
                if pitch not in pitch_freq:
                    pitch_freq[pitch] = 1
                else:
                    pitch_freq[pitch] += 1
                if duration not in duration_freq:
                    duration_freq[duration] = 1
                else:
                    duration_freq[duration] += 1
    return {"pitch_freq": pitch_freq, "duration_freq": duration_freq}


def get_freq_normalizer(freq):
    normalizer = 0
    for key in freq:
        normalizer += freq[key]
    return normalizer


def overlapped_area(est_freq, ref_freq):
    normalizer_est_freq = get_freq_normalizer(est_freq)
    normalizer_ref_freq = get_freq_normalizer(ref_freq)
    area = 0.0
    for key in est_freq:
        if key in ref_freq:
            area += min(
                est_freq[key] / normalizer_est_freq, ref_freq[key] / normalizer_ref_freq
            )
    return area


def evaluate_statistics(est_statistics, ref_statistics):
    return {
        "pitch similarity": overlapped_area(
            est_statistics["pitch_freq"], ref_statistics["pitch_freq"]
        ),
        "duration similarity": overlapped_area(
            est_statistics["duration_freq"], ref_statistics["duration_freq"]
        ),
    }


if __name__ == "__main__":
    est_statistics = get_statistics(R"D:\workplace\eval\data\uncond.mid")
    ref_statistics = get_statistics(
        [R"D:\Dataset\POP909\001\001.mid", R"D:\Dataset\POP909\002\002.mid"]
    )
    print(evaluate_statistics(est_statistics, ref_statistics))
