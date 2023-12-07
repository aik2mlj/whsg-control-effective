from mir import DataEntry
from mir import io
from extractors.midi_utilities import is_percussive_channel, MidiBeatExtractor
from extractors.rule_based_channel_reweight import midi_to_thickness_and_bass_weights
from midi_chord import ChordRecognition
from chord_class import ChordClass
import mir_eval
import numpy as np
import sys


def process_chord(entry, extra_division):
    """

    Parameters
    ----------
    entry: the song to be processed. Properties required:
        entry.midi: the pretty midi object
        entry.beat: extracted beat and downbeat
    extra_division: extra divisions to each beat.
        For chord recognition on beat-level, use extra_division=1
        For chord recognition on half-beat-level, use extra_division=2

    Returns
    -------
    Extracted chord sequence
    """

    midi = entry.midi
    beats = midi.get_beats()
    if extra_division > 1:
        beat_interp = np.linspace(beats[:-1], beats[1:], extra_division + 1).T
        last_beat = beat_interp[-1, -1]
        beats = np.append(beat_interp[:, :-1].reshape((-1)), last_beat)
    downbeats = midi.get_downbeats()
    j = 0
    beat_pos = -2
    beat = []
    for i in range(len(beats)):
        if j < len(downbeats) and beats[i] == downbeats[j]:
            beat_pos = 1
            j += 1
        else:
            beat_pos = beat_pos + 1
        assert beat_pos > 0
        beat.append([beats[i], beat_pos])
    rec = ChordRecognition(entry, ChordClass())
    weights = midi_to_thickness_and_bass_weights(entry.midi)
    channel_names = [
        ins.name for ins in midi.instruments if not is_percussive_channel(ins)
    ]
    print("The name and weight of each channel is:")
    for i in range(len(channel_names)):
        print("%d | %.6f | %s" % (i, weights[i], channel_names[i]))
    rec.process_feature(weights)
    chord = rec.decode(raw_timesteps=True)
    return chord


def eval_chord(midi_path, gt_npy_path, norm_ord=2):
    """
    Perform chord evaluation
    :param midi_path: the path to the midi file
    :param gt_npy_path: the path to the numpy 36-D array file
    :param norm_ord: the number p for L-p norm
    """
    gt_arr = np.load(gt_npy_path)
    assert gt_arr.shape[-1] == 36, "chord data should be 36-D"
    gt_arr = gt_arr.reshape(-1, gt_arr.shape[-1])
    est_arr = np.zeros_like(gt_arr)
    entry = DataEntry()
    entry.append_file(midi_path, io.MidiIO, "midi")
    entry.append_extractor(MidiBeatExtractor, "beat")
    results = process_chord(entry, extra_division=1)
    for [s, t, chord] in results:
        chord_root, chord_chroma, chord_bass = mir_eval.chord.encode(chord)
        if chord_root < 0 or chord_root >= 12:  # N or X
            continue
        est_arr[s : t + 1, chord_root] = 1
        est_arr[s : t + 1, 12:24] = np.roll(chord_chroma, chord_root)[None]
        est_arr[s : t + 1, 24 + chord_bass] = 1
    return np.linalg.norm(est_arr - gt_arr, ord=norm_ord, axis=-1).mean()


if __name__ == "__main__":
    norm_ord = 2
    # midi = "/home/aik2/Learn/ComputerMusic/Models/06_transformer_arrangement_ziyu/ismir_demo/polydis_tfm_r/chd_cond.mid"
    # chd = "/home/aik2/Learn/ComputerMusic/Models/06_transformer_arrangement_ziyu/ismir_demo/polydis_tfm_r/chd.npy"
    # midi = "/home/aik2/Learn/ComputerMusic/Models/06_transformer_arrangement_ziyu/ismir_demo/polydis_sample/chd_cond.mid"
    # chd = "/home/aik2/Learn/ComputerMusic/Models/06_transformer_arrangement_ziyu/ismir_demo/polydis_sample/chd.npy"
    midi = sys.argv[1]
    chd = sys.argv[2]
    print(
        f"Average chord vector distance (L{norm_ord} norm): ",
        # eval_chord('exp/polyf_wm/chd_cond.mid', R'exp/polyf_wm/chd.npy', norm_ord)
        eval_chord(midi, chd, norm_ord),
    )
