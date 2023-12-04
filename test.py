import os

import numpy as np
import torch

import utils


def get_nmat_from_midi(fpath, tracks):
    music = utils.get_music(fpath)
    nmat = utils.get_note_matrix(music, tracks)
    return nmat


def get_nmats_from_dir(dir, tracks):
    nmats = []
    for subdir in os.scandir(dir):
        if subdir.is_dir():
            subnmats = []
            # phrase_config = utils.phrase_config_from_string(subdir.name)
            num_bar = 8
            for f in os.scandir(subdir.path):
                idx = int(f.name.split("-")[-1][:-4])
                fpath = f.path
                nmat = get_nmat_from_midi(fpath, tracks)
                subnmats.append((nmat, num_bar))
            nmats.append(subnmats)
    return nmats


if __name__ == "__main__":
    acc_prompt = np.load("./input/acc_prompt.npy")
    print(acc_prompt.shape)
    # (32, 128, 128)
    acc_prompt = np.reshape(acc_prompt, (32, 4, 32, 128))

    acc_nmats = get_nmats_from_dir("./input/acc", [3])
    print(len(acc_nmats), len(acc_nmats[0]))
    acc_prmat = torch.zeros([4, 32, 4, 32, 128])
    for sub, sub_nmats in enumerate(acc_nmats):
        # [32], nmat, num_bar
        sub_prmat = []
        for idx, idx_nmats in enumerate(sub_nmats):
            acc_prmat[sub, idx, :, :, :] = utils.nmat_to_prmat(
                idx_nmats[0], idx_nmats[1] // 2
            )
