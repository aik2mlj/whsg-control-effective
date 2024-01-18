"""
This is the main script that computes the latent similarity of the given controls
and the generated outputs, hence evaluating the control effectiveness.

Generated output & Control types:
`--type=acc`: Accompaniment conditioned on given texture (see Polydis).
`--type=mel`: Melody conditioned on given rhythm (see EC2VAE).
`--type=chd/chd8`: Chords conditioned on given chord (see Polydis).

"""


import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

import utils
from chd8bar.model_chd_8bar import load_pretrained_chd_enc_dec
from ec2vae.model import EC2VAE
from poly_dis.model import PolyDisVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

ec2vae_model = EC2VAE.init_model()
ec2vae_param_path = "./ec2vae/model_param/ec2vae-v1.pt"
ec2vae_model.load_model(ec2vae_param_path)
print("ec2vae loaded")

polydis_model = PolyDisVAE.init_model()
polydis_param_path = "./poly_dis/model_param/polydis-v1.pt"
polydis_model.load_model(polydis_param_path)
print("polydis loaded")

chd8_params = {
    "input_dim": 36,
    "z_input_dim": 512,
    "hidden_dim": 512,
    "z_dim": 512,
    "n_step": 32,
}
chd8_enc, chd8_dec = load_pretrained_chd_enc_dec("./chd8bar/weights.pt", **chd8_params)
chd8_enc, chd8_dec = chd8_enc.to(device), chd8_dec.to(device)
print("chd8bar loaded")


def record_latent_similarity(z1, z2, fpath, name):
    if z2.dim() > z1.dim():
        z1_expand = z1.unsqueeze(0).expand_as(z2)
    else:
        z1_expand = z1
    print(z1.shape, z1_expand.shape, z2.shape)
    sim = F.cosine_similarity(z1_expand, z2, dim=-1)
    mean = sim.mean()
    std = sim.std()
    with open(fpath, "a") as f:
        f.write(f"\n{name}: {mean:.4f}, {std:.4f}")


def get_nmat_from_midi(fpath, tracks):
    music = utils.get_music(fpath)
    nmat = utils.get_note_matrix(music, tracks)
    return nmat


def get_nmats_from_dir(dir, tracks):
    nmats = []
    for subdir in sorted(os.scandir(dir), key=lambda x: x.name):
        if subdir.is_dir():
            print(subdir.name)
            subnmats = []
            # phrase_config = utils.phrase_config_from_string(subdir.name)
            num_bar = 8
            idx_list = []
            for f in os.scandir(subdir.path):
                idx = int(f.name.split("-")[-1][:-4])
                idx_list.append(idx)
                fpath = f.path
                nmat = get_nmat_from_midi(fpath, tracks)
                subnmats.append((nmat, num_bar))
            subnmats, idx_list = zip(
                *sorted(zip(subnmats, idx_list), key=lambda x: x[1])
            )
            print(idx_list)
            nmats.append(subnmats)
    return nmats


def get_resampled_zchd8(zchd, noise_scale=0.0):
    noise = torch.normal(0.0, 1.0, zchd.shape).to(device)
    zchd_noised = noise * noise_scale + zchd * (1.0 - noise_scale)
    chd = chd8_dec.forward(zchd_noised.reshape(4 * 128, 512), True, 0.0, None)
    utils.chd8_to_midi_file(chd, "./chd8_resampled.mid")

    nmat_rl = get_nmat_from_midi("./chd8_resampled.mid", [0])
    num_bar = 128 * 4 * 8
    chdprmat_rl = utils.nmat_to_chd8(nmat_rl, num_bar // 8, n_beat=32).to(device)
    np.save("./chd8_resampled.npy", chdprmat_rl.cpu().numpy())
    # utils.chd8_to_midi_file(chdprmat_rl, "./chd8_resampled_writeback.mid")

    zchd_resampled = chd8_enc.forward(chdprmat_rl)
    # chd_nn = chd8_dec.forward(zchd_resampled, True, 0.0, None)
    # utils.chd8_to_midi_file(chd_nn, "./chd8_reresampled.mid")
    return zchd_resampled.reshape(4, 128, 512)


def get_resampled_zchd(zchd, noise_scale=0.0):
    noise = torch.normal(0.0, 1.0, zchd.shape).to(device)
    zchd_noised = noise * noise_scale + zchd * (1.0 - noise_scale)
    chd = polydis_model.chd_decode(zchd_noised.reshape(4 * 512, 256))
    utils.chd8_to_midi_file(chd, "./chd_resampled.mid")

    nmat_rl = get_nmat_from_midi("./chd_resampled.mid", [0])
    num_bar = 128 * 4 * 8
    chdprmat_rl = utils.nmat_to_chd8(nmat_rl, num_bar // 2, n_beat=8).to(device)
    # utils.chd8_to_midi_file(chdprmat_rl, "./chd_resampled_writeback.mid")

    zchd_resampled = polydis_model.chd_encode(chdprmat_rl)
    # chd_nn = polydis_model.chd_decode(zchd_resampled)
    # utils.chd8_to_midi_file(chd_nn, "./chd_reresampled.mid")
    return zchd_resampled.reshape(4, 512, 256)


def get_resampled_zr(zr, chd, zp=None):
    zr_exp = zr.unsqueeze(0).expand(4, -1, -1).reshape(4 * 128, -1)
    chd_exp = chd.unsqueeze(0).expand(4, -1, -1, -1).reshape(4 * 128, 32, 12)
    if zp is None:
        zp = torch.normal(0.0, 1.0, zr_exp.shape).to(device)
    else:
        print("zp given")
    mel_new = ec2vae_model.decoder(zp, zr_exp, chd_exp).to(device)
    mel_new_np = mel_new.squeeze(0).cpu().numpy()
    utils.melprmat_to_midi_file(mel_new_np, "mel_resampled.mid")

    # reload melprmat from tmp midi file
    nmat_rl = get_nmat_from_midi("./mel_resampled.mid", [0])
    num_bar = 128 * 4 * 2
    melprmat_rl = utils.nmat_to_melprmat(nmat_rl, num_bar // 2, n_step=32).to(device)
    # utils.melprmat_to_midi_file(melprmat_rl, "reloaded_mel.mid")

    zr_resampled = ec2vae_model.encoder(melprmat_rl, chd_exp)[1].reshape(4, 128, -1)

    # test reresampled
    # mel_nn = ec2vae_model.decoder(zp, zr_resampled, chd)
    # utils.melprmat_to_midi_file(mel_nn.cpu().numpy(), "reload_reresampled_zr.mid")

    return zr_resampled


def get_resampled_ztxt(ztxt):
    ztxt_expand = ztxt.unsqueeze(0).expand(4, -1, -1).reshape(4 * 128, -1)
    zchd = torch.normal(0.0, 1.0, ztxt_expand.shape).to(device)
    acc = polydis_model.pnotree_decode(zchd, ztxt_expand)
    print(acc.shape)
    acc_prmat = polydis_model.pnotree_to_prmat(acc).to(device)
    utils.prmat_to_midi_file(acc_prmat, "./acc_resampled.mid")
    ztxt_resampled = polydis_model.txt_encode(acc_prmat).reshape(4, 128, -1)
    return ztxt_resampled


def get_acc_input(dir, tracks):
    acc_nmats = get_nmats_from_dir(dir, tracks)
    print(len(acc_nmats), len(acc_nmats[0]))
    acc_prmat = torch.zeros([4, 32, 4, 32, 128])
    for sub, sub_nmats in enumerate(acc_nmats):
        # [32], nmat, num_bar
        for idx, idx_nmats in enumerate(sub_nmats):
            acc_prmat[sub, idx, :, :, :] = utils.nmat_to_prmat(
                idx_nmats[0], idx_nmats[1] // 2
            )
    acc_prmat = acc_prmat.reshape([4, 128, 32, 128])
    utils.prmat_to_midi_file(acc_prmat.reshape(128 * 4, 32, 128), f"{dir}/acc_test.mid")
    acc_prmat = acc_prmat.to(device)
    return acc_prmat


def get_mel_input(dir, tracks):
    melchd_nmats = get_nmats_from_dir(dir, tracks)
    print(len(melchd_nmats), len(melchd_nmats[0]))
    mel_prmat = torch.zeros([4, 32, 4, 32, 130])
    chd_prmat = torch.zeros([4, 32, 4, 32, 12])
    for sub, sub_nmats in enumerate(melchd_nmats):
        # [32], nmat, num_bar
        for idx, idx_nmats in enumerate(sub_nmats):
            nmat = idx_nmats[0]
            num_bar = idx_nmats[1]
            melnmat = [x for x in nmat if x[1] >= 60]
            chdnmat = [x for x in nmat if x[1] < 60]
            mel_prmat[sub, idx, :, :, :] = utils.nmat_to_melprmat(
                melnmat, num_bar // 2, n_step=32
            )
            chd_prmat[sub, idx, :, :, :] = utils.nmat_to_chd_ec2vae(
                chdnmat, num_bar // 2, n_step=32
            )
    mel_prmat = mel_prmat.reshape([4, 128, 32, 130])
    chd_prmat = chd_prmat.reshape([4, 128, 32, 12])
    utils.melprmat_to_midi_file(
        mel_prmat.reshape(128 * 4, 32, 130), f"{dir}/mel_test.mid"
    )
    utils.chd_ec2vae_to_midi_file(
        chd_prmat.reshape(128 * 4, 32, 12), f"{dir}/chd_test.mid"
    )
    mel_prmat = mel_prmat.to(device)
    chd_prmat = chd_prmat.to(device)
    return mel_prmat, chd_prmat


def get_chd8_input(dir, tracks):
    melchd_nmats = get_nmats_from_dir(dir, tracks)
    print(len(melchd_nmats), len(melchd_nmats[0]))
    chd_prmat = torch.zeros([4, 32, 4, 32, 36])
    for sub, sub_nmats in enumerate(melchd_nmats):
        # [32], nmat, num_bar
        for idx, idx_nmats in enumerate(sub_nmats):
            nmat = idx_nmats[0]
            num_bar = 32
            chdnmat = [x for x in nmat if x[1] < 48]
            chd_prmat[sub, idx, :, :, :] = utils.nmat_to_chd8(
                chdnmat, num_bar // 8, n_beat=32
            )
    np.save(f"{dir}/chord_test.npy", chd_prmat.reshape([4, 32, 128, 36]).cpu().numpy())
    chd_prmat = chd_prmat.reshape([4, 128, 32, 36])
    utils.chd8_to_midi_file(chd_prmat.reshape(128 * 4, 32, 36), f"{dir}/chd_test.mid")
    chd_prmat = chd_prmat.to(device)
    return chd_prmat


def compute_acc():
    acc_prompt = np.load("./input/acc_prompt.npy")
    print(acc_prompt.shape)
    # (32, 128, 128)
    acc_expand = (
        np.expand_dims(acc_prompt, axis=0).repeat(4, axis=0).reshape((512, 32, 128))
    )
    # print("expand:", acc_expand.shape)
    utils.prmat_to_midi_file(acc_expand, "./input/acc_prompt_expand.mid")
    acc_prompt = np.reshape(acc_prompt, (128, 32, 128))
    acc_prompt = torch.from_numpy(acc_prompt).to(device)
    z_org = polydis_model.txt_encode(acc_prompt)
    print(z_org.shape)
    z_resampled = get_resampled_ztxt(z_org)
    print(z_resampled.shape)

    acc_cond = get_acc_input("./input/acc", [3])
    z_cond = polydis_model.txt_encode(acc_cond.reshape([4 * 128, 32, 128])).reshape(
        4, 128, -1
    )
    print(z_cond.shape)
    acc_uncond = get_acc_input("./input/acc-uncond", [3])
    z_uncond = polydis_model.txt_encode(acc_uncond.reshape([4 * 128, 32, 128])).reshape(
        4, 128, -1
    )
    print(z_uncond.shape)

    fpath = "./results/txt_sim.txt"
    record_latent_similarity(z_org, z_cond, fpath, "cond")
    record_latent_similarity(z_org, z_uncond, fpath, "uncond")
    record_latent_similarity(z_org, z_resampled, fpath, "resampled")


def compute_mel():
    mel_prompt = np.load("./input/melody_prompt.npy")
    print(mel_prompt.shape)
    # (32, 128, 142)
    mel_expand = (
        np.expand_dims(mel_prompt, axis=0).repeat(4, axis=0).reshape((512, 32, 142))
    )
    utils.melprmat_to_midi_file(mel_expand, "./input/mel_prompt_expand.mid")
    mel_prompt = np.reshape(mel_prompt, (128, 32, 142))
    # utils.prmat_to_midi_file(mel_prompt, "./input/acc_prompt.mid")
    mel_prompt = torch.from_numpy(mel_prompt).to(device).float()
    mel_org = mel_prompt[:, :, :130]
    utils.melprmat_to_midi_file(mel_prompt, "./input/mel_prompt.mid")
    chd_org = mel_prompt[:, :, 130:]
    # utils.chd_ec2vae_to_midi_file(chd_org, "mel_chd_prompt.mid")
    z_org = ec2vae_model.encoder(mel_org, chd_org)[1]
    # zp_org = ec2vae_model.encoder(mel_org, chd_org)[0]
    print(z_org.shape)
    z_resampled = get_resampled_zr(z_org, chd_org)
    print(z_resampled.shape)

    mel_cond, chd_cond = get_mel_input("./input/mel", [1])
    z_cond = ec2vae_model.encoder(
        mel_cond.reshape([4 * 128, 32, 130]), chd_cond.reshape(4 * 128, 32, 12)
    )[1].reshape(4, 128, -1)
    print(z_cond.shape)
    mel_uncond, chd_uncond = get_mel_input("./input/mel-uncond", [1])
    z_uncond = ec2vae_model.encoder(
        mel_uncond.reshape([4 * 128, 32, 130]), chd_uncond.reshape(4 * 128, 32, 12)
    )[1].reshape(4, 128, -1)
    print(z_uncond.shape)

    fpath = "./results/mel_sim.txt"
    record_latent_similarity(z_org, z_cond, fpath, "cond")
    record_latent_similarity(z_org, z_uncond, fpath, "uncond")
    record_latent_similarity(z_org, z_resampled, fpath, "resampled")


def compute_chd8():
    with torch.no_grad():
        chd8_prompt = np.load("./input/chd_prompt.npy")
        print(chd8_prompt.shape)
        # (4, 32, 128, 36)
        # utils.prmat_to_midi_file(chd8_prompt, "./input/chd_prompt.mid")
        chd8_prompt = torch.from_numpy(chd8_prompt).to(device)
        chd8_prompt = (
            chd8_prompt.reshape(4, 32, 4, 32, 36)
            .reshape(4, 128, 32, 36)
            .reshape(4 * 128, 32, 36)
        )
        utils.chd8_to_midi_file(chd8_prompt, "./input/chd_prompt.mid")

        chd8_cond = get_chd8_input("./input/red", [0])
        z_cond = chd8_enc.forward(chd8_cond.reshape([4 * 128, 32, 36])).reshape(
            4, 128, -1
        )
        print(z_cond.shape)
        chd8_uncond = get_chd8_input("./input/red-uncond", [0])
        z_uncond = chd8_enc.forward(chd8_uncond.reshape([4 * 128, 32, 36])).reshape(
            4, 128, -1
        )
        print(z_uncond.shape)

        z_org = chd8_enc.forward(chd8_prompt)
        z_org = z_org.reshape(4, 128, 512)
        print(z_org.shape)
        z_resampled = get_resampled_zchd8(z_org)
        print(z_resampled.shape)

        fpath = "./results/chd8_sim.txt"
        record_latent_similarity(z_org, z_cond, fpath, "cond")
        record_latent_similarity(z_org, z_uncond, fpath, "uncond")
        record_latent_similarity(z_org, z_resampled, fpath, "resampled")


def compute_chd():
    chd_prompt = np.load("./input/chd_prompt.npy")
    print(chd_prompt.shape)
    # (4, 32, 128, 36)
    chd_prompt = torch.from_numpy(chd_prompt).to(device)
    chd_prompt = (
        chd_prompt.reshape(4, 32, 16, 8, 36)
        .reshape(4, 512, 8, 36)
        .reshape(4 * 512, 8, 36)
    )
    utils.chd8_to_midi_file(chd_prompt, "./input/chd_prompt.mid")

    z_org = polydis_model.chd_encode(chd_prompt)
    z_org = z_org.reshape(4, 512, 256)
    print(z_org.shape)
    z_resampled = get_resampled_zchd(z_org)
    print(z_resampled.shape)

    chd_cond = get_chd8_input("./input/red", [0]).reshape(4, 512, 8, 36)
    z_cond = polydis_model.chd_encode(chd_cond.reshape([4 * 512, 8, 36])).reshape(
        4, 512, -1
    )
    print(z_cond.shape)
    chd_uncond = get_chd8_input("./input/red-uncond", [0]).reshape(4, 512, 8, 36)
    z_uncond = polydis_model.chd_encode(chd_uncond.reshape([4 * 512, 8, 36])).reshape(
        4, 512, -1
    )
    print(z_uncond.shape)

    fpath = "./results/chd_sim.txt"
    record_latent_similarity(z_org, z_cond, fpath, "cond")
    record_latent_similarity(z_org, z_uncond, fpath, "uncond")
    record_latent_similarity(z_org, z_resampled, fpath, "resampled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="{acc, mel, chd, chd8}")
    args = parser.parse_args()
    match args.type:
        case "acc":
            compute_acc()
        case "mel":
            compute_mel()
        case "chd":
            compute_chd()
        case "chd8":
            compute_chd8()
