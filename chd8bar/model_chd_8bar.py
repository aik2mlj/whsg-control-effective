import torch
import torch.nn as nn

from .chord_enc import RnnEncoder as ChordEncoder
from .chord_dec import ChordDecoder


class Chord_8Bar(nn.Module):
    def __init__(self, chord_enc: ChordEncoder, chord_dec: ChordDecoder):
        super(Chord_8Bar, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chord_enc = chord_enc
        self.chord_dec = chord_dec

    @classmethod
    def load_trained(cls, chord_enc, chord_dec, model_dir):
        model = cls(chord_enc, chord_dec)
        trained_leaner = torch.load(f"{model_dir}/weights.pt")
        model.load_state_dict(trained_leaner["model"])
        return model

    def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
        loss_fun = nn.CrossEntropyLoss()
        root = c[:, :, 0:12].max(-1)[-1].view(-1).contiguous()
        chroma = c[:, :, 12:24].long().view(-1).contiguous()
        bass = c[:, :, 24:].max(-1)[-1].view(-1).contiguous()

        recon_root = recon_root.view(-1, 12).contiguous()
        recon_chroma = recon_chroma.view(-1, 2).contiguous()
        recon_bass = recon_bass.view(-1, 12).contiguous()
        root_loss = loss_fun(recon_root, root)
        chroma_loss = loss_fun(recon_chroma, chroma)
        bass_loss = loss_fun(recon_bass, bass)
        chord_loss = root_loss + chroma_loss + bass_loss
        return {
            "loss": chord_loss,
            "root": root_loss,
            "chroma": chroma_loss,
            "bass": bass_loss,
        }

    def get_loss_dict(self, batch, step, tfr_chd):
        _, _, chord, _ = batch

        z_chd = self.chord_enc(chord).rsample()
        recon_root, recon_chroma, recon_bass = self.chord_dec(
            z_chd, False, tfr_chd, chord
        )
        return self.chord_loss(chord, recon_root, recon_chroma, recon_bass)


def load_pretrained_chd_enc_dec(
    fpath, input_dim, z_input_dim, hidden_dim, z_dim, n_step
):
    chord_enc = ChordEncoder(input_dim, hidden_dim, z_dim)
    chord_dec = ChordDecoder(input_dim, z_input_dim, hidden_dim, z_dim, n_step)
    checkpoint = torch.load(fpath)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    from collections import OrderedDict

    enc_chkpt = OrderedDict()
    dec_chkpt = OrderedDict()
    for k, v in checkpoint.items():
        part = k.split(".")[0]
        name = ".".join(k.split(".")[1:])
        if part == "chord_enc":
            enc_chkpt[name] = v
        elif part == "chord_dec":
            dec_chkpt[name] = v
    chord_enc.load_state_dict(enc_chkpt)
    chord_dec.load_state_dict(dec_chkpt)
    return chord_enc, chord_dec
