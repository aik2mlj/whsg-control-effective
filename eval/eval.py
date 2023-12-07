from eval_midi import *
import sys

if __name__ == "__main__":
    est_f = sys.argv[1]
    ref_f = sys.argv[2]
    est_statistics = get_statistics(
        # "/home/aik2/Learn/ComputerMusic/diffpro/exp/polyf/uncond_gen.mid"
        # "/home/aik2/Learn/ComputerMusic/Models/06_transformer_arrangement_ziyu/ismir_demo/polydis_sample/sample.mid"
        # "exp/ref2.mid"
        est_f
    )
    # fs = []
    # for f in os.listdir(ref_f):
    #     fs.append(f"{ref_f}/{f}")
    # print(len(fs))
    ref_statistics = get_statistics(
        # [R'D:\Dataset\POP909\001\001.mid', R'D:\Dataset\POP909\002\002.mid']
        # "./data/uncond.mid"
        # fs
        ref_f
    )
    print(evaluate_statistics(est_statistics, ref_statistics))
