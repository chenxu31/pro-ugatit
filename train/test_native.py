import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdb
import torch
import numpy
import argparse
from main_utils import *
from models.ugatit_pro import UGATIT
from configs.cfgs_ugatit_native import test_cfgs
import platform
import skimage.io
from skimage.metrics import structural_similarity as SSIM

if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_net_pt as common_net
import common_cmf_pt as common_cmf
import common_pelvic_pt as common_pelvic


def main(device, args):
    model = UGATIT(args)
    model.load_state_dict(os.path.join(args.result_dir, args.dataset, 'train_last.pt'))
    model.G_A.eval()
    model.G_B.eval()
    if args.gpu >= 0:
        model.cuda()

    if args.dataset == "pelvic":
        common_file = common_pelvic
        test_data_s, test_data_t, _, _ = common_pelvic.load_test_data(args.data_dir, valid=True)
    elif args.dataset == "cmf":
        common_file = common_cmf
        test_data_t, test_data_s, _ = common_cmf.load_test_data(args.data_dir)
    else:
        assert 0

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    patch_shape = (args.inc, test_data_s[0].shape[1], test_data_s[0].shape[2])
    test_st_psnr = numpy.zeros((len(test_data_s), 1), numpy.float32)
    test_ts_psnr = numpy.zeros((len(test_data_t), 1), numpy.float32)
    test_st_ssim = numpy.zeros((len(test_data_s), 1), numpy.float32)
    test_ts_ssim = numpy.zeros((len(test_data_t), 1), numpy.float32)
    test_st_mae = numpy.zeros((len(test_data_s), 1), numpy.float32)
    test_ts_mae = numpy.zeros((len(test_data_t), 1), numpy.float32)
    with torch.no_grad():
        for i in range(len(test_data_s)):
            syn_st = common_net.produce_results(device, model.G_A, [patch_shape, ], [test_data_s[i], ],
                                                data_shape=test_data_s[i].shape, patch_shape=patch_shape)
            syn_ts = common_net.produce_results(device, model.G_B, [patch_shape, ], [test_data_t[i], ],
                                                data_shape=test_data_t[i].shape, patch_shape=patch_shape)

            if args.output_dir:
                common_file.save_nii(syn_st, os.path.join(args.output_dir, "syn_st_%d.nii.gz" % i))
                common_file.save_nii(syn_ts, os.path.join(args.output_dir, "syn_ts_%d.nii.gz" % i))

            st_psnr = common_metrics.psnr(syn_st, test_data_t[i])
            ts_psnr = common_metrics.psnr(syn_ts, test_data_s[i])
            st_ssim = SSIM(syn_st, test_data_t[i], data_range=2.)
            ts_ssim = SSIM(syn_ts, test_data_s[i], data_range=2.)
            st_mae = abs(common_file.restore_hu(syn_st) - common_file.restore_hu(test_data_t[i])).mean()
            ts_mae = abs(common_file.restore_hu(syn_ts) - common_file.restore_hu(test_data_s[i])).mean()

            print(i, ts_psnr)

            test_st_psnr[i] = st_psnr
            test_ts_psnr[i] = ts_psnr
            test_st_ssim[i] = st_ssim
            test_ts_ssim[i] = ts_ssim
            test_st_mae[i] = st_mae
            test_ts_mae[i] = ts_mae

    msg = "test_st_psnr:%f/%f  test_st_ssim:%f/%f  test_st_mae:%f/%f  test_ts_psnr:%f/%f  test_ts_ssim:%f/%f  test_ts_mae:%f/%f\n" % \
          (test_st_psnr.mean(), test_st_psnr.std(), test_st_ssim.mean(), test_st_ssim.std(), test_st_mae.mean(), test_st_mae.std(),
           test_ts_psnr.mean(), test_ts_psnr.std(), test_ts_ssim.mean(), test_ts_ssim.std(), test_ts_mae.mean(), test_ts_mae.std())
    print(msg)

    if args.output_dir:
        with open(os.path.join(args.output_dir, "result.txt"), "w") as f:
            f.write(msg)

        numpy.save(os.path.join(args.output_dir, "st_psnr.npy"), test_st_psnr)
        numpy.save(os.path.join(args.output_dir, "ts_psnr.npy"), test_ts_psnr)
        numpy.save(os.path.join(args.output_dir, "st_ssim.npy"), test_st_ssim)
        numpy.save(os.path.join(args.output_dir, "ts_ssim.npy"), test_ts_ssim)
        numpy.save(os.path.join(args.output_dir, "st_mae.npy"), test_st_mae)
        numpy.save(os.path.join(args.output_dir, "ts_mae.npy"), test_ts_mae)


if __name__ == '__main__':
    args = handleParser()
    args.__dict__.update(test_cfgs)

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
