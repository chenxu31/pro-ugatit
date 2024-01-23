import argparse
import os

def str2bool(x):
        return x.lower() in ('true')

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
    
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))
    return args

def handleDatasetPath(datasetName):
    return os.path.join('dataset', datasetName, 'trainA'),os.path.join('dataset', datasetName, 'trainB')

def handleParser():
    desc = "Pytorch implementation of PRO-U-GAT-IT with multi-GPU enabled."
    parser = argparse.ArgumentParser(description=desc)

    # Pharses
    parser.add_argument('--phase', type=str, default='train', help='[train / test / ran]')

    # Dataset Settings
    parser.add_argument('--dataset', type=str, default='coser', help='dataset_name')
    parser.add_argument('--direction', type=str, default='AtoB', help='dataset_name')
    parser.add_argument('--load_size', type=int, default=256, help='dataset_name')
    parser.add_argument('--anime', type=str2bool, default=True, help='dataset_name')

    # Training print/log
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--worker', type=int, default=0, help='num of workers')
    parser.add_argument('--print_freq', type=int, default=10000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')

    # Discord Report Rate
    parser.add_argument('--report_freq', type=int, default=5, help='The number of model process to discord servers freq')
    parser.add_argument('--report_image_freq', type=int, default=10, help='The number of model process images to discord servers freq')

    # Learning rate and decays
    parser.add_argument('--lr', type=float, default=0.0005, help='The learning rate')
    parser.add_argument('--lr_decay_epoch', type=int, default=50, help='The epochs decays the learning rate')
    parser.add_argument('--lr_decay_method', type=str, default='linear', help='The method that decays the learning rate')

    # Lamada for each objective functions
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--step_epoch_addr', type=int, default=8)
    parser.add_argument('--step', type=int, default=1 ,help='init step.') 
    parser.add_argument('--max_step', type=int, default=3)

    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory name to save the results')
    parser.add_argument('--debug', type=int, default=0, help='debug flag')

    return parser.parse_args()

def argFixer(commandArg, pathA, pathB):
    arg = commandArg

    arg.dirA = pathA
    arg.dirB = pathB

    return arg
