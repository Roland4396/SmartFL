import argparse
import datetime
import os


def modify_args(args):
    if args.use_gpu and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx


    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.data == 'cifar10':
        args.num_classes = 10
        args.image_size = (32, 32)
    elif args.data == 'cifar100':
        args.num_classes = 100
        args.image_size = (32, 32)
    elif args.data == 'tiny_imagenet':
        args.num_classes = 200
        args.image_size = (64, 64)
    else:
        raise NotImplementedError

    if (
        (hasattr(args, 'arch') and args.arch and 'vit' in args.arch.lower())
        or (hasattr(args, 'model') and args.model and args.model.lower() == 'vit')
    ):
        args.image_size = (224, 224)

    if not hasattr(args, "save_path") or args.save_path is None:
        # Extract model name from arch (e.g., 'mobilenet_v2_4' -> 'mobilenet')
        if hasattr(args, 'arch') and args.arch:
            if 'mobilenet' in args.arch:
                model_name = 'mobilenet'
            elif 'convnext' in args.arch:
                model_name = 'convnext'
            elif 'vit' in args.arch:
                model_name = 'vit_small'
            elif 'resnet' in args.arch:
                model_name = 'resnet'
            elif 'vgg' in args.arch:
                model_name = 'vgg'
            else:
                model_name = getattr(args, 'model', 'vgg')
        else:
            model_name = getattr(args, 'model', 'vgg')
        dataset_name = getattr(args, 'data', 'cifar100')
        alpha = getattr(args, 'alpha', 100)
        args.save_path = f"outputs/{model_name}_{dataset_name}_{alpha}_independent"

    # Sync supernet training parameters
    if hasattr(args, 'supernet_lr') and args.supernet_lr != args.lr:
        args.lr = args.supernet_lr
    if hasattr(args, 'supernet_epochs') and args.supernet_epochs != args.epochs:
        args.epochs = args.supernet_epochs

    return args


model_names = ['msdnet24_1', 'msdnet24_4',
               'resnet110_1', 'resnet110_4',
               'vgg16_1', 'vgg16_4',
               'mobilenet_v2_1', 'mobilenet_v2_4',
               'convnext_1', 'convnext_4',
               'vit_small_1', 'vit_small_4',
               'vit_tiny_1', 'vit_tiny_4']

arg_parser = argparse.ArgumentParser(
    description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save_path', default=None,
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default=None,
                       choices=['local', 'global'],
                       help='which mode to evaluate')
exp_group.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--phase_timing', action='store_true',
                       help='record per-round phase timing breakdowns to save_path')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu_idx', default=0, type=str, help='Index of available GPU')
exp_group.add_argument('--use_gpu', default=1, type=int, help='Use CPU if zero')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar100',
                        choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
data_group.add_argument('-jj', '--num_fed_workers', default=1, type=int, metavar='N',
                        help='number of fl workers (default: 1)')
# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--model', metavar='MODEL', default='resnet',
                        choices=['resnet', 'vgg', 'mobilenet', 'convnext', 'vit'],
                        help='model type to use (default: resnet)')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet110_4',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet110_4)')
arch_group.add_argument('--ee_locs', type=int, nargs='*', default=[], help='ee locations')

# training related
optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')

optim_group.add_argument('--start_round', default=0, type=int, metavar='N',
                         help='manual round number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', type=int, help='mini-batch size')
optim_group.add_argument('--KD_gamma', type=float, default=0, help='KD gamma')
optim_group.add_argument('--KD_T', type=int, default=3, help='KD T')

# FL related
fl_group = arg_parser.add_argument_group('fl', 'FL setting')
fl_group.add_argument('--vertical_scale_ratios', type=float, nargs='*', default=[0.7, 0.7, 0.75, 1],
                      help='model split ratio vertically for each complexity level')
fl_group.add_argument('--horizontal_scale_ratios', type=int, nargs='*', default=[1, 2, 3, 4],
                      help='model horizontal split indices for each complexity level')
fl_group.add_argument('--client_split_ratios', type=float, nargs='*', default=[0.25, 0.25, 0.25, 0.25],
                      help='client ratio at each complexity level')
fl_group.add_argument('--num_rounds', type=int, default=400,
                      help='number of rounds')
fl_group.add_argument('--num_clients', type=int, default=100,
                      help='number of clients')
fl_group.add_argument('--sample_rate', type=float, default=0.1,
                      help='client sample rate')
fl_group.add_argument('--validate_every', type=int, default=1,
                      help='run local validation every N rounds; 1 means every round')
fl_group.add_argument('--alpha', type=int, default=100,
                      help='data nonIID alpha')
fl_group.add_argument('-trs', '--track_running_stats', action='store_true',
                      help='trs')
fl_group.add_argument('--flops_constraints', type=float, nargs='*', default=[83.4,99.7,138.5,253.1],
                      help='Max FLOPs (M) for each level (0 to 3)')
fl_group.add_argument('--params_constraints', type=float, nargs='*', default=[0.21,0.46,0.86,1.73],
                      help='Max parameters (M) for each level (0 to 3). Optional, same length as flops_constraints if provided.')
fl_group.add_argument('--independent_selection', action='store_true',
                      help='Use independent model selection (no hierarchical constraints). Each level independently maximizes nuclear norm.')
fl_group.add_argument('--use_random_search', action='store_true',
                      help='Use pure random search for architecture generation (true random baseline, no PPO optimization)')

# Time-Domain Decomposition (TDD) parameters
tdd_group = arg_parser.add_argument_group('tdd', 'Time-Domain Decomposition setting')
tdd_group.add_argument('--enable_tdd', type=int, default=0,
                       help='Enable Time-Domain Decomposition (0=disabled, 1=enabled)')
tdd_group.add_argument('--rotation_period', type=int, default=10,
                       help='Rotation period for TDD in rounds (default: 10)')
tdd_group.add_argument('--tdd_growth_ratio', type=float, default=0.5,
                       help='Ratio of devices using Growth mode vs Normal mode (default: 0.5)')
tdd_group.add_argument('--tdd_growth_budget_scale', type=float, default=1.0,
                       help='Allowed Growth memory budget relative to normal training budget (default: 1.0)')
# Three-stage pipeline control
pipeline_group = arg_parser.add_argument_group('pipeline', 'Three-stage pipeline control')
pipeline_group.add_argument('--skip_stage1', action='store_true',
                           help='Skip stage 1: supernet training')
pipeline_group.add_argument('--skip_stage2', action='store_true',
                           help='Skip stage 2: PPO architecture generation')
pipeline_group.add_argument('--skip_stage3', action='store_true',
                           help='Skip stage 3: federated learning training')
pipeline_group.add_argument('--stages_only', type=str, default=None,
                           help='Run only specific stages (e.g., "1,2" or "3")')

# Stage 1: Supernet training parameters
stage1_group = arg_parser.add_argument_group('stage1', 'Supernet training parameters')
stage1_group.add_argument('--supernet_epochs', type=int, default=100,
                         help='Number of epochs for supernet training')
stage1_group.add_argument('--supernet_lr', type=float, default=0.1,
                         help='Learning rate for supernet training')
stage1_group.add_argument('--supernet_batch_size', type=int, default=128,
                         help='Batch size for supernet training')
stage1_group.add_argument('--supernet_save_path', type=str, default='supernet.pth',
                         help='Path to save trained supernet')
stage1_group.add_argument('--lr', type=float, default=0.1,
                         help='Learning rate (alias for supernet_lr)')
stage1_group.add_argument('--epochs', type=int, default=200,
                         help='Training epochs (alias for supernet_epochs)')

# Stage 2: PPO architecture generation parameters
stage2_group = arg_parser.add_argument_group('stage2', 'PPO architecture generation parameters')
stage2_group.add_argument('--num_architectures', type=int, default=50000,
                         help='Number of architectures to generate via PPO')
stage2_group.add_argument('--episodes_per_batch', type=int, default=100,
                         help='PPO episodes per batch for architecture generation')
stage2_group.add_argument('--ppo_learning_rate', type=float, default=0.001,
                         help='Learning rate for PPO agent')
stage2_group.add_argument('--config_library_path', type=str, default=None,
                         help='Path to save/load architecture configuration library. If not specified, will auto-generate based on model and dataset')
