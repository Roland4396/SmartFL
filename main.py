#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import models
from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import get_dataloaders, get_datasets, get_user_groups
from fed import Federator
from models.model_utils import KDLoss
from predict import validate, local_validate
from utils.utils import load_checkpoint, measure_flops, load_state_dict, save_user_groups, load_user_groups
from ppo_architecture_generator import generate_architecture_library

np.set_printoptions(precision=2)

args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)


def file_exists(filepath):
    """Check if file exists"""
    return os.path.exists(filepath) and os.path.isfile(filepath)


def determine_stages_to_run(args):
    """Determine which stages need to be run based on arguments and file existence"""
    stages_to_run = []
    
    # Handle --stages_only parameter
    if args.stages_only:
        specified_stages = [int(s.strip()) for s in args.stages_only.split(',')]
        for stage in specified_stages:
            if stage in [1, 2, 3]:
                stages_to_run.append(stage)
        return stages_to_run
    
    # Determine stages based on skip flags only (no automatic file detection)
    stage1_needed = not args.skip_stage1
    stage2_needed = not args.skip_stage2
    stage3_needed = not args.skip_stage3
    
    if stage1_needed:
        stages_to_run.append(1)
    if stage2_needed:
        stages_to_run.append(2)
    if stage3_needed:
        stages_to_run.append(3)
        
    return stages_to_run


def validate_stage_dependencies(args, stages_to_run):
    """Validate that required files exist for requested stages"""
    # Only check for supernet if stage 2 runs without stage 1
    if 2 in stages_to_run and 1 not in stages_to_run and not file_exists(args.supernet_save_path):
        raise FileNotFoundError(f"Stage 2 requires supernet file: {args.supernet_save_path}")
    # Only check for config library if stage 3 runs without stage 2
    if 3 in stages_to_run and 2 not in stages_to_run and not file_exists(args.config_library_path):
        raise FileNotFoundError(f"Stage 3 requires config library file: {args.config_library_path}")


def run_stage1_supernet_training(args):
    """Execute Stage 1: Supernet Training"""
    print("=" * 60)
    print("STAGE 1: SUPERNET TRAINING")
    print("=" * 60)
    
    # Import and use the existing train_supernet function
    from train_supernet import train_supernet
    
    # Set save path for supernet
    original_save_path = getattr(args, 'save_path', None)
    args.save_path = args.supernet_save_path
    
    try:
        # Call the existing supernet training function
        train_supernet(args)
        print("Stage 1 completed successfully!")
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        raise
    finally:
        # Restore original save_path
        if original_save_path:
            args.save_path = original_save_path


def run_stage2_ppo_generation(args):
    """Execute Stage 2: PPO Architecture Generation"""
    print("=" * 60)
    print("STAGE 2: PPO ARCHITECTURE GENERATION")
    print("=" * 60)
    
    print(f"Loading supernet from: {args.supernet_save_path}")
    print(f"Generating {args.num_architectures} architectures...")
    print(f"Using {args.episodes_per_batch} episodes per batch")
    
    try:
        configs = generate_architecture_library(
            supernet_path=args.supernet_save_path,
            output_path=args.config_library_path,
            num_architectures=args.num_architectures,
            episodes_per_batch=args.episodes_per_batch
        )
        
        print(f"Successfully generated {len(configs)} unique configurations")
        print(f"Saved to: {args.config_library_path}")
        print("Stage 2 completed successfully!")
        
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        raise


def handle_pipeline_stages(args):
    """Handle three-stage pipeline execution"""
    stages_to_run = determine_stages_to_run(args)
    
    if not stages_to_run:
        print("No stages to run. Use --help to see available options.")
        return 0
    
    print("SmartFL Three-Stage Pipeline")
    print("=" * 40)
    print(f"Stages to run: {stages_to_run}")
    
    # Validate dependencies
    validate_stage_dependencies(args, stages_to_run)
    
    # Execute stages
    try:
        if 1 in stages_to_run:
            run_stage1_supernet_training(args)
            print()
        
        if 2 in stages_to_run:
            run_stage2_ppo_generation(args)
            print()
        
        if 3 in stages_to_run:
            print("=" * 60)
            print("STAGE 3: FEDERATED LEARNING TRAINING")
            print("=" * 60)
            # Stage 3 will be handled by the existing main() function logic
            return None  # Continue to existing federated learning code
        
        # If only stages 1 and/or 2 were run, exit successfully
        if 3 not in stages_to_run:
            print("Pipeline completed successfully!")
            return 0
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1
    
    return None  # Continue to stage 3


def main():
    global args

    # Three-stage pipeline control
    stage_control_result = handle_pipeline_stages(args)
    if stage_control_result is not None:
        return stage_control_result

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    config = Config()

    if args.ee_locs:
        config.model_params[args.data][args.arch]['ee_layer_locations'] = args.ee_locs

    model = getattr(models, args.arch)([0,1,2,3],args, {**config.model_params[args.data][args.arch]})
    args.num_exits = config.model_params[args.data][args.arch]['num_blocks']

    if args.use_gpu:
        model = model.cuda()
        criterion = KDLoss(args).cuda()
    else:
        criterion = KDLoss(args)

    if args.resume:
        checkpoint = load_checkpoint(args, load_best=False)
        if checkpoint is not None:
            args.start_round = checkpoint['round'] + 1
            model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    batch_size = args.batch_size if args.batch_size else config.training_params[args.data][args.arch]['batch_size']
    train_set, val_set, test_set = get_datasets(args)
    _, val_loader, test_loader = get_dataloaders(args, batch_size, (train_set, val_set, test_set))
    if val_set is None:
        val_set = val_loader.dataset

    train_user_groups, val_user_groups, test_user_groups = get_user_groups(train_set, val_set, test_set, args)

    prev_user_groups = load_user_groups(args)
    if prev_user_groups is None:
        if args.resume:
            print('Could not find user groups')
            raise RuntimeError
        user_groups = (train_user_groups, val_user_groups, test_user_groups)
        save_user_groups(args, (train_user_groups, val_user_groups, test_user_groups))
    else:
        user_groups = prev_user_groups

    if args.evalmode is not None:
        load_state_dict(args, model)
        if 'global' in args.evalmode:
            validate(model, test_loader, criterion, args,save=True)
            return
        elif 'local' in args.evalmode:
            train_args = eval('argparse.' + open(os.path.join(args.save_path, 'args.txt')).readlines()[0])
            if os.path.exists(os.path.join(args.save_path, 'client_groups.pkl')):
                client_groups = pkl.load(open(os.path.join(args.save_path, 'client_groups.pkl'), 'rb'))
            else:
                client_groups = []
            federator = Federator(model, train_args, client_groups)
            local_validate(federator, test_set, user_groups[1], criterion, args, batch_size)
            return
        else:
            raise NotImplementedError

    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        print(args, file=f)

    federator = Federator(model, args)
    best_acc1, best_round = federator.fed_train(train_set, val_set, user_groups, criterion, args, batch_size,
                                                 config.training_params[args.data][args.arch])

    print('Best val_acc1: {:.4f} at round {}'.format(best_acc1, best_round))
    validate(federator.global_model, test_loader, criterion, args, save=True)

    return


if __name__ == '__main__':
    main()
