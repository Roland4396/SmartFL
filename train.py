#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from utils.utils import adjust_learning_rate, accuracy, AverageMeter


def execute_epoch(model, train_loader, criterion, optimizer, round, epoch, args, train_params, h_level, level, global_model=None, is_growth_mode=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(h_level):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # DEBUG: Setup logging
    import os
    import torch
    debug_log_path = os.path.join(args.save_path, 'debug_train.log')

    # switch to train mode
    model.train()

    end = time.time()

    for i, (inp, target) in enumerate(train_loader):

        adjust_learning_rate(optimizer, round, train_params)

        data_time.update(time.time() - end)

        if args.use_gpu:
            inp = inp.cuda()
            target = target.cuda()

        output = model(inp, manual_early_exit_index=h_level)

        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        if is_growth_mode:
            # Growth 模式：只使用最后一个 exit 的 loss（避免依赖冻结的 blocks）
            loss = criterion.ce_loss(output[-1], target)
        else:
            # Normal 模式：使用所有 exit 的 loss（知识蒸馏）
            for j in range(len(output)):
                if j == len(output) - 1:
                    loss += criterion.ce_loss(output[j], target) * (j + 1)
                else:
                    gamma_active = round > args.num_rounds * 0.25
                    loss += criterion.loss_fn_kd(output[j], target, output[-1], gamma_active) * (j + 1)

        for j in range(len(output)):
            if 'bert' in args.arch:
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), inp.size(0))
            top5[j].update(prec5.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if not is_growth_mode:
            # Normal 模式：归一化 loss（因为有多个 exit 的加权 loss）
            loss /= len(output) * (len(output) + 1) / 2
        # Growth 模式不需要归一化，因为只有一个 CE loss

        # DEBUG: Check for abnormal loss values
        loss_value = loss.item()
        losses.update(loss_value, inp.size(0))

        if loss_value > 1e5 or np.isnan(loss_value) or np.isinf(loss_value):
            with open(debug_log_path, 'a') as f:
                f.write(f"\n[ERROR] Round {round}, Epoch {epoch}, Batch {i}: Abnormal loss detected!\n")
                f.write(f"  Loss value: {loss_value:.6e}\n")
                f.write(f"  Level: {level}, h_level: {h_level}\n")
                f.write(f"  Number of outputs: {len(output)}\n")
                # Check model parameters for NaN/Inf
                nan_params = []
                inf_params = []
                large_params = []
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        nan_params.append(name)
                    if torch.isinf(param).any():
                        inf_params.append(name)
                    max_val = param.abs().max().item()
                    if max_val > 1e5:
                        large_params.append((name, max_val))

                if nan_params:
                    f.write(f"  Parameters with NaN: {nan_params}\n")
                if inf_params:
                    f.write(f"  Parameters with Inf: {inf_params}\n")
                if large_params:
                    f.write(f"  Parameters with large values (>1e5):\n")
                    for name, val in large_params:
                        f.write(f"    {name}: {val:.2e}\n")

        loss.backward()

        # DEBUG: Check for abnormal gradients after backward
        if loss_value > 1e5 or np.isnan(loss_value) or np.isinf(loss_value):
            with open(debug_log_path, 'a') as f:
                nan_grads = []
                inf_grads = []
                large_grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_grads.append(name)
                        if torch.isinf(param.grad).any():
                            inf_grads.append(name)
                        max_grad = param.grad.abs().max().item()
                        if max_grad > 1e5:
                            large_grads.append((name, max_grad))

                if nan_grads:
                    f.write(f"  Gradients with NaN: {nan_grads}\n")
                if inf_grads:
                    f.write(f"  Gradients with Inf: {inf_grads}\n")
                if large_grads:
                    f.write(f"  Gradients with large values (>1e5):\n")
                    for name, val in large_grads:
                        f.write(f"    {name}: {val:.2e}\n")

        optimizer.step()

        # DEBUG: Check model weights after optimizer step
        with open(debug_log_path, 'a') as f:
            for name, param in model.named_parameters():
                if 'features.9.0.weight' in name or 'features.6.0.weight' in name or 'features.8.0.weight' in name:
                    max_val = param.abs().max().item()
                    if max_val > 1e4:  # Lower threshold to catch earlier signs
                        f.write(f"[WARN] Round {round}, Epoch {epoch}, Batch {i}: {name} = {max_val:.2e}\n")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i + 1}/{len(train_loader)}]\t\t' +
                  f'Exit: {len(output)}\t' +
                  f'Time: {batch_time.avg:.3f}\t' +
                  f'Data: {data_time.avg:.3f}\t' +
                  f'Loss: {losses.val:.4f}\t ' +
                  f'Acc@1: {top1[-1].val:.4f}\t' +
                  f'Acc@5: {top5[-1].val:.4f}')

    return losses.avg
