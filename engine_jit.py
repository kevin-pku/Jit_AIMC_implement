import math
import sys
import os
import shutil
import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
from torch_fidelity.metric_fid import (
    fid_input_id_to_statistics,
    fid_statistics_to_metric,
    KEY_METRIC_FID,
)
from torch_fidelity.utils import create_feature_extractor


def _compute_fid_with_stats(save_folder, fid_stats_path, batch_size):
    """Compute FID against cached statistics without reprocessing the reference set."""
    use_cuda = torch.cuda.is_available()
    feat_layer = '2048'
    fid_kwargs = {
        'input1': save_folder,
        'batch_size': max(1, min(batch_size, 512)),
        'verbose': False,
    }
    feat_extractor = create_feature_extractor(
        'inception-v3-compat', [feat_layer], cuda=use_cuda, **fid_kwargs
    )
    stats_fake = fid_input_id_to_statistics(1, feat_extractor, feat_layer, **fid_kwargs)
    stats_real_npz = np.load(fid_stats_path)
    stats_real = {
        'mu': stats_real_npz['mu'],
        'sigma': stats_real_npz['sigma'],
    }
    metric = fid_statistics_to_metric(stats_fake, stats_real, verbose=False)
    return metric[KEY_METRIC_FID]
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accumulation_steps = max(1, getattr(args, 'accumulation_steps', 1))
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accumulation_steps
        loss.backward()

        update_step = (data_iter_step + 1) % accumulation_steps == 0 or (data_iter_step + 1) == len(data_loader)
        if update_step:
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        "ssd/tmp",
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    use_ema = not getattr(args, 'disable_eval_ema', False) and getattr(args, 'eval_ema_index', 1) != 0
    ema_choice = getattr(args, 'eval_ema_index', 1)
    if use_ema:
        # switch to selected ema params
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_source = model_without_ddp.ema_params1 if ema_choice == 1 else model_without_ddp.ema_params2
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_source[i]
        print(f"Switch to ema{ema_choice}")
        model_without_ddp.load_state_dict(ema_state_dict)
    else:
        model_state_dict = None
        print("Skip EMA swap for evaluation")

    # ensure that the number of images per class is equal, or optionally spread labels for small samples.
    class_num = args.class_num
    if args.num_images < class_num or args.num_images % class_num != 0:
        if getattr(args, 'label_strategy', 'repeat') == 'spread':
            # Spread labels across the full class range for diversity
            class_label_gen_world = np.linspace(0, class_num - 1, num=args.num_images, dtype=int)
            print("[INFO] Using spread label strategy for diversity.")
        else:
            print(
                "[WARN] num_images {} not divisible by class_num {}; repeating labels as needed.".format(
                    args.num_images, class_num
                )
            )
            repeat = int(np.ceil(args.num_images / class_num))
            class_label_gen_world = np.tile(np.arange(0, class_num), repeat)[:args.num_images]
    else:
        class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        if misc.is_dist_avail_and_initialized():
            torch.distributed.barrier()

        # Debug prints requested by user
        print(
            "DEBUG [Before Denorm] min/max/mean/std:",
            sampled_images.min().item(),
            sampled_images.max().item(),
            sampled_images.mean().item(),
            sampled_images.std().item()
        )

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        
        print(
            "DEBUG [After Denorm] min/max/mean/std:",
            sampled_images.min().item(),
            sampled_images.max().item(),
            sampled_images.mean().item(),
            sampled_images.std().item()
        )

        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS (only on main process)
    if misc.is_main_process():
        if getattr(args, 'fid_stats_path', None):
            fid_statistics_file = args.fid_stats_path
        elif args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError

        if not os.path.exists(fid_statistics_file):
            print(
                "Skip FID/IS: missing statistics file {}. Keeping generated images at {}".format(
                    fid_statistics_file, save_folder
                )
            )
        else:
            fid_value = _compute_fid_with_stats(save_folder, fid_statistics_file, batch_size)
            is_metrics = torch_fidelity.calculate_metrics(
                input1=save_folder,
                isc=True,
                fid=False,
                kid=False,
                cuda=torch.cuda.is_available(),
                batch_size=max(1, min(batch_size, 1024)),
                verbose=False,
            )
            inception_score = is_metrics['inception_score_mean']
            print("FID: {:.4f}, Inception Score: {:.4f}".format(fid_value, inception_score))

            if log_writer is not None:
                postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
                log_writer.add_scalar('fid{}'.format(postfix), fid_value, epoch)
                log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)

            if not getattr(args, 'keep_generated', False):
                shutil.rmtree(save_folder)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
