import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Number of gradient accumulation steps to simulate larger batches')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument(
        '--timestep_schedule',
        default='linear',
        type=str,
        choices=['linear', 'logit_normal', 'log'],
        help='Inference timestep schedule in [0,1]. Use `linear` to match the paper; use `logit_normal`/`log` to mimic the training logit-normal time distribution.'
    )
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    # Paper-recommended CFG interval: start applying CFG after 10% of the trajectory
    parser.add_argument('--interval_min', default=0.1, type=float,
                        help='CFG interval min (paper default 0.1)')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max (paper default 1.0)')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--label_strategy', default='repeat', type=str,
                        choices=['repeat', 'spread'],
                        help="How to assign class labels when num_images is not divisible by class_num: 'repeat' (default) repeats from class 0 upward; 'spread' spaces labels across the class range for diversity.")
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--disable_eval_ema', action='store_true',
                        help='Do not use EMA weights during evaluation')
    parser.add_argument(
        '--eval_ema_index',
        default=1,
        type=int,
        choices=[0, 1, 2],
        help='Which EMA to use during evaluation: 0 = no EMA, 1 = ema_decay1 (default), 2 = ema_decay2.'
    )
    parser.add_argument('--ffn_bitserial', dest='ffn_bitserial', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable FFN bit-serial CIM simulation (INT8 weights, configurable effective activation and ADC bits) during inference; disable with --no-ffn-bitserial')
    parser.add_argument('--ffn_use_kl_scales', default='', type=str,
                        help='Path to FFN KL calibration scales (json/npz) for static quant; optional')
    parser.add_argument('--ffn_int7_weights', default='', type=str,
                        help='Path to pre-quantized FFN INT7 weights state_dict; optional')
    parser.add_argument('--ffn_weight_clip_pct', default=0.0, type=float,
                        help='Optional percentile clipping (e.g., 99.9) on FFN weights before quantization in bit-serial mode')
    parser.add_argument('--ffn_weight_nbit', default=8, type=int,
                        help='Bitwidth for FFN bit-serial weight quantization (e.g., 8 for INT8, 7 for INT7)')
    parser.add_argument('--ffn_act_nbit', default=12, type=int,
                        help='Effective activation bitwidth (fixed INT12 split into 7-bit MSB + 5-bit LSB) for bit-serial CIM simulation')
    parser.add_argument('--ffn_msb_samples', default=2, type=int,
                        help='Number of repeated MSB samples to average for noise reduction (default 2, up to 4)')
    parser.add_argument('--ffn_lsb_gain_shift', default=2, type=int,
                        help='Left-shift applied to the 5-bit LSB slice before CIM (default 2 -> 4x DAC gain)')
    parser.add_argument('--ffn_adc_nbit', default=10, type=int,
                        help='ADC bitwidth used in bit-serial CIM simulation (default 10)')
    parser.add_argument('--ffn_msb_noise_sigma_lsb', default=2.0, type=float,
                        help='Gaussian noise sigma (in ADC LSBs) applied per MSB sample before averaging; set 0 to disable')
    parser.add_argument('--keep_generated', action='store_true',
                        help='Do not delete generated samples after FID/IS (for exporting)')
    parser.add_argument('--fid_stats_path', default='', type=str,
                        help='Optional path to FID stats npz; if empty, use built-in jit_in{img_size}_stats.npz')

    # dataset
    parser.add_argument('--data_path', default='/data/imagenet/imagenet-folder', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--subset_frac', default=1.0, type=float,
                        help='Fraction of training samples to use (0 < frac <= 1)')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', dest='local_rank', default=-1, type=int,
                        help='Alias for torch.distributed.launch compatibility')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def setup_logging(args, global_rank):
    """Create a tensorboard writer on the main process only."""
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        return SummaryWriter(log_dir=args.output_dir)
    return None


def build_data_loader(args, seed, global_rank, num_tasks):
    """Construct the training dataset, optional subset, and dataloader."""
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    full_train_len = len(dataset_train)
    if not 0 < args.subset_frac <= 1:
        raise ValueError('--subset_frac must be in (0, 1]')
    if args.subset_frac < 1.0:
        subset_size = max(1, int(full_train_len * args.subset_frac))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(full_train_len, generator=generator)[:subset_size]
        dataset_train = torch.utils.data.Subset(dataset_train, indices)
        if global_rank == 0:
            pct = 100.0 * subset_size / full_train_len
            print(f'Using subset of ImageNet: {subset_size}/{full_train_len} samples ({pct:.2f}%)')
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    return data_loader_train


def create_model_and_optimizer(args, device):
    """Instantiate the denoiser model and its optimizer."""
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    return model, model_without_ddp, optimizer


def resume_or_initialize(args, model_without_ddp, optimizer):
    """Resume from checkpoint when available, otherwise initialize EMA buffers."""
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
        return

    model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
    model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
    print("Training from scratch")


def maybe_run_evaluation(model_without_ddp, args, seed, log_writer):
    """Run standalone evaluation path when requested."""
    if not args.evaluate_gen:
        return False

    print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        with torch.no_grad():
            evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
    return True


def train_epochs(model, model_without_ddp, data_loader_train, optimizer, device, log_writer, args, seed):
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    if not hasattr(args, 'gpu') or args.gpu is None:
        # When not launched via torch.distributed, default to GPU 0
        args.gpu = device.index if device.index is not None else 0

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    log_writer = setup_logging(args, global_rank)
    data_loader_train = build_data_loader(args, seed, global_rank, num_tasks)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    # Fall back to eager if a backend compiler (e.g., inductor) fails, instead of crashing
    torch._dynamo.config.suppress_errors = True

    model, model_without_ddp, optimizer = create_model_and_optimizer(args, device)
    resume_or_initialize(args, model_without_ddp, optimizer)

    if maybe_run_evaluation(model_without_ddp, args, seed, log_writer):
        return

    train_epochs(model, model_without_ddp, data_loader_train, optimizer, device, log_writer, args, seed)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
