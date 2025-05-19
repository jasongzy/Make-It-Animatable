import argparse
import datetime
import json
import os
import time

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(8)

import util.misc as misc
from engine import evaluate, train_one_epoch
from model import PCAE
from util.dataset_mixamo import collate, seed_worker
from util.utils import find_ckpt, fix_random, str2bool, str2list


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--point_cloud_size", default=32768, type=int, help="input size")
    parser.add_argument("--num_latents", default=512, type=int)
    parser.add_argument("--input_normal", type=str2bool, default=False)
    parser.add_argument("--input_attention", type=str2bool, default=False)
    parser.add_argument("--actvn", default="softmax", type=str, choices=("softmax", "sigmoid", "relu", "softplus"))
    parser.add_argument("--aug_rotation", type=str2bool, default=True)
    parser.add_argument("--drop_normal_ratio", type=float, default=0.0)
    parser.add_argument("--predict_bw", type=str2bool, default=True)
    parser.add_argument("--predict_joints", type=str2bool, default=False)
    parser.add_argument("--predict_joints_tail", type=str2bool, default=False)
    parser.add_argument("--joints_attn", type=str2bool, default=False)
    parser.add_argument("--joints_attn_masked", type=str2bool, default=True)
    parser.add_argument("--joints_attn_causal", type=str2bool, default=False)
    parser.add_argument("--predict_global_trans", type=str2bool, default=False)
    parser.add_argument("--predict_pose_trans", type=str2bool, default=False)
    parser.add_argument(
        "--pose_mode",
        type=str,
        default="ortho6d",
        choices=(
            "transl_quat",
            "dual_quat",
            "transl_ortho6d",
            "target_quat",
            "target_ortho6d",
            "quat",
            "ortho6d",
            "local_quat",
            "local_ortho6d",
        ),
    )
    parser.add_argument("--pose_input_joints", type=str2bool, default=True)
    parser.add_argument("--pose_attn", type=str2bool, default=False)
    parser.add_argument("--pose_attn_masked", type=str2bool, default=True)
    parser.add_argument("--pose_attn_causal", type=str2bool, default=False)

    # Optimizer parameters
    parser.add_argument("--loss", default="l1", type=str, choices=("l2", "l1", "kl"))
    parser.add_argument("--use_joints_connect_loss", type=str2bool, default=True)
    parser.add_argument("--use_joints_rest_loss", type=str2bool, default=False)
    parser.add_argument("--use_pose_rest_loss", type=str2bool, default=False)
    parser.add_argument("--use_rest_prior_loss", type=str2bool, default=False)
    parser.add_argument("--use_pose_connect_loss", type=str2bool, default=False)
    parser.add_argument("--use_pose_adv_loss", type=str2bool, default=False)
    parser.add_argument(
        "--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument("--layer_decay", type=float, default=0.75, help="layer-wise lr decay from ELECTRA/BEiT")
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )
    parser.add_argument("--warmup_epochs", type=float, default=0.02, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_path", default="data/Mixamo", type=str, help="dataset path")
    parser.add_argument("--extra_char_path", default=None, type=str2list(str))
    parser.add_argument("--extra_anim_path", default="data/Mixamo/animation_extra", type=str2list(str))
    parser.add_argument("--use_additional_bones", type=str2bool, default=False)
    parser.add_argument("--output_dir", default="./output/", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output/", help="path where to tensorboard log")
    parser.add_argument("--expname", default="debug")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", type=str, help="resume from checkpoint")
    parser.add_argument("--resume_strict", type=str2bool, default=True)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--train_with_val", type=str2bool, default=False)
    parser.add_argument("--eval", type=str2bool, default=False, help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        type=str2bool,
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        type=str2bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--sample_frames", default=2, type=int)
    parser.add_argument("--sample_vertices", default=1e4, type=int)
    parser.add_argument("--hands_resample_ratio", default=0.0, type=float)
    parser.add_argument("--geo_resample_ratio", default=0.0, type=float)

    # distributed training parameters
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=str2bool, default=False)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def main(args):
    misc.init_distributed_mode(args)
    # print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    # print(f"{args}".replace(", ", ",\n"))
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    fix_random(args.seed + misc.get_rank())

    if args.use_additional_bones:
        from util.dataset_mixamo_additional import JOINTS_NUM, KINEMATIC_TREE, MixamoDataset
    else:
        from util.dataset_mixamo import JOINTS_NUM, KINEMATIC_TREE, MixamoDataset

    dataset_train = MixamoDataset(
        args.data_path,
        extra_character_dir=args.extra_char_path,
        extra_animation_dir=args.extra_anim_path,
        sample_points=args.point_cloud_size,
        sample_vertices=args.sample_vertices if args.predict_bw else 1,
        sample_frames=args.sample_frames,
        hands_resample_ratio=args.hands_resample_ratio,
        geo_resample_ratio=args.geo_resample_ratio,
        include_rest=True,
        get_normals=args.input_normal,
        split="train",
        train_with_val=args.train_with_val,
    )
    print(f"Train set: {len(dataset_train)}")
    dataset_val = MixamoDataset(
        args.data_path,
        sample_points=args.point_cloud_size,
        sample_vertices=-1 if args.predict_bw else 1,
        sample_frames=[0, 10, 20, 30],
        hands_resample_ratio=args.hands_resample_ratio,
        geo_resample_ratio=args.geo_resample_ratio,
        include_rest=True,
        get_normals=args.input_normal,
        split="val",
    )
    print(f"Val set: {len(dataset_val)}")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # print(f"Sampler_train = {str(sampler_train)}")
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        collate_fn=collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Train loader: {len(data_loader_train)}")
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        collate_fn=collate,
        # batch_size=args.batch_size,
        batch_size=1,
        num_workers=args.num_workers,
        # num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Val loader: {len(data_loader_val)}")

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.expname)
        if not args.distributed or global_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
    log_writer = None
    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, args.expname)
        if (not args.distributed or global_rank == 0) and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)

    model = PCAE(
        N=args.point_cloud_size,
        input_normal=args.input_normal,
        input_attention=args.input_attention,
        num_latents=args.num_latents,
        deterministic=args.deterministic,
        hierarchical_ratio=args.hands_resample_ratio + args.geo_resample_ratio,
        output_dim=JOINTS_NUM,
        output_actvn=args.actvn,
        output_log=args.loss == "kl",
        kinematic_tree=KINEMATIC_TREE,
        tune_decoder_self_attn=True,
        tune_decoder_cross_attn=True,
        predict_bw=args.predict_bw,
        predict_joints=args.predict_joints,
        predict_joints_tail=args.predict_joints_tail,
        joints_attn=args.joints_attn,
        joints_attn_masked=args.joints_attn_masked,
        joints_attn_causal=args.joints_attn_causal,
        predict_global_trans=args.predict_global_trans,
        predict_pose_trans=args.predict_pose_trans,
        pose_mode=args.pose_mode,
        pose_input_joints=args.pose_input_joints,
        pose_attn=args.pose_attn,
        pose_attn_masked=args.pose_attn_masked,
        pose_attn_causal=args.pose_attn_causal,
    )
    if args.resume:
        args.resume = find_ckpt(args.resume)
        misc.load_model(
            args=args,
            model_without_ddp=model,
            model_preprocess=lambda x: model.adapt_ckpt(x),
            strict=args.resume_strict,
        )
    else:
        model.load_base(
            "output/ae/ae_d512_m512/checkpoint-199.pth"
            if args.deterministic
            else "output/ae/kl_d512_m512_l8/checkpoint-199.pth"
        )
    model.freeze_base()
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Model = {str(model_without_ddp)}")
    print(f"number of params (M): {n_parameters / 1.0e6:.2f}")
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.log_dir and misc.is_main_process():
        with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(args.__repr__() + "\n")

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = misc.NativeScalerWithGradNormCount()
    args.loss = args.loss.lower()
    if args.loss == "l2":
        criterion = torch.nn.MSELoss()
    elif args.loss == "l1":
        criterion = torch.nn.SmoothL1Loss()
    elif args.loss == "kl":
        criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    else:
        raise ValueError(f"Invalid loss {args.loss}")
    print(f"criterion = {str(criterion)}")

    # if args.resume:
    #     args.resume = find_ckpt(args.resume)
    #     misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        raise NotImplementedError

    print(f"Start training for {args.epochs} epochs from epoch {args.start_epoch}")
    start_time = time.perf_counter()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
            report_every=0.05,
            data_loader_val=data_loader_val,
        )
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_only=True,
            )

        if epoch % 1 == 0 or epoch + 1 == args.epochs:
            test_stats, _ = evaluate(data_loader_val, model, device, args)
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.log_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.perf_counter() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = get_args_parser().parse_args()
    main(args)
