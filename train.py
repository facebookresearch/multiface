# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from models import DeepAppearanceVAE, WarpFieldVAE
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import Renderer


def main(args, camera_config, test_segment):
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset_train = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_train,
        args.tex_size,
        camset=None if camera_config is None else camera_config["train"],
        exclude_prefix=test_segment,
    )
    dataset_test = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_test,
        args.tex_size,
        camset=None if camera_config is None else camera_config["test"],
        valid_prefix=test_segment,
    )

    train_sampler = DistributedSampler(dataset_train)
    test_sampler = DistributedSampler(dataset_test)

    train_loader = DataLoader(
        dataset_train,
        args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.n_worker,
    )
    test_loader = DataLoader(
        dataset_test,
        args.val_batch_size,
        sampler=test_sampler,
        num_workers=args.n_worker,
    )

    if local_rank == 0:
        print("#train samples", len(dataset_train))
        print("#test samples", len(dataset_test))
        writer = SummaryWriter(log_dir=args.result_path)

    n_cams = len(set(dataset_train.cameras).union(set(dataset_test.cameras)))
    if args.arch == "base":
        model = DeepAppearanceVAE(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams
        ).to(device)
    elif args.arch == "res":
        model = DeepAppearanceVAE(
            args.tex_size,
            args.mesh_inp_size,
            n_latent=args.nlatent,
            res=True,
            n_cams=n_cams,
        ).to(device)
    elif args.arch == "warp":
        model = WarpFieldVAE(
            args.tex_size, args.mesh_inp_size, z_dim=args.nlatent, n_cams=n_cams
        ).to(device)
    elif args.arch == "non":
        model = DeepAppearanceVAE(
            args.tex_size,
            args.mesh_inp_size,
            n_latent=args.nlatent,
            res=False,
            non=True,
            n_cams=n_cams,
        ).to(device)
    elif args.arch == "bilinear":
        model = DeepAppearanceVAE(
            args.tex_size,
            args.mesh_inp_size,
            n_latent=args.nlatent,
            res=False,
            non=False,
            bilinear=True,
            n_cams=n_cams,
        ).to(device)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)
    renderer = Renderer()

    if args.model_ckpt is not None:
        print("loading checkpoint from", args.model_ckpt)
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        model.load_state_dict(torch.load(args.model_ckpt, map_location=map_location))

    optimizer = optim.Adam(model.module.get_model_params(), args.lr, (0.9, 0.999))
    optimizer_cc = optim.Adam(model.module.get_cc_params(), args.lr, (0.9, 0.999))
    mse = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    texmean = cv2.resize(dataset_train.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_train.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_train.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = dataset_train.texstd
    vertmean = (
        torch.tensor(dataset_train.vertmean, dtype=torch.float32)
        .view((1, -1, 3))
        .to(device)
    )
    vertstd = dataset_train.vertstd
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = (
        torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )

    os.makedirs(args.result_path, exist_ok=True)

    def run_net(data):
        M = data["M"].cuda()
        gt_tex = data["tex"].cuda()
        vert_ids = data["vert_ids"].cuda()
        uvs = data["uvs"].cuda()
        uv_ids = data["uv_ids"].cuda()
        avg_tex = data["avg_tex"].cuda()
        view = data["view"].cuda()
        transf = data["transf"].cuda()
        verts = data["aligned_verts"].cuda()
        photo = data["photo"].cuda()
        mask = data["mask"].cuda()
        cams = data["cam"].cuda()
        batch, channel, height, width = avg_tex.shape

        output = {}

        if args.arch == "warp":
            pred_tex, pred_verts, unwarped_tex, warp_field, kl = model(
                avg_tex, verts, view, cams=cams
            )
            output["unwarped_tex"] = unwarped_tex
            output["warp_field"] = warp_field
        else:
            pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams)
        vert_loss = mse(pred_verts, verts)

        pred_verts = pred_verts * vertstd + vertmean
        pred_tex = (pred_tex * texstd + texmean) / 255.0
        gt_tex = (gt_tex * texstd + texmean) / 255.0

        loss_mask = loss_weight_mask.repeat(batch, 1, 1, 1)
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)

        if args.lambda_screen > 0:
            screen_mask, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, loss_mask, args.resolution
            )
            pred_screen, rast_out = renderer.render(
                M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, args.resolution
            )
            screen_loss = (
                torch.mean((pred_screen - photo) ** 2 * screen_mask)
                * (255**2)
                / (texstd**2)
            )
        else:
            screen_loss, pred_screen = torch.zeros([]), None

        total_loss = 0
        if args.lambda_verts > 0:
            total_loss = total_loss + args.lambda_verts * vert_loss
        if args.lambda_tex > 0:
            total_loss = total_loss + args.lambda_tex * tex_loss
        if args.lambda_screen > 0:
            total_loss = total_loss + args.lambda_screen * screen_loss
        if args.lambda_kl > 0:
            total_loss = total_loss + args.lambda_kl * kl

        losses = {
            "total_loss": total_loss,
            "vert_loss": vert_loss,
            "screen_loss": screen_loss,
            "tex_loss": tex_loss,
            "kl": kl,
        }

        output["pred_screen"] = pred_screen
        output["pred_verts"] = pred_verts
        output["pred_tex"] = pred_tex

        return losses, output

    def save_img(data, output, tag=""):
        gt_screen = data["photo"] * 255
        gt_tex = data["tex"].cuda() * texstd + texmean
        pred_tex = torch.clamp(output["pred_tex"] * 255, 0, 255)
        if output["pred_screen"] is not None:
            pred_screen = torch.clamp(output["pred_screen"] * 255, 0, 255)
            Image.fromarray(
                pred_screen[-1].detach().cpu().numpy().astype(np.uint8)
            ).save(os.path.join(args.result_path, "pred_%s.png" % tag))
        Image.fromarray(gt_screen[-1].detach().cpu().numpy().astype(np.uint8)).save(
            os.path.join(args.result_path, "gt_%s.png" % tag)
        )
        Image.fromarray(
            gt_tex[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
        ).save(os.path.join(args.result_path, "gt_tex_%s.png" % tag))
        Image.fromarray(
            pred_tex[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
        ).save(os.path.join(args.result_path, "pred_tex_%s.png" % tag))

        if args.arch == "warp":
            warp = output["warp_field"]
            grid_img = (
                torch.tensor(
                    np.array(
                        Image.open("grid.PNG").resize((args.tex_size, args.tex_size)),
                        dtype=np.float32,
                    )[None, ...]
                )
                .permute(0, 3, 1, 2)
                .to(warp.device)
            )
            grid_img = F.grid_sample(grid_img, warp[-1:])
            Image.fromarray(
                grid_img[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
            ).save(os.path.join(args.result_path, "warp_grid_%s.png" % tag))

    prev_loss = 1e8
    prev_vert_loss = 1e8
    prev_kl = 1e8
    batch_idx, val_idx = 0, 0
    best_screen_loss = 1e8
    best_tex_loss = 1e8
    best_vert_loss = 1e8
    model.train()
    train_screen_losses = []
    train_tex_losses = []
    train_vert_losses = []
    window = 20

    begin_time = time.time()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            losses, output = run_net(data)
            if batch_idx % args.val_every == 0:
                if local_rank == 0:
                    torch.save(
                        model.state_dict(), os.path.join(args.result_path, "model.pth")
                    )
                    print(
                        "model.pth saved [Epoch {} Batch Index {}]".format(
                            epoch, batch_idx
                        )
                    )

            if (
                (losses["total_loss"].item() > args.pass_thres * prev_loss)
                or (losses["vert_loss"].item() > args.pass_thres * prev_vert_loss)
                or (losses["kl"].item() > args.pass_thres * prev_kl)
            ):
                print("throw away batch")
                continue

            if local_rank == 0:
                writer.add_scalar('train/loss_tex',losses['tex_loss'].item(), batch_idx)
                writer.add_scalar('train/loss_verts', losses['vert_loss'].item(), batch_idx)
                writer.add_scalar('train/loss_screen', losses['screen_loss'].item(), batch_idx)
                writer.add_scalar('train/loss_kl', losses['kl'].item(), batch_idx)

            prev_loss = losses["total_loss"].item()
            prev_vert_loss = losses["vert_loss"].item()
            prev_kl = losses["kl"].item()

            train_screen_losses.append(losses["screen_loss"].item())
            train_tex_losses.append(losses["tex_loss"].item())
            train_vert_losses.append(losses["vert_loss"].item())
            if len(train_screen_losses) > window:
                del train_screen_losses[0]
                del train_tex_losses[0]
                del train_vert_losses[0]

            optimizer.zero_grad()
            optimizer_cc.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer_cc.step()

            if batch_idx % args.log_every == 0:
                print(
                    "%d loss %.3f vert %.3f tex %.3f screen %.3f kl %.3f"
                    % (
                        batch_idx,
                        losses["total_loss"].item(),
                        losses["vert_loss"].item(),
                        losses["tex_loss"].item(),
                        losses["screen_loss"].item(),
                        losses["kl"].item(),
                    )
                )
                if local_rank == 0:
                    save_img(data, output, "train_%d" % batch_idx)

            if batch_idx % args.val_every == 0:
                model.eval()
                total, vert, tex, screen, kl = [], [], [], [], []
                for i, data in enumerate(test_loader):
                    optimizer_cc.zero_grad()
                    losses, output = run_net(data)
                    losses["total_loss"].backward()
                    optimizer_cc.step()
                    if i == args.val_num:
                        break

                for i, data in enumerate(test_loader):
                    with torch.no_grad():
                        losses, output = run_net(data)
                        total.append(losses["total_loss"].item())
                        vert.append(losses["vert_loss"].item())
                        tex.append(losses["tex_loss"].item())
                        screen.append(losses["screen_loss"].item())
                        kl.append(losses["kl"].item())
                    if i == args.val_num:
                        break

                tex_loss = np.array(tex).mean()
                vert_loss = np.array(vert).mean()
                screen_loss = np.array(screen).mean()
                kl = np.array(kl).mean()
                if local_rank == 0:
                    writer.add_scalar('val/loss_tex',tex_loss, val_idx)
                    writer.add_scalar('val/loss_verts', vert_loss, val_idx)
                    writer.add_scalar('val/loss_screen', screen_loss, val_idx)
                    writer.add_scalar('val/loss_kl', kl, val_idx)
                    save_img(data, output, "val_%d" % val_idx)

                val_idx += 1
                print(
                    "val %d vert %.3f tex %.3f screen %.3f kl %.3f"
                    % (val_idx, vert_loss, tex_loss, screen_loss, kl)
                )
                best_screen_loss = min(best_screen_loss, screen_loss)
                best_tex_loss = min(best_tex_loss, tex_loss)
                best_vert_loss = min(best_vert_loss, vert_loss)
                if local_rank == 0:
                    if (args.lambda_screen > 0 and best_screen_loss == screen_loss) or (
                        args.lambda_screen == 0 and best_tex_loss == tex_loss
                    ):
                        torch.save(
                            model.state_dict(),
                            os.path.join(args.result_path, "best_model.pth"),
                        )
                model.train()

            if batch_idx >= args.max_iter:
                print(
                    "best screen loss %f, best tex loss %f best vert loss %f"
                    % (best_screen_loss, best_tex_loss, best_vert_loss)
                )
                if local_rank == 0:
                    torch.save(
                        model.state_dict(), os.path.join(args.result_path, "model.pth")
                    )
                train_screen_loss = np.mean(np.array(train_screen_losses))
                train_tex_loss = np.mean(np.array(train_tex_losses))
                train_vert_loss = np.mean(np.array(train_vert_losses))
                end_time = time.time()
                print("Training takes %f seconds" % (end_time - begin_time))
                return (
                    best_screen_loss,
                    best_tex_loss,
                    best_vert_loss,
                    screen_loss,
                    tex_loss,
                    vert_loss,
                    train_screen_loss,
                    train_tex_loss,
                    train_vert_loss,
                )

            batch_idx += 1
        scheduler.step()
    print(
        "best screen loss %f, best tex loss %f best vert loss %f"
        % (best_screen_loss, best_tex_loss, best_vert_loss)
    )
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.result_path, "model.pth"))
    train_screen_loss = np.mean(np.array(train_screen_losses))
    train_tex_loss = np.mean(np.array(train_tex_losses))
    train_vert_loss = np.mean(np.array(train_vert_losses))
    end_time = time.time()
    print("Training takes %f seconds" % (end_time - begin_time))
    return (
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
        train_screen_loss,
        train_tex_loss,
        train_vert_loss,
    )


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed run"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Validation batch size"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="base",
        help="Model architecture - base|warp|res|non|bilinear",
    )
    parser.add_argument(
        "--nlatent", type=int, default=256, help="Latent code dimension - 128|256"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--resolution",
        default=[2048, 1334],
        nargs=2,
        type=int,
        help="Rendering resolution",
    )
    parser.add_argument("--tex_size", type=int, default=1024, help="Texture resolution")
    parser.add_argument(
        "--mesh_inp_size", type=int, default=21918, help="Input mesh dimension"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS",
        help="Directory to dataset root",
    )
    parser.add_argument(
        "--krt_dir",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/KRT",
        help="Directory to KRT file",
    )
    parser.add_argument(
        "--loss_weight_mask",
        type=str,
        default="./loss_weight_mask.png",
        help="Mask for weighted loss of face",
    )
    parser.add_argument(
        "--framelist_train",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt",
        help="Frame list for training",
    )
    parser.add_argument(
        "--framelist_test",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt",
        help="Frame list for testing",
    )
    parser.add_argument(
        "--test_segment_config",
        type=str,
        default=None,
        help="Directory of expression segments for testing (exclude from training)",
    )
    parser.add_argument(
        "--lambda_verts", type=float, default=1, help="Multiplier of vertex loss"
    )
    parser.add_argument(
        "--lambda_screen", type=float, default=0, help="Multiplier of screen loss"
    )
    parser.add_argument(
        "--lambda_tex", type=float, default=1, help="Multiplier of texture loss"
    )
    parser.add_argument(
        "--lambda_kl", type=float, default=1e-2, help="Multiplier of KL divergence"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200000,
        help="Maximum number of training iterations, overrides epoch",
    )
    parser.add_argument(
        "--log_every", type=int, default=1000, help="Interval of printing training loss"
    )
    parser.add_argument(
        "--val_every", type=int, default=5000, help="Interval of validating on test set"
    )
    parser.add_argument(
        "--val_num", type=int, default=500, help="Number of iterations for validation"
    )
    parser.add_argument(
        "--n_worker", type=int, default=8, help="Number of workers loading dataset"
    )
    parser.add_argument(
        "--pass_thres",
        type=int,
        default=50,
        help="If loss is x times higher than the previous batch, discard this batch",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./runs/experiment",
        help="Directory to output files",
    )
    parser.add_argument(
        "--camera_config",
        type=str,
        default=None,
        help="Directory to camera set config file",
    )
    parser.add_argument(
        "--camera_setting",
        type=str,
        default=None,
        help="Key of camera setting to camera config file",
    )
    parser.add_argument(
        "--model_ckpt", type=str, default=None, help="Model checkpoint path"
    )
    experiment_args = parser.parse_args()
    print(experiment_args)

    if experiment_args.camera_config is not None:
        f = open(experiment_args.camera_config, "r")
        camera_config = json.load(f)
        f.close()
        if experiment_args.camera_setting is not None:
            camera_set = camera_config[experiment_args.camera_setting]
        else:
            camera_set = None
    else:
        camera_set = None

    if experiment_args.test_segment_config is not None:
        f = open(experiment_args.test_segment_config, "r")
        test_segment_config = json.load(f)
        f.close()
        test_segment = test_segment_config["segment"]
    else:
        test_segment = ["EXP_ROM", "EXP_free_face"]

    (
        best_screen_loss,
        best_tex_loss,
        best_vert_loss,
        screen_loss,
        tex_loss,
        vert_loss,
        train_screen_loss,
        train_tex_loss,
        train_vert_loss,
    ) = main(experiment_args, camera_set, test_segment)
    if torch.distributed.get_rank() == 0:
        print(
            best_screen_loss,
            best_tex_loss,
            best_vert_loss,
            screen_loss,
            tex_loss,
            vert_loss,
            train_screen_loss,
            train_tex_loss,
            train_vert_loss,
        )
        f = open(os.path.join(experiment_args.result_path, "result.txt"), "a")
        f.write("\n")
        f.write(
            "Best screen loss %f, best tex loss %f,  best vert loss %f, screen loss %f, tex loss %f, vert_loss %f, train screen loss %f, train tex loss %f, train vert loss %f"
            % (
                best_screen_loss,
                best_tex_loss,
                best_vert_loss,
                screen_loss,
                tex_loss,
                vert_loss,
                train_screen_loss,
                train_tex_loss,
                train_vert_loss,
            )
        )
        f.close()
