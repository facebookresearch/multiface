# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dataset import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models import WarpFieldVAE, DeepAppearanceVAE
from utils import Renderer
import cv2
import numpy as np
import os
# from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import torch.nn.functional as F
import json
import argparse
from collections import OrderedDict


def main(args, camera_config, test_segment):
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    dataset_train = Dataset(args.data_dir, args.krt_dir, args.framelist_test, args.tex_size,
                        camset=None if camera_config is None else camera_config['train'],
                        exclude_prefix=test_segment)

    dataset_test = Dataset(args.data_dir, args.krt_dir, args.framelist_test, args.tex_size,
                            camset=None if camera_config is None else camera_config['test'],
                            valid_prefix=test_segment)

    dataset_visual = Dataset(args.data_dir, args.krt_dir, args.framelist_test, args.tex_size,
                            camset=None if camera_config is None else camera_config['visual'],
                            valid_prefix=test_segment)


    visual_sampler = DistributedSampler(dataset_visual)

    visual_loader = DataLoader(dataset_visual, args.val_batch_size, sampler=visual_sampler, num_workers=args.n_worker)

    if local_rank == 0:
        print('#visual samples', len(dataset_visual))
        # writer = SummaryWriter(log_dir=args.result_path)

    n_cams = len(dataset_train.cameras) + len(dataset_test.cameras)
    if args.arch == 'base':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams).to(device)
    elif args.arch == 'res':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, res=True, n_cams=n_cams).to(device)
    elif args.arch == 'warp':
        model = WarpFieldVAE(args.tex_size, args.mesh_inp_size, z_dim=args.nlatent, n_cams=n_cams).to(device)
    elif args.arch == 'non':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, res=False, non=True, n_cams=n_cams).to(device)
    elif args.arch == 'bilinear':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, res=False, non=False, bilinear=True, n_cams=n_cams).to(device)
    else:
        raise NotImplementedError

    # by default load the best_model.pth
    print('loading model from', args.model_path)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    state_dict = torch.load(args.model_path, map_location=map_location)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)


    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)
    renderer = Renderer()

    optimizer_cc = optim.Adam(model.module.get_cc_params(), args.lr, (0.9, 0.999))
    optimizer_enc = optim.Adam(model.module.enc.parameters(), args.lr, (0.9, 0.999))
    mse = nn.MSELoss()

    texmean = cv2.resize(dataset_test.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_test.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_test.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = dataset_test.texstd
    vertmean = torch.tensor(dataset_test.vertmean, dtype=torch.float32).view((1, -1, 3)).to(device)
    vertstd = dataset_test.vertstd
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)

    os.makedirs(args.result_path, exist_ok=True)

    def run_net(data):
        M = data['M'].cuda()
        gt_tex = data['tex'].cuda()
        vert_ids = data['vert_ids'].cuda()
        uvs = data['uvs'].cuda()
        uv_ids = data['uv_ids'].cuda()
        avg_tex = data['avg_tex'].cuda()
        view = data['view'].cuda()
        transf = data['transf'].cuda()
        verts = data['aligned_verts'].cuda()
        photo = data['photo'].cuda()
        mask = data['mask'].cuda()
        cams = data['cam'].cuda()
        batch, channel, height, width = avg_tex.shape

        output = {}

        if args.arch == 'warp':
            pred_tex, pred_verts, unwarped_tex, warp_field, kl = model(avg_tex, verts, view, cams=cams)
            output['unwarped_tex'] = unwarped_tex
            output['warp_field'] = warp_field
        else:
            pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams)
        vert_loss = mse(pred_verts, verts)

        pred_verts = pred_verts * vertstd + vertmean
        pred_tex = (pred_tex * texstd + texmean) / 255.
        gt_tex = (gt_tex * texstd + texmean) / 255.

        loss_mask = loss_weight_mask.repeat(batch, 1, 1, 1)
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)

        if args.lambda_screen > 0:
            screen_mask, rast_out = renderer.render(M, pred_verts, vert_ids, uvs, uv_ids, loss_mask, args.resolution)
            pred_screen, rast_out = renderer.render(M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, args.resolution)
            screen_loss = torch.mean((pred_screen - photo)**2 * screen_mask) * (255**2) / (texstd**2)
            data['screen_mask'] = screen_mask
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
            'total_loss': total_loss,
            'vert_loss': vert_loss,
            'screen_loss': screen_loss,
            'tex_loss': tex_loss,
            'kl': kl
        }

        output['pred_screen'] = pred_screen
        output['pred_verts'] = pred_verts
        output['pred_tex'] = pred_tex

        return losses, output

    def save_img(data, output, i, key, tag=''):
        screen_mask = data['screen_mask'][i].detach().cpu()
        gt_screen = data['photo'][i] * 255
        #gt_tex = data['tex'][i].cuda() * texstd + texmean
        #pred_tex = torch.clamp(output['pred_tex'][i] * 255, 0, 255)
        if output['pred_screen'][i] is not None:
            pred_screen = torch.clamp(output['pred_screen'][i] * 255, 0, 255)
            Image.fromarray(pred_screen.detach().cpu().numpy().astype(np.uint8)).save(os.path.join(args.result_path, 'pred_%s.png' % tag))
        Image.fromarray(gt_screen.detach().cpu().numpy().astype(np.uint8)).save(os.path.join(args.result_path, 'gt_%s.png' % tag))
        #Image.fromarray(gt_tex[-1].detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)).save(os.path.join(args.result_path, 'gt_tex_%s.png' % tag))
        #Image.fromarray(pred_tex.detach().permute((1, 2, 0)).cpu().numpy().astype(np.uint8)).save(os.path.join(args.result_path, 'pred_tex_%s.png' % tag))

        # compute difference
        diff_img = abs(pred_screen.detach().cpu().numpy() - (gt_screen * screen_mask).detach().cpu().numpy()).astype(np.uint8)
        diff_img  = cv2.normalize(diff_img, diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        diff_img = cv2.applyColorMap(np.uint8(255 * (255 - diff_img)), cv2.COLORMAP_JET)[:, :, ::-1]
        Image.fromarray(diff_img).save(os.path.join(args.result_path, 'diff_%s.png' % tag))

        if key not in pred_imgs:
            pred_imgs[key] = []
        if key not in gt_imgs:
            gt_imgs[key] = []
        if key not in diff_imgs:
            diff_imgs[key] = []

        pred_imgs[key].append(os.path.join(args.result_path, 'pred_%s.png' % tag))
        gt_imgs[key].append(os.path.join(args.result_path, 'gt_%s.png' % tag))
        diff_imgs[key].append(os.path.join(args.result_path, 'diff_%s.png' % tag))

    def save_video(key):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = 20
        videowrite_path = os.path.join(args.result_path, "%s.avi" % (key))
        cur_gt_img = gt_imgs[key].copy()
        cur_pred_img = pred_imgs[key].copy()
        cur_diff_img = diff_imgs[key].copy()
        cur_gt_img = sorted(cur_gt_img)
        cur_pred_img = sorted(cur_pred_img)
        cur_diff_img = sorted(cur_diff_img)

        assert(len(cur_gt_img) == len(cur_pred_img))
        assert(len(cur_diff_img) == len(cur_pred_img))

        for idx, img in enumerate(cur_gt_img):
            gt = cv2.imread(img)
            pred = cv2.imread(cur_pred_img[idx])
            diff = cv2.imread(cur_diff_img[idx])
            cv2.putText(gt, "GT", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(pred, "PRED", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(diff, "DIFF", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            full_img = np.hstack((gt, pred, diff)) #combine horizontally
            h, w = full_img.shape[:2]
            title =  np.zeros((100, w, 3), dtype = "uint8")
            cv2.putText(title, key, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            full_img = np.vstack((title, full_img))
            h, w = full_img.shape[:2]
            if idx == 0:
                videowrite = cv2.VideoWriter(videowrite_path, fourcc, fps, (w, h))
            videowrite.write(full_img)

        videowrite.release()

    val_idx = 0
    best_screen_loss = 1e8
    best_tex_loss = 1e8
    best_vert_loss = 1e8
    model.train()

    model.eval()
    iter = 1
    begin_time = time.time()

    gt_imgs = {}
    pred_imgs = {}
    diff_imgs = {}
    test_segment_full_name = set()

    for j in range(iter):
        total, vert, tex, screen, kl = [], [], [], [], []
        for i, data in enumerate(visual_loader):
            losses, output = run_net(data)
            optimizer_cc.zero_grad()
            optimizer_enc.zero_grad()
            total.append(losses['total_loss'].item())
            vert.append(losses['vert_loss'].item())
            tex.append(losses['tex_loss'].item())
            screen.append(losses['screen_loss'].item())
            kl.append(losses['kl'].item())
            losses['total_loss'].backward()
            optimizer_cc.step()
            optimizer_enc.step()
            if i == args.val_num and j != (iter - 1):
                break
            if j == (iter - 1) and local_rank == 0:
                # need to process one by one
                for k in range(args.val_batch_size):
                    if str(data['exp'][k]) not in test_segment_full_name:
                        test_segment_full_name.add(str(data['exp'][k]))
                    save_img(data, output, k, "%s_%s" % (str(data['exp'][k]), str(data['cam_idx'][k])),'val_%s_%s_%s' % (str(data['exp'][k]), str(data['cam_idx'][k]), str(data['frame'][k])))
                if i > 1:
                    break
    for cam in camera_config['visual']:
        for exp in list(test_segment_full_name):
            key = str(exp) + "_" + str(cam)
            save_video(key)

    tex_loss = np.array(tex).mean()
    vert_loss = np.array(vert).mean()
    screen_loss = np.array(screen).mean()
    kl = np.array(kl).mean()


    if local_rank == 0:
        pass
        # writer.add_scalar('val/loss_tex',losses['tex_loss'].item(), val_idx)
        # writer.add_scalar('val/loss_verts', losses['vert_loss'].item(), val_idx)
        # writer.add_scalar('val/loss_screen', losses['screen_loss'].item(), val_idx)
        # writer.add_scalar('val/loss_kl', losses['kl'].item(), val_idx)


    val_idx += 1
    print('val %d vert %.3f tex %.3f screen %.5f kl %.3f' %
        (val_idx, vert_loss, tex_loss, screen_loss, kl))

    best_screen_loss = min(best_screen_loss, screen_loss)
    best_tex_loss = min(best_tex_loss, tex_loss)
    best_vert_loss = min(best_vert_loss, vert_loss)

    end_time = time.time()
    print('Testing takes %f seconds' % (end_time - begin_time))
    print('best screen loss %f, best tex loss %f best vert loss %f' % (best_screen_loss, best_tex_loss, best_vert_loss))
    return best_screen_loss, best_tex_loss, best_vert_loss, screen_loss, tex_loss, vert_loss


if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed run')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Validation batch size')
    parser.add_argument('--arch', type=str, default='base', help='Model architecture - base|warp|res|non|bilinear')
    parser.add_argument('--nlatent', type=int, default=256, help='Latent code dimension - 128|256')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--resolution', default=[2048, 1334], nargs=2, type=int, help='Rendering resolution')
    parser.add_argument('--tex_size', type=int, default=1024, help='Texture resolution')
    parser.add_argument('--mesh_inp_size', type=int, default=21918, help='Input mesh dimension')
    parser.add_argument('--data_dir', type=str, default='/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS', help='Directory to dataset root')
    parser.add_argument('--krt_dir', type=str, default='/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/KRT', help='Directory to KRT file')
    parser.add_argument('--loss_weight_mask', type=str, default='./loss_weight_mask.png', help='Mask for weighted loss of face')
    parser.add_argument('--framelist_test', type=str, default='/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt', help='Frame list for testing')
    parser.add_argument('--test_segment_config', type=str, default='/mnt/captures/ecwuu/test_segment.json', help='Directory of expression segments for visualization')
    parser.add_argument('--lambda_verts', type=float, default=1, help='Multiplier of vertex loss')
    parser.add_argument('--lambda_screen', type=float, default=1, help='Multiplier of screen loss')
    parser.add_argument('--lambda_tex', type=float, default=1, help='Multiplier of texture loss')
    parser.add_argument('--lambda_kl', type=float, default=1e-2, help='Multiplier of KL divergence')
    parser.add_argument('--max_iter', type=int, default=200000, help='Maximum number of training iterations, overrides epoch')
    parser.add_argument('--log_every', type=int, default=1000, help='Interval of printing training loss')
    parser.add_argument('--val_every', type=int, default=5000, help='Interval of validating on test set')
    parser.add_argument('--val_num', type=int, default=500, help='Number of iterations for validation')
    parser.add_argument('--n_worker', type=int, default=8, help='Number of workers loading dataset')
    parser.add_argument('--pass_thres', type=int, default=50, help='If loss is x times higher than the previous batch, discard this batch')
    parser.add_argument('--result_path', type=str, default='./runs/experiment', help='Directory to output files')
    parser.add_argument('--camera_config', type=str, default=None, help='Directory to camera set config file')
    parser.add_argument('--camera_setting', type=str, default=None, help='Key of camera setting to camera config file')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    parser.add_argument('--save_video', type=bool, default=True, help='Save visualization as .mp4')

    experiment_args = parser.parse_args()
    print(experiment_args)
    assert(experiment_args.camera_config != None)
    assert(experiment_args.test_segment_config != None)

    if experiment_args.camera_config is not None:
        f = open(experiment_args.camera_config, 'r')
        camera_config = json.load(f)
        f.close()
        if experiment_args.camera_setting is not None:
            camera_set = camera_config[experiment_args.camera_setting]
        else:
            camera_set = None
    else:
        camera_set = None

    f = open(experiment_args.test_segment_config, 'r')
    test_segment_config = json.load(f)
    f.close()
    test_segment = test_segment_config["segment"]


    main(experiment_args, camera_set, test_segment)
    print("visualization completed")
