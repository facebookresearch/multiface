# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nvdiffrast batched rendering
import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from dataset import Dataset
from PIL import Image


class Renderer:
    def __init__(self):
        self.glctx = dr.RasterizeGLContext()

    def render(self, M, pos, pos_idx, uv, uv_idx, tex, resolution=[2048, 1334]):
        ones = torch.ones((pos.shape[0], pos.shape[1], 1)).to(pos.device)
        pos_homo = torch.cat((pos, ones), -1)
        projected = torch.bmm(M, pos_homo.permute(0, 2, 1))
        projected = projected.permute(0, 2, 1)
        proj = torch.zeros_like(projected)
        proj[..., 0] = (
            projected[..., 0] / (resolution[1] / 2) - projected[..., 2]
        ) / projected[..., 2]
        proj[..., 1] = (
            projected[..., 1] / (resolution[0] / 2) - projected[..., 2]
        ) / projected[..., 2]
        clip_space, _ = torch.max(projected[..., 2], 1, keepdim=True)
        proj[..., 2] = projected[..., 2] / clip_space

        pos_view = torch.cat(
            (proj, torch.ones(proj.shape[0], proj.shape[1], 1).to(proj.device)), -1
        )
        pos_idx_flat = pos_idx.view((-1, 3)).contiguous()
        uv_idx = uv_idx.view((-1, 3)).contiguous()
        tex = tex.permute((0, 2, 3, 1)).contiguous()

        rast_out, rast_out_db = dr.rasterize(
            self.glctx, pos_view, pos_idx_flat, resolution
        )
        texc, _ = dr.interpolate(uv, rast_out, uv_idx)
        color = dr.texture(tex, texc, filter_mode="linear")
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background.
        return color, rast_out
