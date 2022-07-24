# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# obj file dataset
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy.lib.format import MAGIC_PREFIX
from PIL import Image
from torch.autograd import Variable


def load_obj(filename):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
    # make sure triangle ids are 0 indexed
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    return obj


def check_path(path):
    if not os.path.exists(path):
        sys.stderr.write("%s does not exist!\n" % (path))
        sys.exit(-1)


def load_krt(path):
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "intrin": np.array(intrin),
                "dist": np.array(dist),
                "extrin": np.array(extrin),
            }

    return cameras


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir,
        krt_dir,
        framelistpath,
        size=1024,
        camset=None,
        valid_prefix=None,
        exclude_prefix=None,
    ):
        self.uvpath = "{}/unwrapped_uv_1024".format(base_dir)
        self.meshpath = "{}/tracked_mesh".format(base_dir)
        self.photopath = "{}/images".format(base_dir)
        self.size = size
        self.camera_ids = {}

        check_path(self.uvpath)
        check_path(self.meshpath)
        check_path(framelistpath)

        framelist = np.genfromtxt(framelistpath, dtype=np.str)
        self.mesh_topology = None

        # set cameras
        krt = load_krt(krt_dir)
        self.krt = krt
        self.cameras = list(krt.keys())
        for i, k in enumerate(self.cameras):
            self.camera_ids[k] = i

        if camset is not None:
            self.cameras = camset
        self.allcameras = sorted(self.cameras)

        # load train list (but check that images are not dropped!)
        self.framelist = []

        for i, x in enumerate(framelist):
            if i % 1000 == 0:
                print("checking {}".format(i))

            # filter valid prefixes
            if valid_prefix is not None:
                valid = False
                for p in valid_prefix:
                    if x[0].startswith(p):
                        valid = True
                        break
                if not valid:
                    continue

            if exclude_prefix is not None:
                valid = True
                for p in exclude_prefix:
                    if x[0].startswith(p):
                        valid = False
                        break
                if not valid:
                    continue

            # check if has average texture
            avgf = "{}/{}/average/{}.png".format(self.uvpath, x[0], x[1])

            if os.path.isfile(avgf) is not True:
                continue
            # check if has per-view uvwrap
            for i, cam in enumerate(self.cameras):
                f = tuple(x) + (cam,)
                path = "{}/{}/{}/{}.png".format(self.uvpath, f[0], f[2], f[1])
                if os.path.isfile(path) is True:
                    self.framelist.append(f)

        # compute view directions of each camera
        campos = {}
        for cam in self.cameras:
            extrin = krt[cam]["extrin"]
            campos[cam] = -np.dot(extrin[:3, :3].T, extrin[:3, 3])
        self.campos = campos

        # load mean image and std
        texmean = np.asarray(
            Image.open("{}/tex_mean.png".format(base_dir)), dtype=np.float32
        )
        self.texmean = np.copy(np.flip(texmean, 0))
        self.texstd = float(np.genfromtxt("{}/tex_var.txt".format(base_dir)) ** 0.5)
        self.texmin = (
            np.zeros_like(self.texmean, dtype=np.float32) - self.texmean
        ) / self.texstd
        self.texmax = (
            np.ones_like(self.texmean, dtype=np.float32) * 255 - self.texmean
        ) / self.texstd

        self.vertmean = np.fromfile(
            "{}/vert_mean.bin".format(base_dir), dtype=np.float32
        )
        self.vertstd = float(np.genfromtxt("{}/vert_var.txt".format(base_dir)) ** 0.5)

    def __len__(self):
        return len(self.framelist)

    def __getitem__(self, idx):
        sentnum, frame, cam = self.framelist[idx]
        cam_id = self.camera_ids[cam]

        # geometry
        if self.mesh_topology is None:
            path = "{}/{}/{}.obj".format(self.meshpath, sentnum, frame)
            obj = load_obj(path)
            self.mesh_topology = obj

        # geometry
        path = "{}/{}/{}.bin".format(self.meshpath, sentnum, frame)
        verts = np.fromfile(path, dtype=np.float32)
        verts -= self.vertmean
        verts /= self.vertstd

        # average image
        path = "{}/{}/average/{}.png".format(self.uvpath, sentnum, frame)
        avgtex = np.asarray(Image.open(path), dtype=np.float32)[::-1, ...]
        mask = avgtex == 0
        avgtex -= self.texmean
        avgtex /= self.texstd
        avgtex[mask] = 0.0
        avgtex = cv2.resize(avgtex, (self.size, self.size)).transpose((2, 0, 1))

        # image
        path = "{}/{}/{}/{}.png".format(self.photopath, sentnum, cam, frame)
        photo = np.asarray(Image.open(path), dtype=np.float32)
        photo = photo / 255.0

        # texture
        path = "{}/{}/{}/{}.png".format(self.uvpath, sentnum, cam, frame)
        tex = np.asarray(Image.open(path), dtype=np.float32)[::-1, ...]
        mask = tex == 0
        tex -= self.texmean
        tex /= self.texstd
        tex[mask] = 0.0
        tex = cv2.resize(tex, (self.size, self.size)).transpose((2, 0, 1))
        mask = 1.0 - cv2.resize(
            mask.astype(np.float32), (self.size, self.size)
        ).transpose((2, 0, 1))

        # view direction
        transf = np.genfromtxt(
            "{}/{}/{}_transform.txt".format(self.meshpath, sentnum, frame)
        )
        R_f = transf[:3, :3]
        t_f = transf[:3, 3]
        campos = np.dot(R_f.T, self.campos[cam] - t_f).astype(np.float32)
        view = campos / np.linalg.norm(campos)

        extrin, intrin = self.krt[cam]["extrin"], self.krt[cam]["intrin"]
        R_C = extrin[:3, :3]
        t_C = extrin[:3, 3]
        camrot = np.dot(R_C, R_f).astype(np.float32)
        camt = np.dot(R_C, t_f) + t_C
        camt = camt.astype(np.float32)

        M = intrin @ np.hstack((camrot, camt[None].T))

        return {
            "cam_idx": cam,
            "frame": frame,
            "exp": sentnum,
            "cam": cam_id,
            "M": M.astype(np.float32),
            "uvs": self.mesh_topology["uvs"],
            "vert_ids": self.mesh_topology["vert_ids"],
            "uv_ids": self.mesh_topology["uv_ids"],
            "avg_tex": avgtex,
            "mask": mask,
            "tex": tex,
            "view": view,
            "transf": transf.astype(np.float32),
            "aligned_verts": verts.reshape((-1, 3)).astype(np.float32),
            "photo": photo,
        }
