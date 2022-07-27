# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import multiprocessing as mp
import os

import requests
from bs4 import BeautifulSoup

MAX_TRY = 50
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dataset Download")


def worker(path):
    if path[-4:] == ".tar":
        try_count = 0
        while True and (try_count < MAX_TRY):
            try_count = try_count + 1
            os.system(
                "tar -xvf %s -C %s && touch %s.unzip"
                % (path, "/".join(path.split("/")[:-1]), path)
            )
            if os.path.isfile("%s.unzip" % (path)):
                os.system("rm " + path)
                break
            else:
                logging.info("Unzipped %s failed. Re-unzipping..." % (path))

        logging.info("Done %s" % (path))


def unzip(unzip_tar):
    """create pool to extract all"""
    pool = mp.Pool(min(mp.cpu_count(), len(unzip_tar)))  # number of workers
    pool.map(worker, unzip_tar)
    pool.close()


def checksum(tar_files, file_root, entity, checksum_file):
    tar_files = set(tar_files)
    excluded_tar_files = []
    checksum_file = open("%s" % (checksum_file), "r")
    lines = checksum_file.readlines()

    for line in lines:
        code = line.split(" ")[0].strip()
        tar_name = line.split(" ")[-1].strip()
        tar_file = os.path.join(file_root, entity + tar_name)
        if tar_file not in tar_files:
            continue
        try:
            file = open(tar_file)
        except Exception as e:
            logging.info("File %s not found! Recheck downloading process!" % (tar_file))
        try_count = 0
        while True and (try_count < MAX_TRY):
            try_count = try_count + 1
            cmd = "md5sum %s " % (tar_file)
            os.system("%s > tmp" % (cmd))
            if open("tmp", "r").read().split(" ")[0] != str(code):
                print(open("tmp", "r").read().split(" ")[0], "  ", str(code))
                excluded_tar_files.append(tar_file)
                logging.info("File %s does not pass checksum!" % (tar_file))
            else:
                os.system("touch %s.checksum" % (tar_file))
            os.remove("tmp")
            if os.path.isfile("%s.checksum" % (tar_file)):
                logging.info("File %s PASS checksum!" % (tar_file))
                break
    return excluded_tar_files


def download_tar(
    download_dest,
    entity,
    download_img,
    download_tex,
    download_mesh,
    download_audio,
    download_metadata,
    expression,
):
    """download all the selected files"""

    # extract urls
    misc = set(["CHECKSUM", "index.html"])
    root = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/v0.0/identities/"
    entity_urls = []
    for e in entity:
        entity_urls.append(root + e + "/index.html")

    for url in entity_urls:
        entity = url.split("/")[-2]
        logging.info("Start downloading entity %s...." % (entity))
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, "html.parser")
        tar_files = []
        checksum_file = None

        for link in soup.find_all("a"):
            file_name = link.get("href").split("/")[-1]
            if "unwrapped_uv" in file_name and not download_tex:
                continue
            if "tracked_mesh" in file_name and not download_mesh:
                continue
            if "images" in file_name and not download_img:
                continue
            if "audio" in file_name and not download_audio:
                continue
            if "metadata" in file_name and not download_metadata:
                continue

            included_file = False
            if file_name in misc or "metadata" in file_name or "audio" in file_name:
                included_file = True
            else:
                for exp in expression:
                    if exp in file_name:
                        included_file = True
                        break

            if included_file is False:
                continue

            file_path = os.path.join(download_dest, entity + file_name)
            try_count = 0
            while True and (try_count < MAX_TRY):
                try_count = try_count + 1
                cmd = "wget -O %s %s && touch %s.download" % (
                    file_path,
                    link.get("href"),
                    file_path,
                )
                os.system(cmd)
                if os.path.isfile("%s.download" % (file_path)):
                    break
                else:
                    logging.info(
                        "Downloaded %s failed. Re-downloading..." % (file_path)
                    )

            if "CHECKSUM" in file_name:
                checksum_file = file_path

            tar_files.append(file_path)

        # check CHECKSUM
        excluded_tar_files = checksum(tar_files, download_dest, entity, checksum_file)
        logging.info("%s checksum has completed" % (entity))

        # unzip tar
        unzip_tar = [f for f in tar_files if f not in excluded_tar_files]
        unzip(unzip_tar)

    return True


def main(args):

    f = open(args.download_config, "r")
    download_config = json.load(f)
    f.close()

    download_path = args.dest

    entity = download_config["entity"]
    download_img = download_config["image"]
    download_tex = download_config["texture"]
    download_mesh = download_config["mesh"]
    download_audio = download_config["audio"]
    download_metadata = download_config["metadata"]
    expression = download_config["expression"]

    if download_tar(
        download_dest=download_path,
        entity=entity,
        download_img=download_img,
        download_tex=download_tex,
        download_mesh=download_mesh,
        download_audio=download_audio,
        download_metadata=download_metadata,
        expression=expression,
    ):
        logging.info("%s .tar extraction has completed" % (entity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        metavar="Download Destimation",
        type=str,
        required=False,
        help="Directory of data to be downloaded",
        default="./",
    )
    parser.add_argument(
        "--download_config",
        metavar="Download Config File",
        type=str,
        required=False,
        help="File path of download_config file",
        default="./download_config.json",
    )

    args = parser.parse_args()
    main(args)
