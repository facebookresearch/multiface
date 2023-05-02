import os
import multiprocessing as mp
import glob
import logging
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

if __name__ == "__main__":
    ids = ["5067077"]
    for i in ids:
        #hardcode path
        unzip_tar = sorted(glob.glob(f"/mnt/captures/xuhuahuang/workshop/{i}*.tar"))
        unzip(unzip_tar)
        logging.info("Unzip all tar of id %s" % (i))
