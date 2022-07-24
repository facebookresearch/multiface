# Installation
The provided code is in PyTorch with support of `DistributedDataParallel`. This repository requires Nvdiffrast and CUDA for differentiable rendering.

1. For installation and more information about Nvdiffrast, you may refer to their [document](https://nvlabs.github.io/nvdiffrast/). We recommend using with [Docker](https://www.docker.com/) as in their tutorial, and we also provide script for launching with docker.

2. To install Docker, visit \
[https://www.docker.com/get-started](https://www.docker.com/get-started/)

3. To clone this repository and Nvdiffrast repositor, run 
```
git clone https://github.com/facebookresearch/multiface
git clone https://github.com/NVlabs/nvdiffrast
```

4. To build docker image with nvdiffrast and the required python packages in this repository, one can modify the Dockerfile in nvdiffrast or simply swap in our provided Dockerfile
```
mv nvdiffrast/docker/Dockerfile nvdiffrast/docker/Dockerfile.original
cp multiface/Dockerfile nvdiffrast/docker/Dockerfile
cd nvdiffrast
chmod +x run_sample.sh
./run_sample.sh --build-container
cd ..
```

Alternatively, one can install in local environment if all the CUDA dependencies are taken care of using 
```
cd nvdiffrast 
pip install .
```
However, we did NOT test instllation on this method, thus would strongly recommend using option 4.1. 

5. To download the mini-dataset (for sanity check), run
```
pip3 install -r requirements.txt
python3 download_dataset.py --dest "/path/to/mini_dataset/" --download_config "./mini_download_config.json"```
```
This will download 2 expressions (images, textures and mesh) from enity ```6795937``` under directory ```/path/to/mini_dataset/```.\
In the last part of installation, we will instruct how to donwload the entire dataset (400TB).
<br/>
<br/>
6. You can run the training script in docker using command: 
```cd multiface
docker run --rm -it --gpus all --user $(id -u):$(id -g) -v `pwd`:/app -v /path/to/mini_dataset/:/dataset --workdir /app --shm-size 256g TORCH_EXTENSIONS_DIR=/app/tmp gltorch:latest python -m torch.distributed.launch --nproc_per_node=1 train.py --data_dir /path/to/mini_dataset/m--20180227--0000--6795937--GHS --krt_dir /path/to/mini_dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /path/to/mini_dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /path/to/mini_dataset/m--20180227--0000--6795937--GHS/frame_list.txt --test_segment "./mini_test_segment.json --lambda_screen 1"
```
Assuming you have downloaded and put the dataset on your local directory `/path/to/mini_dataset`. Running the above command sets the shm memory size to 256GB, and uses 1 gpus as specified by `nproc_per_node` argument. It will launch training on the identity `m--20180227--0000--6795937--GHS` (```6795937```).

If you have installed Nvdiffrast to your system you can simply launch the training script by:

```
python -m torch.distributed.launch --nproc_per_node=1 train.py --data_dir /path/to/mini_dataset/m--20180227--0000--6795937--GHS --krt_dir /path/to/mini_dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /path/to/mini_dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test "./mini_frame_list.txt"  --result_path "./mini_dataset" --test_segment "./mini_test_segment.json" --lambda_screen 1 --model_path "./mini_dataset/best_model.pth"
```
And the testing script by:
```
python -m torch.distributed.launch --nproc_per_node=1 test.py --data_dir /path/to/mini_dataset/m--20180227--0000--6795937--GHS --krt_dir /path/to/mini_dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /path/to/mini_dataset/m--20180227--0000--6795937--GHS/frame_list.txt --test_segment "./mini_test_segment.json"
``` 

We provide the pretrained model [pretrained model](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/mini.pth) on this minidataset. You can check your training result by comparing training and testing loss using split of ```mini_test_segment.json```:
| Training Loss (screen/ mesh / tex)  | Testing Loss (screen / mesh / tex)|
| ------------- |:-------------:|
|   0.268333 / 0.769636 / 0.654152  | 0.154118 / 0.758897 / 0.379572  | 
  
<br/>

  
  
7. To donwload the <b>ENTIRE</b> dataset (13 entities), you will need ~120 TB during downloading and ~65TB for storage in total. Size of images, mesh and textures of each entity is provided:
  
| Entity ID   | Name of Root Folder | Images (TB)    |  Mesh (GB) | Textures (GB)|
| -------------|:-------------:|:-------------:|:-------------:|:-------------:|
| 002645310    | m--20190828--1318--002645310--GHS  |7.8 | 45 | 2400 |
| 002643814   |m--20180426--0000--002643814--GHS     | 1.6 | 21 | 412 |
| 002539136  |m--20180105--0000--002539136--GHS    |   1.5   |   20 | 381 |
| 7889059    |m--20180927--0000--7889059--GHS     |   1.8 | 29 | 493    |   
| 002757580   | m--20171024--0000--002757580--GHS      |    1.4      |    21 | 380 |
|2183941      |m--20180418--0000--2183941--GHS     |  2.1  |  32 | 577 |
| 5372021     |m--20180510--0000--5372021--GHS    |  2.0 |27 |513   |  
|8870559     |m--20180406--0000--8870559--GHS   |  2.4  | 32 | 611 | 
| 6674443    |m--20180226--0000--6674443--GHS    |  1.7 | 26 | 429    |  
| 5067077    |m--20190529--1004--5067077--GHS    |  11   |  49 | 3100|
| 002914589     |m--20181017--0000--002914589--GHS     |  1.1 | 21 | 367   |  
| 6795937    |m--20180227--0000--6795937--GHS|  1.9 | 27 | 521|  
|002421669  |m--20190529--1300--002421669--GHS     | 13   |  51| 3100|
| Total | - | 49.3|401|13284|



  Run
```python3 download_dataset.py --dest "path/to/dataset" --download_config "./download_config.json"``` 
  
You may refer to [DATA_STRUCTURE.md](DATA_STRUCTURE.md) to check the completness of data for each entity under its root folder.

You may select the directory of data to be downloaded by specifying ```--dest```. To make the download flexiable, you may use ```download_config.json``` (specify by ```--download_config```) to select the data to be downloaded with the following variables: 

| Variable        | Type          | Default  |
| ------------- |:-------------:| -----|
| entity     | list of string | all the entity will be downloaded |
| image     | boolean      |  raw images of enities selected will be downloaded|
| mesh |boolean      |    tracked mesh of enities selected will be downloaded |
| texture |boolean      |    unwrapped texture of enities selected will be downloaded |
| metadata |boolean      |    metadata of enities selected will be downloaded |
| audio |boolean      |    audio of enities selected will be downloaded |
| expression |list of string     |     all the facial expression (contains both v1 and v2 scripts) will be downloaded  |


Notice that 
* audio is NOT a necessity to train a deep appearance model. 
* download the entire dataset is time- and space-consumming, we STRONGLY recommend users to download the mini-dataset (specifying ```--download_config "./path/to/mini_download_config.json"```) to check the flow works property first.
* entities are downloaded one-by-one, which means that all the ```.tar``` files will be deleted after files are unzipped. 

  <br/>
  
8. We provide pretrained models for each identity except ```002539136``` and ```002757580```. The model is a simple VAE model that reconstruct mesh and view-specific texture. It is trained using ground-truth mesh and texture only. For entities ``` 002643814, 7889059, 2183941, 5372021, 8870559, 6674443, 002914589, 6795937``` (V1), we use expression ```EXP_ROM07_Facial_Expressions``` as the testing set, and for entites ```002645310, 5067077, 002421669``` (V2), we use expression ```EXP_free_face``` as the testing set.

| Entity ID       | Pretrained Model        | Script | Testing Loss (mesh / tex)|
| :-------------: |:-----------------------:|:------:|:------------------------:|
| 002645310    |  [002645310_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/002645310_model.pth)| [training](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/train_002645310.sh) / [testing](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/test_002645310.sh) | 0.0403 / 0.1928 |
| 002643814   | [002643814_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/002643814_model.pth)| [training](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/train_002643814.sh) / [testing](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/test_002643814.sh) |  0.0373 / 0.1376|
| 7889059|[7889059_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/7889059_model.pth)| [training](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/train_7889059.sh) / [testing](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/test_7889059.sh) | 0.0242 / 0.1133|
| 5372021 |[5372021_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/5372021_model.pth)| [training](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/train_5372021.sh) / [testing](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/test_5372021.sh) | 0.0580 / 0.1894|
|2183941 |[2183941_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/2183941_model.pt)|[training](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/train_2183941.sh) / [testing](https://github.com/facebookresearch/multiface/blob/main/pretrained_model/script/test_2183941.sh) | 0.0248 / 0.1127|
|8870559 |[8870559_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/8870559_model.pth)|[training]() / [testing]() | 0.0222 / 0.1166 |
|6674443 |TBA| |
|5067077 |TBA| |
|002914589 |[002914589_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/002914589_model.pth)|[training]() / [testing]() | 0.0341 / 0.1411|
|6795937 |[6795937_model.pth](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/6795937_model.pth)| [training]() / [testing]() | 0.0254 / 0.1120 |
|002421669 |TBA| |

