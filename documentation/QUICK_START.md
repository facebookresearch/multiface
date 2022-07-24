# Quick Start
### Visualization
Users may visualize the specific segment and views for an entity by running ```visualie.py```
For example, you will get the following results by running
```
python -m torch.distributed.launch --nproc_per_node=1 visualize.py --data_dir "/path/to/dataset/m--20180418--0000--2183941--GHS" --krt_dir "/path/to/dataset/m--20180418--0000--2183941--GHS/KRT" --framelist "/path/to/dataset/m--20180418--0000--2183941--GHS/frame_list.txt"  --result_path "./visual" --test_segment "./test_segment.json" --lambda_screen 1 --model_path "./m--20180418--0000--2183941--GHS/best_model.pth"  --camera_config "./camera-split-config.json" --camera_setting "full" 
```
<p align="center">
<img src="https://github.com/facebookresearch/multiface/blob/main/images/EXP-ROM07-Facial-Expressions-400.gif?raw=true" width="700" height="450" />
</p>

### Model Architecture
Users can select 5 different model architectures by specifying the `arch` argument while calling the both training and testing script. In the codes, it will construct the corresponding model architectures.

```python
 if args['arch'] == 'base':
        model = DeepAppearanceVAE(args['tex_size'], args['mesh_inp_size'], n_latent=args['nlatent']).to(device)
    elif args['arch'] == 'res':
        model = DeepAppearanceVAE(args['tex_size'], args['mesh_inp_size'], n_latent=args['nlatent'], res=True).to(device)
    elif args['arch'] == 'warp':
        model = WarpFieldVAE(args['tex_size'], args['mesh_inp_size'], z_dim=args['nlatent']).to(device)
    elif args['arch'] == 'non':
        model = DeepAppearanceVAE(args['tex_size'], args['mesh_inp_size'], n_latent=args['nlatent'], res=False, non=True).to(device)
    elif args['arch'] == 'bilinear':
        model = DeepAppearanceVAE(args['tex_size'], args['mesh_inp_size'], n_latent=args['nlatent'], res=False, non=False, bilinear=True).to(device)
    else:
        raise NotImplementedError
```

For example, if you want to train a model with Warp Field architecture, you may use:<br/>
`python -m torch.distributed.launch --nproc_per_node=4 train.py --data_dir /path/to/dataset/m--20180426--0000--002643814--GHS --krt_dir /path/to/dataset/m--20180426--0000--002643814--GHS/KRT --framelist_train /path/to/dataset/m--20180426--0000--002643814--GHS/frame_list.txt --framelist_test /path/to/dataset/m--20180426--0000--002643814--GHS/frame_list.txt --test_segment "./test_segment.json" --camera_config "./camera-split-config.json" --camera_setting "full" --arch "warp"`

Please refer to our [Technical Report]() for the effect of different model architecutres can bring the model performance. 

### Camera Split
Users can select certain cameras for training and the other for testing by specifying the `camset` argument in `Dataset` class. This requires users to define a camera split configuration file and pass it to the `Dataset` constructor. For example:
```python
import json
f = open('path/to/camera-split-config.json', 'r')
camera_config = json.load(f)
f.close()
```
Then during construction, one can specify
```python
dataset_train = Dataset(args.data_dir, args.krt_dir, args.photo_dir, args.framelist_train, args.tex_size,
                            camset=camera_config['train'])
dataset_test = Dataset(args.data_dir, args.krt_dir, args.photo_dir, args.framelist_test, args.tex_size,
                            camset=camera_config['test'])
```
where the `camera-split-config.json` specifies the training and testing cameras IDs as follows:
```json
{
    "train": ["400069", "400002", "400023", "400070",
              "400042", "400013", "400031", "400004", "400050", "400037",
              "400059", "400051", "400009", "400007", "400053", "400055", "400010",
              "400030", "400061", "400063", "400027", "400018",
              "400067", "400008", "400064", "400039", "400026", "400041",
              "400028", "400015"],
    "test": ["400029", "400017", "400012", "400048", "400019", "400016", "400026",
             "400049", "400060"]
}
```
Notice the camera sets are capture-specific. One cannot load images from a camera if the camera does not present in the capture.

### Expression Split
Users can select certain expressions for training and the other for testing by specifying the `--test_segment_config` argument. This requires users to define a set of expressions that will be used for testing. For example:
`test_segment.json` specifies the testing segments as follows:
```json
{
	"segment": ["EXP_ROM", "EXP_free_face"]
}

```
