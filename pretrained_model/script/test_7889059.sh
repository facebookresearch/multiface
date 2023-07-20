python -m torch.distributed.launch --nproc_per_node=1 /mnt/home/xuhuahuang/multiface/test.py --data_dir /mnt/captures/ecwuu/m--20180927--0000--7889059--GHS --krt_dir /mnt/captures/ecwuu/m--20180927--0000--7889059--GHS/KRT --framelist_test /mnt/captures/ecwuu/m--20180927--0000--7889059--GHS/frame_list.txt --test_segment_config "/mnt/home/xuhuahuang/multiface/test_segment.json" --result_path "/mnt/home/xuhuahuang/multiface/eval/7889059--GHS-warp_nosl" --arch "warp" --model_path "/mnt/home/xuhuahuang/multiface/release/7889059--GHS-warp_nosl/best_model.pth" | tee eval/7889059_warp_nosl.txt