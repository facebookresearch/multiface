python -m torch.distributed.launch --nproc_per_node=1 /mnt/home/xuhuahuang/multiface/train.py --data_dir /mnt/captures/ecwuu/m--20190529--1300--002421669--GHS --krt_dir /mnt/captures/ecwuu/m--20190529--1300--002421669--GHS/KRT --framelist_train /mnt/captures/ecwuu/m--20190529--1300--002421669--GHS/frame_list.txt --framelist_test /mnt/captures/ecwuu/m--20190529--1300--002421669--GHS/frame_list.txt --test_segment "/mnt/home/xuhuahuang/multiface/test_segment.json" --result_path "/mnt/home/xuhuahuang/multiface/release/002421669--GHS-warp_nosl" --arch "warp" | tee release/002421669_warp_nosl.txt