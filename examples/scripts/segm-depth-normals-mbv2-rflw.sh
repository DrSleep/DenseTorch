# Training Multi-Task Light-Weight-RefineNet with MobileNet-v2
# on three tasks: segmentation, depth and surface normals estimation
# using NYUDv2.
python src/train.py \
    --tasks segm depth normals \
    --enc-backbone "mobilenetv2" \
    --enc-return-layers 1 2 3 4 5 6 \
    --dec-combine-layers "[[0, 1], [2, 3], 4, 5]" \
    --num-classes 40 1 3 \
    --ignore-indices 255 0 1 \
    --train-list-path ./src/data/train_list3.txt \
    --val-list-path ./src/data/val_list3.txt \
    --train-dir ./src/datasets/nyudv2/ \
    --val-dir ./src/datasets/nyudv2/ \
    --val-every 5 \
    --depth-scale 5000.0 \
    --initial-values 0.0 10000.0 10000.0 \
    --saving-criterions "up" "down" "down" \
    --tasks-loss-weights 0.5 0.5 5.0 \
    --enc-lr 1e-3 \
    --dec-lr 1e-2 \
    --ckpt-dir "./checkpoints-mbv2-sdn/" \
    --ckpt-path "./checkpoint.pth.tar2" \
    --freeze-bn 0 \
    --high-scale 1.0 \
    --low-scale -0.2 \
    --crop-size 512
    # Albumentations scale is very dangerous as it adds bias of +1.0 for tuples....
