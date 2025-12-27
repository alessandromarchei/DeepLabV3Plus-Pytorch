
#efficientnet_b0
#lr 0.01
#backbone lr coeff 1.0 (same as classifier)
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b0 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 4 \
    --total_itrs 160000 \
    --output_stride 16 \
    --enable_vis \
    --lr=0.01 \
    --lr_backbone_coeff=1.0 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b0_lr0_01_bn0_01_lrbackbone1_0 \
    --num_workers=4 \
    --prefetch_factor=2



#efficientnet_b0
#lr 0.01
#backbone lr coeff 0.1 (same as classifier)
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b0 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 4 \
    --total_itrs 160000 \
    --output_stride 16 \
    --enable_vis \
    --lr=0.01 \
    --lr_backbone_coeff=0.1 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b0_lr0_01_bn0_01_lrbackbone0_1 \