
#efficientnet_b1
#lr 0.01
#backbone lr coeff 1.0 (same as classifier)
#bn_momentum 0.01
#aspp from layer 7
python main.py \
    --model deeplabv3plus_efficientnet_b1 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 80000 \
    --output_stride 16 \
    --enable_vis \
    --lr=0.01 \
    --lr_backbone_coeff=1.0 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --model_kwargs "aspp_from=stage7" \
    --exp_name deeplabv3plus_efficientnet_b1_lr0_01_bn0_01_stage7_lrbackbone1_0 \



#efficientnet_b1
#lr 0.01
#backbone lr coeff 0.1 (same as classifier)
#bn_momentum 0.01
#aspp from layer 7
python main.py \
    --model deeplabv3plus_efficientnet_b1 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 80000 \
    --output_stride 16 \
    --enable_vis \
    --lr=0.01 \
    --lr_backbone_coeff=0.1 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --model_kwargs "aspp_from=stage7" \
    --exp_name deeplabv3plus_efficientnet_b1_lr0_01_bn0_01_stage7_lrbackbone0_1 \



#efficientnet_b3
#lr 0.01
#backbone lr coeff 1.0 (same as classifier)
#bn_momentum 0.01
#aspp from layer 7
python main.py \
    --model deeplabv3plus_efficientnet_b3 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 80000 \
    --output_stride 16 \
    --enable_vis \
    --lr=0.01 \
    --lr_backbone_coeff=1.0 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --model_kwargs "aspp_from=stage7" \
    --exp_name deeplabv3plus_efficientnet_b3_lr0_01_bn0_01_stage7_lrbackbone1_0 \



#efficientnet_b3
#lr 0.01
#backbone lr coeff 0.1 (same as classifier)
#bn_momentum 0.01
#aspp from layer 7
python main.py \
    --model deeplabv3plus_efficientnet_b3 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 80000 \
    --output_stride 16 \
    --enable_vis \
    --lr=0.01 \
    --lr_backbone_coeff=0.1 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --model_kwargs "aspp_from=stage7" \
    --exp_name deeplabv3plus_efficientnet_b3_lr0_01_bn0_01_stage7_lrbackbone0_1 \
