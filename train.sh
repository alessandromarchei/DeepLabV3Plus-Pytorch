

#efficientnet_b0
#lr 0.01
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b0 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 16 \
    --output_stride 16 \
    --enable_vis \
    --vis_port 8000 \
    --lr=0.01 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b0_lr0_01_bn0_01 \


#efficientnet_b0
#lr 0.01
#bn_momentum 0.1
python main.py \
    --model deeplabv3plus_efficientnet_b0 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 16 \
    --output_stride 16 \
    --enable_vis \
    --vis_port 8000 \
    --lr=0.01 \
    --bn_momentum=0.1 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b0_lr0_01_bn0_1 \



#efficientnet_b2
#lr 0.01
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b2 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 16 \
    --output_stride 16 \
    --enable_vis \
    --vis_port 8000 \
    --lr=0.01 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b2_lr0_01_bn0_01 \



#efficientnet_b2
#lr 0.01
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b2 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 16 \
    --output_stride 16 \
    --enable_vis \
    --vis_port 8000 \
    --lr=0.01 \
    --bn_momentum=0.1 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b2_lr0_01_bn0_1 \




