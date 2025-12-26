

#efficientnet_b3
#lr 0.01
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b3 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 60000 \
    --output_stride 16 \
    --enable_vis \
    --vis_env deeplabv3plus_efficientnet_b3_lr0_01_bn0_01 \
    --vis_port 8001 \
    --lr=0.01 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b3_lr0_01_bn0_01 \



#efficientnet_b1
#lr 0.01
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b1 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 60000 \
    --output_stride 16 \
    --enable_vis \
    --vis_env deeplabv3plus_efficientnet_b1_lr0_01_bn0_01 \
    --vis_port 8001 \
    --lr=0.01 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b1_lr0_01_bn0_01 \


#efficientnet_b4
#lr 0.01
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b4 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 60000 \
    --output_stride 16 \
    --enable_vis \
    --vis_env deeplabv3plus_efficientnet_b4_lr0_01_bn0_01 \
    --vis_port 8001 \
    --lr=0.01 \
    --bn_momentum=0.01 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b4_lr0_01_bn0_01 \




#efficientnet_b1
#lr 0.01
#bn_momentum 0.1
python main.py \
    --model deeplabv3plus_efficientnet_b1 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 60000 \
    --output_stride 16 \
    --enable_vis \
    --vis_env deeplabv3plus_efficientnet_b1_lr0_01_bn0_1 \
    --vis_port 8001 \
    --lr=0.01 \
    --bn_momentum=0.1 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b1_lr0_01_bn0_1 \


#efficientnet_b3
#lr 0.01
#bn_momentum 0.1
python main.py \
    --model deeplabv3plus_efficientnet_b3 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 60000 \
    --output_stride 16 \
    --enable_vis \
    --vis_env deeplabv3plus_efficientnet_b3_lr0_01_bn0_1 \
    --vis_port 8001 \
    --lr=0.01 \
    --bn_momentum=0.1 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b3_lr0_01_bn0_1 \



#efficientnet_b4
#lr 0.01
#bn_momentum 0.01
python main.py \
    --model deeplabv3plus_efficientnet_b4 \
    --dataset cityscapes    \
    --crop_size 768 \
    --multiscale_train \
    --batch_size 8 \
    --total_itrs 60000 \
    --output_stride 16 \
    --enable_vis \
    --vis_env deeplabv3plus_efficientnet_b4_lr0_01_bn0_1 \
    --vis_port 8001 \
    --lr=0.01 \
    --bn_momentum=0.1 \
    --val_interval=500 \
    --exp_name deeplabv3plus_efficientnet_b4_lr0_01_bn0_1 \