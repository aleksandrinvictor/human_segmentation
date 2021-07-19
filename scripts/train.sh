python humseg/train.py \
    model=unet \
    training.model_id=unet \
    augmentation=basic


python humseg/train.py \
    model=unet \
    training.model_id=unet_augs \
    augmentation=unet

python humseg/train.py \
    model=unet_fpa \
    training.model_id=unet_fpa \
    augmentation=unet_fpa
