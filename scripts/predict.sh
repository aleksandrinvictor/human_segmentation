# python -W ignore humseg/inference.py \
#     --model_path="./outputs/unet" \
#     --output_path="./results/unet"

python -W ignore humseg/inference.py \
    --model_path="./outputs/unet_augs" \
    --output_path="./results/unet_augs"


# python -W ignore humseg/inference.py \
#     --model_path="./outputs/unet_fpa" \
#     --output_path="./results/unet_fpa"
