
# # baseline
# for s in 0 1 2 3 4 5 6; do
#   python train_imagenet_small.py \
#     --seed $s \
#     --train_dir /root/autodl-tmp/imagenet_small/train \
#     --val_dir /root/autodl-tmp/imagenet_small/val \
#     --epochs 10 \
#     --lr 0.001
# done

# proposal
for s in 0 1 2 3 4 5 6; do
  python train_imagenet_small.py \
    --seed $s \
    --train_dir /root/autodl-tmp/imagenet_small/train \
    --val_dir /root/autodl-tmp/imagenet_small/val \
    --epochs 10 \
    --lr 0.0005
done
