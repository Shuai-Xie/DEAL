# full
python train.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname full_mobilenet \
--backbone mobilenet \
--init-percent 100 \
--seed 701

# ----------------------------

# 10% debug
python train.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname pam_class \
--backbone mobilenet --with-mask --mask-loss wce \
--init-percent 10 \
--seed 701

# drn
python train.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--segmodel drn --lr 0.001 --optimizer Adam \
--checkname drn --with-mask --mask-loss wce \
--init-percent 10 \
--seed 701

# retrain
python retrain.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 1 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname retrain_diff_score40 \
--backbone resnet101 \
--init-percent 10 \
--seed 100


# drn: bs=8, 22872MiB, 1epoch, 15min
# resnet50, 14149MiB, 5min

# balance_ce 会对主干影响太大

# diff_score
python train_active.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname active_diff_score \
--backbone mobilenet --with-mask --mask-loss wce \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --strategy diff_score \
--seed 200


# rand 701
# cit 100
# core 200
python train_active.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 2 \
--lr 0.01 --lr-scheduler poly \
--checkname active_region_entropy \
--backbone mobilenet --with-mask --mask-loss wce \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --hard-levels 8 --strategy region_entropy \
--seed 200

# nomask
python train_active_txt.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 2 \
--lr 0.01 --lr-scheduler poly \
--checkname diff_score_nomask \
--backbone mobilenet \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--resume-iteration 6 \
--resume-dir runs/Cityscapes/diff_score_100_Jul01_092747 \
--seed 100

# retrain map sum
# 208 entro 10  54.97
# 210 cit 8     53.81
python acquire_test.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname diff_entropy10 \
--backbone mobilenet --with-mask --mask-loss wce \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --strategy diff_entropy --hard-levels 10 \
--resume-iteration 4 --resume-dir runs/Cityscapes/diff_entropy10_100_Jul03_135148 \
--seed 100

----------------------------------------------

# random
python train_active.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname active_random \
--backbone mobilenet \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode random \
--seed 200

# entropy
python train_active.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 2 \
--lr 0.01 --lr-scheduler poly \
--checkname active_entropy_200 \
--backbone mobilenet \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode entropy \
--seed 200

# coreset
python train_active.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname active_coreset \
--backbone mobilenet \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode coreset \
--seed 200

# vaal
# 208 200
python train_active.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname active_vaal \
--backbone mobilenet \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode coreset \
--seed 200


# dropout
# 209 cit 100
python train_active.py --dataset Cityscapes --base-size 688,688 --crop-size 688,688 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=8 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname active_dropout \
--backbone mobilenet \
--init-percent 10 --select-num 150 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode dropout \
--seed 200




















