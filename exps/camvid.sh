# full
python train.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname full_bs4 \
--backbone mobilenet \
--seed 701

python retrain.py --dataset CamVid --base-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--lr 0.01 --lr-scheduler cos \
--use-balanced-weights \
--checkname retrain_levelregion \
--backbone mobilenet --with-mask --mask-loss weight_ce \
--init-percent 10 \
--seed 701

## active

## region sum 截断 效果一般

# diff_entropy
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname active_diff_entropy8 \
--backbone mobilenet --with-mask --mask-loss weight_ce \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --hard-levels 8 --strategy diff_entropy \
--seed 400

# diff_score
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname diff_score \
--backbone mobilenet --with-mask --mask-loss weight_ce \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask \
--seed 400



# todo: without PAM
# 209 entro200 701
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname active_nopam \
--backbone mobilenet --with-mask --mask-loss weight_ce \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --hard-levels 8 --strategy region_entropy \
--seed 701 --with-pam False


# without mask branch
# 209 rand 300
# entro 400
python train_active_txt.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname active_nomask_region_entropy8 \
--backbone mobilenet \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--resume-dir runs/CamVid/active_region_entropy8_400_Jun23_221500 \
--seed 400

# 110 region 701
python acquire_test.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname diff_entropy8 \
--backbone mobilenet --with-mask --mask-loss weight_ce \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --strategy diff_entropy --hard-levels 8 \
--resume-iteration 1 --resume-dir runs/CamVid/active_region_entropy8_701_Jun21_191044 \
--seed 701

# 207 score
python acquire_test.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname diff_score \
--backbone mobilenet --with-mask --mask-loss weight_ce \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --strategy diff_score \
--resume-iteration 1 --resume-dir runs/CamVid/active_region_entropy8_701_Jun21_191044 \
--seed 701

# 207 mc 只用 score
python acquire_test.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname diff_only_score \
--backbone mobilenet --with-mask --mask-loss weight_ce \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode error_mask --strategy diff_score \
--resume-iteration 1 --resume-dir runs/CamVid/active_region_entropy8_701_Jun21_191044 \
--seed 701

---------------------------------

# coreset
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname active_coreset \
--backbone mobilenet \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode coreset \
--seed 400

# vaal
# 209 rand 701
# entro 100
# 208 sum 200
# cit 300
# rand 400
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname active_vaal \
--backbone mobilenet \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode coreset \
--seed 400

# entropy
# 207 entro 300
# rand 400
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname active_entropy \
--backbone mobilenet \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode entropy \
--seed 400

# dropout
# 209 mc 300
# entro 400
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 0 \
--lr 0.01 --lr-scheduler poly \
--checkname active_dropout \
--backbone mobilenet \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode dropout \
--seed 400

# random
python train_active.py --dataset CamVid --base-size 360,480 --crop-size 360,480 --workers 4 \
--epochs 100 --eval-interval 5 --batch-size=4 \
--gpu-ids 1 \
--lr 0.01 --lr-scheduler poly \
--checkname active_random \
--backbone mobilenet \
--init-percent 10 --select-num 20 \
--max-iterations 6 --percent-step 5 \
--active-selection-mode random \
--seed 400

