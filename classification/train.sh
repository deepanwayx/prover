CUDA_VISIBLE_DEVICES=0 python train_classifier.py --lr 5e-6 --mnli-percent 1
CUDA_VISIBLE_DEVICES=0 python train_classifier.py --lr 5e-6 --mnli-percent 1  --aug-percent 1 --aug "prover"
CUDA_VISIBLE_DEVICES=0 python train_classifier.py --lr 5e-6 --mnli-percent 1  --aug-percent 1 --aug "entailer"