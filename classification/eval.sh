# Replace --model-path with your trained model weights

CUDA_VISIBLE_DEVICES=0 python eval_lonli.py --model-path "saved/1695618252/best_loss.pt"
CUDA_VISIBLE_DEVICES=0 python eval_med.py --model-path "saved/1695618252/best_f1.pt"