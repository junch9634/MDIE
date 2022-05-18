python train_mult.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 1 --model_type single
python train_mult.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 2 --model_type multi-last-avg
python train_mult.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 3 --model_type multi-tot-avg