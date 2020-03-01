python run.py configs/native/balanced/predict_hasbro.yaml predict_ensemble --gpu 0 --num-workers 8
python run.py configs/native/ensemble/predict_hasbro.yaml predict_ensemble --gpu 0 --num-workers 8
python run.py configs/native/sampling/predict_hasbro.yaml predict_ensemble --gpu 0 --num-workers 8

python run.py configs/gp/balanced/predict_hasbro.yaml predict_ensemble --gpu 0 --num-workers 8
python run.py configs/gp/ensemble/predict_hasbro.yaml predict_ensemble --gpu 0 --num-workers 8
python run.py configs/gp/sampling/predict_hasbro.yaml predict_ensemble --gpu 0 --num-workers 8
