CUDA_VISIBLE_DEVICES=0 python run_seq2seq_trainer.py --learning_rate=1e-5 --adafactor \
--train_file="data/train_multitask.json" --validation_file="data/valid_multitask.json" \
--output_dir=saved/multitask --model_name_or_path="t5-large" \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16 --weight_decay=0.005 \
--num_train_epochs 5 --do_train True --do_eval True --evaluation_strategy="epoch" --save_strategy="epoch" \
--save_total_limit=5 --text_column input --summary_column output --source_prefix="" \
--overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_seq2seq_trainer.py --learning_rate=1e-5 --adafactor \
--train_file="data/train_composition.json" --validation_file="data/valid_composition.json" \
--output_dir=saved/composition --model_name_or_path="t5-large" \
--per_device_train_batch_size=16 --per_device_eval_batch_size=16 --weight_decay=0.005 \
--num_train_epochs 5 --do_train True --do_eval True --evaluation_strategy="epoch" --save_strategy="epoch" \
--save_total_limit=5 --text_column input --summary_column output --source_prefix="" \
--overwrite_output_dir