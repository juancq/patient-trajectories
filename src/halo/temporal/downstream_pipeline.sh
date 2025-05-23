path=/mnt/data_volume/apdc/study1/preprocessed
window="six_months"
# these steps are intertwined with downstream task, need to document the order of processing
python get_embeddings.py --model temporal --pooling_method last --input_files $path/tasks/temporal_downstream_train_random_index_${window}.parquet $path/tasks/temporal_downstream_test_random_index_${window}.parquet --output_path $path/tasks
python final_fine_tune.py --model temporal --train $path/tasks/temporal_downstream_train_random_index_${window}.parquet --test $path/tasks/temporal_downstream_test_random_index_${window}.parquet 
python fine_tune_bootstrap_eval.py --model temporal --train $path/tasks/temporal_downstream_train_random_index_${window}.parquet --test $path/tasks/temporal_downstream_test_random_index_${window}.parquet 
