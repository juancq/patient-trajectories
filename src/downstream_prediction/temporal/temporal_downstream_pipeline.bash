path=/mnt/data_volume/apdc/study1/preprocessed
# imput costs
python temporal/impute_cost.py --input_file $path/temporal_downstream_full_cohort.parquet --output $path/temporal_downstream_full_cohort_cost_imputed.parquet --abf-dnr-path /mnt/data_volume/apdc/

# split downstream train/test 
python temporal_split_downstream.py --input_file $path/temporal_downstream_full_cohort_cost_imputed.parquet --output_path $path

# pick index point and generate baseline features
python temporal/pick_prediction_point_scikit.py --train $path/temporal_downstream_train.parquet --death_record /mnt/data_volume/apdc/rbdm_processed.parquet --output_path $path/tasks
python temporal/pick_prediction_point_scikit.py --test $path/temporal_downstream_test.parquet --death_record /mnt/data_volume/apdc/rbdm_processed.parquet --output_path $path/tasks

# --> converts downstream data into HALO format
window="six_months"
python temporal/map_halo_cohort_for_finetune.py --index_file $path/tasks/temporal_cost_${window}_train.parquet  --cohort $path/temporal_downstream_train.parquet --output $path/tasks/temporal_downstream_train_random_index_${window}.parquet
python temporal/map_halo_cohort_for_finetune.py --index_file $path/tasks/temporal_cost_${window}_test.parquet  --cohort $path/temporal_downstream_test.parquet --output $path/tasks/temporal_downstream_test_random_index_${window}.parquet

# generate halo embeddings
python get_embeddings.py --model temporal --pooling_method last --input_files $path/tasks/temporal_downstream_train_random_index.parquet $path/tasks/temporal_downstream_test_random_index.parquet --output_path $path/tasks

# train/test downstream models
python evaluate_temporal_models.py --config temporal/temporal_eval_config.yaml
