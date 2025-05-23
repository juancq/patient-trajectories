python temporal_split_pretrain_downstream.py
python create_data_map.py --file /mnt/data_volume/apdc/study1/preprocessed/temporal_pretrain_full_cohort.parquet --temporal
python create_variable_map.py temporal_variable_mapping_for_halo.pkl temporal_df_vars.xlsx
python map_apdc_data.py --file /mnt/data_volume/apdc/study1/preprocessed/temporal_pretrain_full_cohort.parquet --temporal
