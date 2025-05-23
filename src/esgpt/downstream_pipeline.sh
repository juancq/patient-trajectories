# Downstream pipeline
# generates downstream dataset (keeps only record up to a randomly selected index point)
# random index selection happens in pick_prediction_point.py
python generate_task_dataset.py +dataset_dir="/mnt/data_volume/apdc/study1/preprocessed/esgpt_data/study1/"
# generates embeddings for trajectories generated in previous step
python generate_prediction_point_embeddings.py --config-name=study1_convert_embed --config-path=configs
