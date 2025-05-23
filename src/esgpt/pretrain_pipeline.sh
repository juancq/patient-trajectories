# converts raw data to ESGPT internal format
python build_dataset.py --config-path=configs --config-name=build_study1
python pretrain.py --config-path=configs --config-name=pretrain_study1.yaml
