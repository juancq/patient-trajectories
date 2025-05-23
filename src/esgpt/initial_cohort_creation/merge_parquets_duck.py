import duckdb

duckdb.execute("PRAGMA enable_progress_bar;")
duckdb.execute("PRAGMA threads=4;")
duckdb.execute("PRAGMA temp_directory='/mnt/fast_drive/temp';")
data_path = '/mnt/data_volume/apdc'
name= 'eddc'
duckdb.sql(f"COPY (select * from read_parquet('{data_path}/str_group_files/{name}_group_*_str.parquet')) to '{data_path}/preprocessed/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")