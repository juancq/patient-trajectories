import polars as pl
pl.Config.set_tbl_cols(7)

head_size = 5
print(pl.scan_parquet('data/apdc/subjects_df.parquet').head(head_size).collect())

print(pl.scan_parquet('data/apdc/events_df.parquet').head(head_size).collect())

print(pl.scan_parquet('data/apdc/dynamic_measurements_df.parquet').head(head_size).collect())
