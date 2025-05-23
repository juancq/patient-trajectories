import os
import duckdb
import polars as pl


def main():
    #os.environ['POLARS_MAX_THREADS'] = "1"
    data_path = '/mnt/data_volume/apdc/str_group_files'
    cohort = pl.scan_parquet(f'{data_path}/apdc_group*.parquet')
    cohort_static = cohort.group_by('ppn_int').first()
    STATIC_NAME = f'{data_path}/apdc_static.parquet'
    #cohort_static.sink_parquet(STATIC_NAME)

    duckdb.execute("PRAGMA temp_directory='/mnt/fast_drive/temp';")
    duckdb.execute("PRAGMA enable_progress_bar;")
    duckdb.execute("PRAGMA threads=2;")
    duckdb.execute("PRAGMA memory_limit='50GB';")
    duckdb.execute("PRAGMA max_temp_directory_size='40GB';")

    #print(duckdb.query('copy(SELECT * FROM duckdb_settings()) to "hello.csv" (format csv);'))

    out_name = 'apdc.parquet'
    query = f"""
    COPY (
    SELECT * FROM
        (
            SELECT *, ROW_NUMBER() 
            OVER(PARTITION BY ppn_int,firstadm ORDER BY age_recode) as rn
            FROM '{data_path}/apdc_group*.parquet'
        ) sub
        WHERE rn = 1
    ) to '{data_path}/{out_name}' (FORMAT 'PARQUET', CODEC 'ZSTD');
        """
    duckdb.sql(query)

if __name__ == '__main__':
    main()
