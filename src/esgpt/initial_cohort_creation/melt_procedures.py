import os
import argparse
import polars as pl
import polars.selectors as cs
from pathlib import Path
from loguru import logger


def melt_apdc_diagnosis(data_path, name, lazy_frame=None):
    id_vars = ['ppn', 'ppn_int', 'episode_start_date', 'firstadm']

    if lazy_frame is None:
        df = pl.scan_parquet(data_path / name)
    else:
        df = lazy_frame

    logger.info(f'Melting diagnosis codes from {name}')
    value_name = 'diagnosis_code'
    variable_name = f'{value_name}_src'

    df_block = df.melt(id_vars=id_vars, value_vars=cs.starts_with('diagnosis_code'),
                variable_name=variable_name, value_name=value_name)
    # drop source name from melt, because it just stores source column name
    df_block = df_block.drop(variable_name)
    # replace empty strings with nulls
    df_block = df_block.with_columns(pl.when(pl.col(value_name).str.lengths()==0)
                                    .then(None)
                                    .otherwise(pl.col(value_name))
                                    .keep_name()
                                    )
    df_block = df_block.drop_nulls(value_name)
    df_block = df_block.with_columns(pl.col(value_name).str.replace('\..*$', '').alias('diagnosis_short'))
    df_block = df_block.with_columns(pl.col('episode_start_date').cast(pl.Date))

    out_name = name.replace('.parquet', f'_{value_name}.parquet')
    drop_dups_and_save(df_block, data_path, out_name, subset=id_vars + ['diagnosis_short'])

    logger.info('Done')


def old_drop_dups_and_save(df, data_path, out_name, subset=None):
    df = df.collect(streaming=True)
    logger.info('Saving with duplicates')
    df.write_parquet(data_path / out_name, use_pyarrow=True)

    logger.info('Dropping duplicates and saving')
    out_name = out_name.replace('.parquet', f'_unique.parquet')
    df.unique(subset=subset).write_parquet(data_path / out_name, use_pyarrow=True)


def drop_dups_and_save(df, out_name, subset=None):
    logger.info('Dropping duplicates and saving')
    out_name = out_name.with_stem(out_name.stem + f'_unique')
    df.unique(subset=subset).collect().write_parquet(out_name)


def melt_apdc_procedures(df: pl.LazyFrame):
    id_vars = ['ppn', 'ppn_int', 'episode_start_date']

    logger.info('Melting procedure blocks')
    value_name = 'procedure_block'
    variable_name = f'{value_name}_src'

    df_block = df.melt(id_vars=id_vars, value_vars=cs.starts_with('block_num'),
                variable_name=variable_name, value_name=value_name)
    # drop procedure_blockx source name from melt
    df_block = df_block.drop(variable_name)
    # replace empty strings with nulls
    df_block = df_block.with_columns(pl.when(pl.col(value_name).str.len_bytes()==0)
                                    .then(None)
                                    .otherwise(pl.col(value_name))
                                    .alias(value_name)
                                    )
    df_block = df_block.drop_nulls(value_name)

    procedure_class = [109, 159, 299, 369, 449, 519, 599, 799, 849, 1039, 1159, 1239, 
                        1329, 1359, 1599, 1739, 1785, 1819, 1939]
    procedure_chapter = list(map(str, range(1, len(procedure_class)+2)))

    df_block = df_block.with_columns(pl.col(value_name).cast(pl.Int16)
                            .cut(procedure_class, labels=procedure_chapter)
                            .cast(pl.Utf8).cast(pl.Int8).alias('procedure_class'))

    df_block = df_block.unique(subset=None)

    logger.info('Melting procedure codes')
    value_name = 'procedure'
    variable_name = f'{value_name}_src'

    df_proc = df.melt(id_vars=id_vars, value_vars=cs.starts_with('procedure_code'),
                variable_name=variable_name, value_name=value_name)
    df_proc = df_proc.drop(variable_name)
    df_proc = df_proc.drop_nulls(value_name)

    df_proc = df_proc.unique(subset=None)

    logger.info('Done')
    
    return df_block, df_proc


def main():
    os.environ['POLARS_MAX_THREADS'] = "2"
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='APDC path to file, i.e. /mnt/data_volume/apdc/apdc.parquet', 
                        default=None, type=str)
    args = parser.parse_args()

    #'/mnt/fast_drive/apdc/five_year_predeath_apdc.parquet'
    melt_apdc_procedures(Path(args.file))


if __name__ == '__main__':
    main()
