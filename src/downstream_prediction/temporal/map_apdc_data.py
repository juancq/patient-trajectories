import argparse
import pickle
from pathlib import Path
import polars as pl
import numpy as np
from loguru import logger

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)

# streaming = {'engine': 'streaming'}
streaming_flag = True

def map_data(df: pl.DataFrame, df_vars: pl.DataFrame, variable_mapping: dict) -> pl.DataFrame:
    """
    Maps categorical variables to integers based on the given encoding.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to be mapped.
    df_vars : pl.DataFrame
        The DataFrame with variable start and end index for indexing in HALO binary vectors.
    variable_mapping : dict
        dictionary with variable mappings

    Returns
    -------
    pl.DataFrame
        The mapped DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame is None")
    if df.select(pl.len()).collect(streaming=streaming_flag).item() == 0:
        raise ValueError("Input DataFrame is empty")

    if df_vars is None:
        raise ValueError("Variable mapping is None")

    start_idx_dict = dict(df_vars.select(pl.col(["name", "start_idx"])).iter_rows())

    encoding = variable_mapping['encoding']
    percentiles = variable_mapping['percentiles']

    for key,value in percentiles.items():
        if value is None:
            raise ValueError(f"Percentiles for column {key} is None")
        if len(value) == 0:
            raise ValueError(f"Percentiles for column {key} is empty")
        percentiles[key] =  [int(i) for i in value]

    def bin_labels(b): 
        if b is None:
            raise ValueError("Bin labels are None")
        return list(map(str,range(len(b)+1)))

    for key,value in percentiles.items():
        if value is None:
            raise ValueError(f"Percentiles for column {key} is None")
        logger.info(bin_labels(value))
    
    int_type = pl.Int32
    #length of stay 0 is set to 0 by default for same-day stays
    df = df.with_columns(
        *[pl.col(col).cut(values, labels=bin_labels(values))
                .cast(int_type)+start_idx_dict[col] for col,values in percentiles.items()],
        *[pl.col(col).replace_strict(values, default=encoding[col].get(None, None)) 
            .cast(int_type) + start_idx_dict[col] for col,values in encoding.items() if 'procedure' not in col],
        pl.col("procedure_list").list.eval(
            pl.element().replace_strict(encoding['procedure_list'], default=encoding['procedure_list'][None])+start_idx_dict["procedure_list"]
            ).cast(pl.List(int_type)),
    )

    df = df.select(pl.all().shrink_dtype())

    # concat and group by ppn so that each row corresponds to one patient
    #label_columns = df.select(pl.col(['sex']))
    visit_col_names = [name for name in df_vars.get_column('name') if name != 'sex']
    #visit_columns = df.select(pl.col(visit_col_names))
    #visit_columns = df.select(pl.col(dv.filter(pl.col("TYPE") == "visit").select("NAME").to_series()))
    #df = df.collect(streaming=True)
    df = df.with_columns(pl.concat_list(['sex']).list.drop_nulls().alias("labels"),
                        pl.concat_list(visit_col_names).list.drop_nulls().alias("visits"))
    df = df.group_by("ppn_int").agg(pl.first("labels"), pl.col("visits"))

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='path to group files, i.e. /mnt/data_volume/apdc/apdc.parquet', 
                        default=None, 
                        type=str)
    parser.add_argument('--temporal', help='for temporal split', default=False, action='store_true')
    args = parser.parse_args()


    if args.file is None:
        project_path = Path('/mnt/data_volume/apdc/study1/preprocessed')
        if not project_path.exists():
            raise FileNotFoundError(f"Path does not exist: {project_path}")
        n_groups = 50
        source_files = [project_path / f'group_{i}.parquet' for i in range(n_groups)]
    else:
        source_files = Path(args.file)
        if not source_files.exists() or source_files.is_dir():
            raise FileNotFoundError(f"Path does not exist: {source_files}")
        project_path = source_files.parent

    df = pl.scan_parquet(source_files)

    if args.temporal:
        df_vars = pl.read_excel("temporal_df_vars.xlsx")
        variable_mapping_fname = 'temporal_variable_mapping_for_halo.pkl'
    else:
        df_vars = pl.read_excel("df_vars.xlsx")
        variable_mapping_fname = 'variable_mapping_for_halo.pkl'

    if df_vars is None:
        raise ValueError("df_vars.xlsx is missing")
    if len(df_vars) == 0:
        raise ValueError("df_vars.xlsx is empty")

    variable_mapping_fname = Path(variable_mapping_fname)
    if not variable_mapping_fname.exists() or variable_mapping_fname.is_dir():
        raise FileNotFoundError(f"Path does not exist: {variable_mapping_fname}")

    variable_mapping = pickle.load(open(variable_mapping_fname, "rb"))

    df_mapped = map_data(df, df_vars, variable_mapping)
    df_mapped = df_mapped.collect(streaming=streaming_flag)

    logger.info("Shuffle")
    df_mapped = df_mapped.sample(fraction=1, shuffle=True, seed=123)

    logger.info('writing parquet')
    output_fname = 'df_ppn.parquet'
    if args.temporal:
        output_fname = 'temporal_' + output_fname
    df_mapped.write_parquet(project_path / output_fname)

    # non-temporal split was done via ESGPT to align ESGPT and HALO
    if args.temporal:
        num_patients = len(df_mapped)
        logger.info("Splitting Datasets")
        val_size = int(np.ceil(0.1 * num_patients))
        df_val = df_mapped.head(val_size)
        df_trn = df_mapped.tail(num_patients - val_size)
        logger.info(f"Number of train/validation patients: {len(df_trn)}/{len(df_val)}")

        # save
        logger.info("Saving")
        df_trn.write_parquet(project_path / "temporal_df_train_halo.parquet")
        df_val.write_parquet(project_path / "temporal_df_val_halo.parquet")


if __name__ == '__main__':
    main()
