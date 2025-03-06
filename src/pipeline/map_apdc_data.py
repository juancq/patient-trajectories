import pickle
from pathlib import Path
import polars as pl

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)

streaming_flag = False

def map_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Maps categorical variables to integers based on the given encoding.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to be mapped.

    Returns
    -------
    pl.DataFrame
        The mapped DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame is None")
    if df.select(pl.len()).collect(streaming=streaming_flag).item() == 0:
        raise ValueError("Input DataFrame is empty")

    dv = pl.read_excel("df_vars.xlsx")
    if dv is None:
        raise ValueError("df_vars.xlsx is missing")
    if len(dv) == 0:
        raise ValueError("df_vars.xlsx is empty")

    start_idx_dict = dict(dv.select(pl.col(["name", "start_idx"])).iter_rows())

    variable_map = pickle.load(open("procedure_mapping.pkl", "rb"))
    encoding = variable_map['encoding']
    percentiles = variable_map['percentiles']

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
        print(bin_labels(value))
    
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
    visit_col_names = [name for name in dv.get_column('name') if name != 'sex']
    #visit_columns = df.select(pl.col(visit_col_names))
    #visit_columns = df.select(pl.col(dv.filter(pl.col("TYPE") == "visit").select("NAME").to_series()))
    #df = df.collect(streaming=True)
    df = df.with_columns(pl.concat_list(['sex']).list.drop_nulls().alias("labels"),
                        pl.concat_list(visit_col_names).list.drop_nulls().alias("visits"))
    df = df.group_by("ppn_int").agg(pl.first("labels"), pl.col("visits"))

    return df


def main():
    apdc_eddc_path = Path('/mnt/data_volume/apdc/study1')

    project_path = apdc_eddc_path / 'preprocessed' 
    n_groups = 50
    source_files = [project_path / f'group_{i}.parquet' for i in range(n_groups)]
    df = pl.scan_parquet(source_files)

    df_mapped = map_data(df)
    df_mapped = df_mapped.collect(streaming=True)

    print("Shuffle")
    df_mapped = df_mapped.sample(fraction=1, shuffle=True, seed=123)

    print('writing parquet')
    df_mapped.write_parquet(project_path / "df_ppn.parquet")


if __name__ == '__main__':
    main()
