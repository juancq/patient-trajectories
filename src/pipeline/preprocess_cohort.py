import polars as pl
import polars.selectors as cs
from pathlib import Path
from rich.progress import track


def main():
    """
    This function preprocesses data from APDC and EDDC files, performing various operations such as data cleaning, 
    data transformation, and data merging. It reads data from parquet files, applies specific preprocessing steps 
    for APDC and EDDC data, joins the two datasets, and performs additional generic preprocessing steps. The 
    preprocessed data is then written to new parquet files.

    Parameters:
    None

    Returns:
    None
    """
    root_path = Path('/mnt/data_volume/apdc')
    apdc_eddc_path = root_path / 'study1'

    groups = 50
    for i in track(range(groups)):

        apdc_path = apdc_eddc_path / f'cohort_apdc_group_{i}_data.parquet'
        ed_path = apdc_eddc_path / f'cohort_eddc_group_{i}_data.parquet'

        # read apdc
        apdc = pl.scan_parquet(apdc_path).with_columns(pl.lit(1).alias('apdc'))
        apdc = apdc.rename(lambda column_name: column_name.lower())
        
        # read eddc
        ed = pl.scan_parquet(ed_path).with_columns(pl.lit(1).alias('eddc'))
        ed = ed.rename(lambda column_name: column_name.lower())

        # apdc specific preprocessing
        apdc = apdc.with_columns(
            pl.when((pl.col('hospital_type') > 1) & (pl.col('peer_group').is_null()))
            .then(pl.lit('PRIV'))
            .otherwise(pl.col('peer_group'))
            .alias('peer_group'),
            
            # at some point facility_identifier_recode was accidentally renamed to facility_identifier_e
            pl.col('facility_identifier_e').alias('facility_identifier'),
        
            # convert hospital type to binary instead of having 3 values
            # > 1 would be a private hospital (values of 2 or 4)
            pl.col(["hospital_type"]) > 1,
        )


        # eddc specific preprocessing
        ed = ed.with_columns(
            pl.col('arrival_date').alias('episode_start_date'),
            pl.col('actual_departure_date').alias('episode_end_date'),
            pl.col('arrival_time').alias('episode_start_time'),
            pl.col('actual_departure_time').alias('episode_end_time'),
            pl.col('mthyr_birth_date').alias('mthyr_birth'),
        )

        # join apdc and eddc
        df = pl.concat([apdc, ed], how='diagonal')

        # generic preprocess that applies to both apdc and eddc
        # convert string dates to date type
        dates = ["episode_start_date", "episode_end_date"]
        time_cols = ["episode_start_time", "episode_end_time"]
        date_to_convert = apdc.select(dates).select(cs.string()).columns

        df = df.with_columns(
            pl.col(date_to_convert).str.to_date("%m/%d/%Y"),
            pl.col(time_cols).cast(pl.Int32).fill_null(0).clip(0, 86399),
        )

        # add time to date to make date-time column
        df = df.with_columns(
            pl.col('episode_start_date').cast(pl.Datetime) 
                    + pl.duration(seconds=pl.col('episode_start_time')),
            pl.col('episode_end_date').cast(pl.Datetime) 
                    + pl.duration(seconds=pl.col('episode_end_time'))
        )

        # impute age, sex, and date of birth
        df = df.with_columns(
            pl.col(["age_recode", "sex", 'mthyr_birth'])
            .forward_fill()
            .backward_fill()
            .over("ppn_int")
        )

        # sorting by episode_end_date seems preferable to account for nested episodes
        df = df.sort(["ppn_int", "episode_start_date"])

        # compute days between episodes
        df = df.with_columns(
            episode_count_ppn=pl.col("ppn").cum_count().over("ppn")-1,
            previous_end_date=pl.col("episode_end_date").shift().over("ppn_int"),
        )
        df = df.with_columns(
            days_between_episodes=(pl.col("episode_start_date") - pl.col("previous_end_date"))
                                .dt.total_days()
                                .fill_null(0)
                                .clip(lower_bound=0),
            length_of_stay=(pl.col("episode_end_date") - pl.col("episode_start_date"))
                                .dt.total_days()
                                .fill_null(0)
                                .clip(lower_bound=0),
        )

        procedure_columns = cs.starts_with("block_num")
        df = df.with_columns(pl.concat_list(procedure_columns).alias("procedure_list")
                                .list.unique().drop_nulls().list.eval(pl.element().filter(pl.element() != ""), 
                                parallel=True))

        df.collect(streaming=True).write_parquet(apdc_eddc_path / 'preprocessed' / f'group_{i}.parquet')


if __name__ == "__main__":
    main()
