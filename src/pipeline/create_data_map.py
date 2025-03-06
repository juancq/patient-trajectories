import polars as pl
import pickle
import numpy as np
from pathlib import Path


def main():
    """
    The main function is responsible for processing data from the APDC dataset.
    It configures the Polars library, loads data from parquet files, and performs 
    various data processing tasks such as handling categorical variables, 
    creating mappings for procedures, and binning numerical variables.
    
    The function does not take any parameters and does not return any values. 
    Instead, it saves the processed data mappings to a pickle file named 
    "procedure_mapping.pkl".
    """
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(20)

    root_path = Path('/mnt/data_volume/apdc')
    apdc_eddc_path = root_path / 'study1'

    try:
        df = pl.scan_parquet(apdc_eddc_path / 'preprocessed' / 'group_*.parquet')
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    categorical = ['sex', 'hospital_type', 'emergency_status_recode', 'episode_of_care_type',
                    # discharge on leave, transferred, etc., 0-11
                    'mode_of_separation_recode', 
                    'peer_group', 
                    'facility_type', 
                    'facility_identifier',
                    'facility_trans_from_e',
                    'facility_trans_to_e',
                    # outpatients, palliative care, etc., 01-13
                    'referred_to_on_separation_recode',
                    'acute_flag',
                    'ed_status',
                    # born in hospital, emergency department, etc., 00-16
                    'source_of_referral_recode',
                    #---------------------
                    # ed
                    # review in ed, home nursing, outpatient clinic, specialist, etc. 01-12
                    'referred_to_on_departure_recode',
                    'ed_visit_type',
                    'triage_category', 
                    # ambulance, public transport, private vehicle, etc., 1-11
                    'mode_of_arrival',
                    'mode_of_separation',
                    # family, specialist, outpatient clinic, etc 1-19,99
                    'ed_source_of_referral',
                    ]

    procedures = df.select(pl.col('procedure_list').flatten().unique())
    all_procedures = procedures.collect(streaming=True).get_column('procedure_list').to_list()
    if not all_procedures:
        print("No procedures found")
        return
    print(len(all_procedures))
    # map from procedure to id, and from id to procedure
    procedure_to_id = {proc:i for i,proc in enumerate(all_procedures)}
    id_to_procedure = {i:proc for i,proc in enumerate(all_procedures)}


    # numerical variables, to be broken into bins: age, length of stay, 
    # days since last episode
    to_bin = ['age_recode', 'length_of_stay', 'days_between_episodes'] 
    percentiles = {}
    for name in to_bin:
        values = df.select(name).collect(streaming=True).to_series(0)
        if values.empty:
            print(f"No values for {name}")
            continue
        percentiles[name] = np.unique(np.percentile(values, np.arange(5, 100, 5)))
        print(f"{name} percentiles: {percentiles[name]}")

    encoding = {}

    # replace categories with counts below a threshold with "other" label
    replace = {}
    category_threshold = 20
    for col in categorical:
        # get counts of categories
        counts = df.select(pl.col(col).value_counts(sort=True)).collect(streaming=True).unnest(col)
        # get only categorie labels below count threshold
        counts = counts.filter(pl.col('count') < category_threshold).get_column(col).to_list()
        # build a replacement dictionary to pass in a call to .replace
        replace[col] = {category:'other' for category in counts}

    # replace low count categories with "other" for all categorical columns
    df = df.with_columns(
        *[pl.col(c).replace(replace[c]) for c in categorical]
    )

    # create encoding
    for col in categorical:
        unique_values = df.select(pl.col(col).unique()).collect().to_series(0)
        encoding[col] = {value:i for i,value in enumerate(unique_values)}
        print(df.select(pl.col(col).value_counts(sort=True)).unnest(col).collect(streaming=True))
        #df = df.with_columns(pl.col(c).replace_strict(encoding[c]))
        #print(df.select(pl.col(c).value_counts(sort=True)).unnest(c).collect(streaming=True))

    encoding['procedure_list'] = procedure_to_id

    # save mappings
    pickle.dump({'encoding': encoding, 'percentiles': percentiles},
        open("procedure_mapping.pkl", "wb"))

if __name__ == "__main__":
    main()