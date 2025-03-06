import pytest
import polars as pl
from datetime import date
from pipeline.inclusion_exclusion_criteria import main

def test_remove_single_episodes():
    df = pl.DataFrame({
        'ppn_int': [1, 1, 2, 3, 3],
        'episode_start_date': ['2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01', '2010-05-01'],
        'episode_end_date': ['2010-01-10', '2010-02-10', '2010-03-10', '2010-04-10', '2010-05-10']
    })
    result = df.filter((pl.len() > 1).over('ppn_int'))
    assert result.shape[0] == 4
    assert set(result['ppn_int'].unique().to_list()) == {1, 3}

def test_remove_unqualified_newborns():
    df = pl.DataFrame({
        'ppn_int': [1, 1, 2, 2, 3],
        'SRG': ['74', '75', '74', '76', '77'],
        #               1         1     2        2     3
        'age_recode': [0.000000, 0.1, 0.000000, 1.0, 5.0],
        'episode_start_date': ['2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01', '2010-05-01'],
        'episode_end_date': ['2010-01-10', '2010-02-10', '2010-03-10', '2010-04-10', '2010-05-10']
    })
    neonate_age_limit = 0.000001
    unqualified_neonate_expr = (pl.col('SRG') == '74') & (pl.col('age_recode') < neonate_age_limit)
    result = df.filter(~unqualified_neonate_expr)
    assert result.shape[0] == 3
    assert set(result['ppn_int'].unique().to_list()) == {1, 2, 3}

def test_remove_zombies():
    df = pl.DataFrame({
        'ppn_int': [1, 1, 2, 2, 3],
        'episode_start_date': ['2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01', '2010-05-01'],
        'episode_end_date': ['2010-01-10', '2010-02-10', '2010-03-10', '2010-04-10', '2010-05-10']
    })
    death = pl.DataFrame({
        'ppn_int': [1, 2, 3],
        'DEATH_DATE': ['2010-01-15', '2010-03-15', '2011-01-01']
    })
    last_episode = df.group_by('ppn_int').agg(pl.col('episode_start_date').max())
    last_episode = last_episode.join(death, on='ppn_int')
    zombies = last_episode.filter(pl.col('DEATH_DATE') < pl.col('episode_start_date'))
    result = df.join(zombies.select('ppn_int'), on='ppn_int', how='anti')
    assert set(zombies['ppn_int'].unique().to_list()) == {1, 2}
    assert result.shape[0] == 1
    assert set(result['ppn_int'].unique().to_list()) == {3}

def test_remove_no_age_information():
    df = pl.DataFrame({
        'ppn_int': [1, 1, 2, 2, 3],
        'age_recode': [10.0, 11.0, None, None, 5.0],
        'episode_start_date': ['2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01', '2010-05-01'],
        'episode_end_date': ['2010-01-10', '2010-02-10', '2010-03-10', '2010-04-10', '2010-05-10']
    })
    no_age = df.filter((pl.col('age_recode').is_null().all()).over('ppn_int'))
    result = df.join(no_age.select('ppn_int'), on='ppn_int', how='anti')
    assert result.shape[0] == 3
    assert set(result['ppn_int'].unique().to_list()) == {1, 3}

# Add more tests as needed
