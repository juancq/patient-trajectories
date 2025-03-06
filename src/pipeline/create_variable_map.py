import pickle
import polars as pl


def main():
    """
    Create a dataframe df_vars.xlsx that contains the name, number of codes, labels, 
    start index and end index of each column in the encoding dictionary

    The encoding dictionary is a dictionary where the keys are the column names and the values are dictionaries 
    with the keys as the codes and the values as the description of the codes.

    The percentiles dictionary is a dictionary where the keys are the column names and the values are lists 
    of the percentiles of the corresponding column.
    """
    variable_map = pickle.load(open("procedure_mapping.pkl", "rb"))
    encoding = variable_map['encoding']
    percentiles = variable_map['percentiles']
    encoding.update(percentiles)
    columns = list(encoding.keys())
    num_codes = [len(encoding[key])+1 for key in columns]
    labels = [list(labels.keys()) if type(labels) is dict else list(labels) for _,labels in encoding.items()]

    df = pl.DataFrame({'name':columns, 'num_codes':num_codes, 'labels':labels})
    df = df.with_columns(pl.cum_sum("num_codes").alias("end_idx"))
    df = df.with_columns(pl.col("end_idx").shift(fill_value=0).alias("start_idx"))
    df.write_excel("df_vars.xlsx")


if __name__ == '__main__':
    main()