import stackprinter
stackprinter.set_excepthook(style="darkbg2")

import torch
import polars as pl
import polars.selectors as cs
import pandas as pd
import pickle
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

from openTSNE import TSNE


def main():
    embed_path = "H:\\pyscripts\\apdc_gpt\\pretrain_noproc_25\\pretrain\\2023-09-10_12-52-51\\embeddings\\"
    e = torch.load(embed_path + 'train_embeddings.pt')
    df = pickle.load(open('cached_data.pq', 'rb'))

    embed = pl.LazyFrame(e.numpy(), schema=[f'feature_{i+1}' for i in range(e.shape[1])])
    embed = embed.with_columns(pl.Series('ppn_int',df.subject_ids).cast(pl.Int32))

    # load one of the parquet files (static)
    data_path = 'H:/apdc'
    static = pl.scan_parquet(data_path+'/static_25.parquet')

    # join based on ppn id
    embed = embed.join(static, on='ppn_int')

    #stratify as < 50 and > 50
    embed = embed.with_columns((pl.col('age_recode') > 50).cast(pl.Int8).alias('age_group'))
    embed = embed.with_columns((pl.col('SEX') == "1" ).cast(pl.Int8).alias('male'))
    embed = embed.with_columns((pl.col('age_group')+pl.col('male')).fill_null(0).alias('category'))
    embed = embed.with_columns(pl.col('age_recode').fill_null(0).cut([10, 30, 50, 60], labels=list(map(str,[0, 1, 2, 3,4]))).cast(pl.Int8).alias('cut'))


    colors = ['red','blue', 'green', 'black', 'orange']
    embed = embed.collect()

    X = embed.select(cs.starts_with('feature_'))#.collect()

    alg = PCA(n_components=2)

    # runs out of memory
    #alg = Isomap(n_components=2, n_neighbors=10)

    # runs out of memory
    #alg = KernelPCA(n_components=2, kernel='rbf')

    alg.fit(X)
    components = alg.transform(X)

    ind_colors = [colors[i] for i in embed.get_column('category')]
    #ind_colors = [colors[int(i)] for i in embed.get_column('cut')]
    #components = TSNE().fit(X.to_numpy())

    #plt.scatter(components[:, 0],components[:, 1], c=['red' if i ==1 else 'green' for i in embed.get_column('age_group')])
    #plt.scatter(components[:, 0],components[:, 1], c=['red' if i == 1 else 'green' for i in embed.get_column('male')])

    plt.scatter(components[:, 0],components[:, 1], c=ind_colors, alpha=0.5)
    # reduce dimension using pca or isomap
    # plot points using scatterplot, where color being the age stratified variable

    plt.savefig('pca_by_category.png', dpi=200)
    #plt.savefig('pca_by_age_stratified.png', dpi=200)
    #plt.savefig('tse_by_sex.png', dpi=200)
    #plt.savefig('tse_by_category.png', dpi=200)
    #plt.savefig('tsne_by_age_stratified.png', dpi=200)
    #pickle.dump(components, open('tsne_components.pickle', 'wb'))


if __name__ == "__main__":
    main()
