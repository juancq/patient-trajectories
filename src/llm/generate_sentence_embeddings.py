import numpy as np
import pandas as pd
import polars as pl
from functools import partial
from loguru import logger
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict
import torch
from tqdm.auto import tqdm  # Use tqdm.auto for automatic notebook/terminal detection
from torch.utils.data import DataLoader, Dataset


def join_sentences(group: pl.Series) -> str:
    if 'he' in group[0]:
        sex = 'male'
    else:
        sex = 'female'
    sentence = f'This is the medical information of a {sex} patient with {len(group)-1} hospital stays.'

    # this would return only high level summary, good baseline
    #return sentence

    sentence = sentence + ' '.join(group)
    return sentence

def row_to_sentence(row: pl.Series, procedure_map: Dict) -> str:
    age = row['age_recode']
    sex = 'male' if row['sex'] == "1" else 'female'
    event_date = row['episode_start_date'].strftime('%Y-%m-%d')
    hospital_type = 'public' if row['hospital_type'] == 1 else 'private'
    procedure = row['block_nump']
    procedure = procedure_map.get(int(procedure), None) if procedure else None
    totlos = row['length_of_stay']

    if sex == 'male':
        pronoun = 'he'
    else:
        pronoun = 'she'

    sentence = f'This is the medical information of a {sex} patient.'
    sentence += f'On {event_date}, {pronoun} was {age:.1f}-years-old and visited a {hospital_type} hospital'
    if procedure: 
        sentence += f' for procedure {procedure.lower()}'

    sentence += f', and the length of stay was {totlos} days.'
    return sentence

# Define a custom Dataset class for efficient batching
class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def generate_embeddings_fastest(sentences, model_name="all-MiniLM-L6-v2", batch_size=256, num_workers=0):
    """
    Generates sentence embeddings for a list of sentences as fast as possible using SentenceTransformers,
    optimized for speed with DataLoader and efficient batching.

    Args:
        sentences (list): A list of strings (sentences).
        model_name (str, optional): The SentenceTransformer model name to use.
                                     "all-MiniLM-L6-v2" is a good balance of speed and quality.
                                     Defaults to "all-MiniLM-L6-v2".
        batch_size (int, optional): The batch size for encoding sentences.
                                     Increase this as much as your GPU memory allows.
                                     Defaults to 256.
        num_workers (int, optional): Number of CPU workers for data loading.
                                      Set to a reasonable value based on your CPU cores (e.g., 4-8).
                                      Defaults to 0 (main process only).

    Returns:
        numpy.ndarray: A NumPy array where each row is the embedding for the corresponding sentence.
    """

    # 1. Check for GPU availability and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the SentenceTransformer model (move to GPU if available)
    model = SentenceTransformer(model_name, device=device)
    model.eval()  # Set to evaluation mode

    # 3. Create a SentenceDataset and DataLoader
    dataset = SentenceDataset(sentences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for embedding generation
        num_workers=num_workers, # Use multiple CPU workers for data loading
        pin_memory=True # Recommended for GPU data loading
    )

    # 4. Initialize an empty list to store embeddings (as tensors on GPU)
    all_embeddings = []

    # 5. Process sentences in batches using DataLoader and model.encode()
    with torch.no_grad():  # Disable gradient calculations for faster inference
        for batch_sentences in tqdm(dataloader, desc="Encoding Batches"):
            batch_embeddings = model.encode(
                batch_sentences,
                batch_size=batch_size, # Redundant here as DataLoader provides batches, but kept for clarity
                show_progress_bar=False,
                convert_to_tensor=True, # Keep embeddings as PyTorch tensors on GPU
                device=device
            )
            all_embeddings.append(batch_embeddings)

    # 6. Concatenate all batch embeddings into a single tensor and then convert to NumPy
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    embeddings_array = embeddings_tensor.cpu().numpy() # Move embeddings to CPU only at the end for NumPy conversion

    return embeddings_array


def main():
    apdc_eddc_path = Path('/mnt/data_volume/apdc/study1')
    project_path = apdc_eddc_path / 'preprocessed' 
    split = 'held_out'
    df = pl.scan_parquet(project_path / f'full_cohort_{split}.parquet')
    sentence_vars = [
        'age_recode', 'sex', 'hospital_type', 'episode_start_date', 'block_nump', 'length_of_stay'
    ]
    df = df.select(pl.col(['ppn_int']+sentence_vars))

    # prediction point path for various tasks
    index_date_path = project_path / 'tasks'

    procedure_map = pd.read_csv('procedure_block_map.csv')
    procedure_map = pd.Series(procedure_map['text'].values, index=procedure_map['block']).to_dict()
    row_to_sentence_partial = partial(row_to_sentence, procedure_map=procedure_map)

    index_date = pl.scan_parquet(index_date_path / f'length_of_stay_two_year.parquet')
    print(index_date.collect())
    print(df.select(pl.col('ppn_int').n_unique()).collect())
    df_window = df.join(index_date, on='ppn_int', coalesce=True)
    print(df_window.select(pl.col('ppn_int').n_unique()).collect())
    df_window = df_window.filter(
        (pl.col('episode_start_date').cast(pl.Date) <= pl.col('index_date'))
        .over('ppn_int')
    )
    print(df_window.select(pl.col('ppn_int').n_unique()).collect())

    df_window = df_window.with_columns(
        pl.struct(['*']).map_elements(row_to_sentence_partial, return_dtype=str).alias('sentence')
    )
    df_window = df_window.group_by('ppn_int').agg(
        pl.col('sentence').map_elements(join_sentences, return_dtype=str).alias('sentence')
    )

    logger.info('Getting embeddings from LLM')

    #df_window = df_window.head(10000).collect(streaming=True)
    df_window = df_window.collect(streaming=True)
    sentences = df_window.get_column('sentence')
    #model_name='minilm_l6'
    model_name='multilingual-e5-large-instruct'

    print(f"Generating embeddings for sentences...")
    embeddings = generate_embeddings_fastest(sentences, 
                    model_name=f'./{model_name}',
                    batch_size=512, num_workers=4) # Increased batch_size and added num_workers

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first sentence):\n{embeddings[0]}")

    df_embeddings = pl.DataFrame(embeddings, 
                schema=[f'feature_{x}' for x in range(embeddings.shape[-1])])
    df_embeddings = df_embeddings.with_columns(df_window.get_column('ppn_int'))
    df_embeddings.write_parquet(index_date_path / f"two_year_{model_name}_embeddings.parquet")
    # Save embeddings (optional)
    #np.save("sentence_embeddings_v2.npy", embeddings)


if __name__ == '__main__':
    main()