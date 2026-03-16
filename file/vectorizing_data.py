from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

class vectorizing_data:
  def __init__(self, tokenizer_directory, feature_extractor ):
    self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_directory)
    self.feature_extractor = feature_extractor

    # Tokenization
  def tokenization(self, example, **kwargs):
      return self.tokenizer(example["Loan Title"], truncation=True, padding="max_length", max_length=512)
      

  #Extraction des embeddings
  def extract_embeddings(self, dataloader):
      X, Y = [], []

      with torch.no_grad():
          # Vérifier si labels existent
          first_batch = next(iter(dataloader))
          
          has_labels = "labels" in first_batch.keys()

          for batch in dataloader:
              # Déplacer les entrées sur GPU (sauf labels)
              inputs = {k: v.to("cpu") for k, v in batch.items() if k != "labels"}

              # Extraire les embeddings
              outputs = self.feature_extractor(**inputs)
              # Exemple : CLS token
              #cls_embeddings = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
              X.append(outputs.cpu().numpy())

              # Ajouter les labels si présents
              if has_labels:
                  Y.append(batch["labels"].cpu().numpy())

      # Empiler tous les batches
      X = np.vstack(X)
      if has_labels:
          Y = np.vstack(Y)
          return X, Y
      else:
          return X





