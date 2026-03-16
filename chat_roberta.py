from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, AutoModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
#import evaluate
import pandas as pd
import numpy as np
import os
import torch
import datetime
import joblib


import sys
sys.path.append("file")

from RobertaFeatureExtractor import RobertaFeatureExtractor
import vectorizing_data
import preprocessing_data


class model:
    def __init__(self):

        repo_dir = snapshot_download(repo_id="Jojo225consulting/Demdem")

        self.base_model = AutoModel.from_pretrained(
            repo_dir + "/save_model/pretrained_embeddings_model",
            add_pooling_layer=False
        )
        self.base_model.eval()
        self.feature_extractor = RobertaFeatureExtractor(self.base_model)  # modèle “vide” avec la même architecture

        self.feature_extractor.load_state_dict(
            torch.load(
                repo_dir + "/save_model/final_embedding_model/feature_extractor.pt",
                map_location=torch.device("cpu")
            )
        )
        self.feature_extractor.eval()
        self.model_svm = joblib.load(
            repo_dir + "/save_model/svm/svm_on_roberta_v5.joblib"
        )

    def load_models(self):
        dataset_test = load_dataset("jingjietan/essays-big5")["test"].select(range(5))
        print("Nom des colonnes du jeu de données train brut : ", dataset_test.column_names)

        class_vectorizing = vectorizing_data.vectorizing_data("C:/Users/etulyon1/Downloads" + "/save_model/tokenizer", self.feature_extractor)
        tokenized_test = dataset_test.map(class_vectorizing.tokenization, batched=True)

        tokenized_test = preprocessing_data.CombineLabels(tokenized_test).dataset_with_labels
        tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        X_test, Y_test = class_vectorizing.extract_embeddings(DataLoader(tokenized_test, batch_size=16, shuffle=True))
        probas_test = self.model_svm.predict_proba(X_test)

        return probas_test


    # load_directory = "C:\\Users\\etulyon1\\Downloads\\drive-download-20260316T100612Z-3-001"
    # print("Bonjour")
    # tokenizer = AutoTokenizer.from_pretrained(load_directory + "/tokenizer")
    # print("Tokenizer chargé avec succès !")
    # base_model = AutoModel.from_pretrained(load_directory + "/pretrained_embeddings_model", add_pooling_layer=False)
    # base_model.eval()

    # feature_extractor = RobertaFeatureExtractor(base_model)  # modèle “vide” avec la même architecture
    # feature_extractor.load_state_dict(torch.load( load_directory + "/final_embedding_model/feature_extractor.pt", map_location=torch.device('cpu')))  # charger les poids entraînés
    # feature_extractor.eval()

    # model_svm = joblib.load(load_directory + "/svm/svm_on_roberta_v5.joblib")


if __name__ == "__main__":
    prediction = model().load_models()
    print("Modèles chargés avec succès !")

    print("prédictions \n", prediction)

# Chargement de données
