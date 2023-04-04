import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config


def load_raw_data_as_df(dir_data, which_dataset="german", turkish_large=False, spaces=True):
    header_names = ["lemma", "labels", "features"]

    if which_dataset == "german":
        file_train = os.path.join(dir_data, "deu_600.train")
        file_valid = os.path.join(dir_data, "deu.dev")
        file_test = os.path.join(dir_data, "deu.gold")
    elif which_dataset == "turkish":
        if turkish_large:
            file_train = os.path.join(dir_data, "tur_large.train")
        else:
            file_train = os.path.join(dir_data, "tur_small.train")
        file_valid = os.path.join(dir_data, "tur.dev")
        file_test = os.path.join(dir_data, "tur.gold")

    df_train = pd.read_csv(file_train, sep="\t", names=header_names)
    df_valid = pd.read_csv(file_valid, sep="\t", names=header_names)
    df_test = pd.read_csv(file_test, sep="\t", names=header_names)

    for df_variant in (df_train, df_valid, df_test):
        if spaces:
            df_variant["inputs"] = df_variant["lemma"] + ' ' + df_variant["features"]
        else:
            df_variant["inputs"] = df_variant["lemma"] + df_variant["features"]

    return df_train, df_valid, df_test


class MorphInflectionDataset(Dataset):
    def __init__(self, dict_data):
        self.dict_data = dict_data

    def __len__(self):
        return len(self.dict_data["labels"])

    def __getitem__(self, idx):
        dict_sample = {}
        input_ids = self.dict_data["input_ids"][idx]
        attention_mask = self.dict_data["attention_mask"][idx]
        labels = self.dict_data["labels"][idx]
        return input_ids, attention_mask, labels


def tokenize_function(tokenizer, df_data, input_column_name="inputs"):
    tokenized_dict = {}

    inputs = tokenizer(df_data[input_column_name].to_list(),
                       padding="longest",
                       return_tensors="pt")
    labels = tokenizer(df_data["labels"].to_list(),
                       padding="longest",
                       return_tensors="pt").input_ids

    tokenized_dict["input_ids"] = inputs["input_ids"]
    tokenized_dict["attention_mask"] = inputs["attention_mask"]
    tokenized_dict["labels"] = labels

    return tokenized_dict


def get_tokenized_data(tokenizer, df_train, df_valid, df_test):
    tokenized_train = tokenize_function(tokenizer, df_train)
    tokenized_valid = tokenize_function(tokenizer, df_valid)
    tokenized_test = tokenize_function(tokenizer, df_test)
    return tokenized_train, tokenized_valid, tokenized_test


def get_dataloader(tokenized_data, batch_size=16, num_workers=4):
    dataloader = DataLoader(MorphInflectionDataset(tokenized_data),
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    return dataloader


def get_tokenizer(dir_path_model, model_name="google/byt5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(dir_path_model)
    return tokenizer


def get_byt5_model(device, model_name="google/byt5-small", pretrained=True):
    if pretrained:
        morph_inflection_model = T5ForConditionalGeneration.from_pretrained(
            model_name).to(device)
    else:
        config = T5Config()
        morph_inflection_model = T5ForConditionalGeneration(config).to(device)
    return morph_inflection_model


def get_optimizer(morph_inflection_model, learning_rate):
    optimizer = AdamW(morph_inflection_model.parameters(), lr=learning_rate)
    return optimizer


def train_loop(morph_inflection_model,
               train_dataloader,
               optimizer,
               device,
               dir_path_model,
               num_epochs=20):
    list_train_losses = []
    num_train_batches = len(train_dataloader)

    for epoch in range(num_epochs):
        loss_for_epoch = 0.0
        for input_ids, attention_mask, labels in train_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = morph_inflection_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             labels=labels)
            loss = outputs.loss
            loss_for_epoch += loss
            loss.backward()
            optimizer.step()
        loss_for_epoch /= num_train_batches
        print(f"epoch: {epoch + 1} / {num_epochs}, loss: {loss:.4f}")
        list_train_losses.append(loss_for_epoch.cpu().detach().numpy())

    morph_inflection_model.save_pretrained(dir_path_model)
    return np.array(list_train_losses)
