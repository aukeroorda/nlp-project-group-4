import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, T5Tokenizer


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
            df_variant["inputs"] = df_variant["lemma"] + \
                ' ' + df_variant["features"]
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


def get_dataloader(tokenized_data, batch_size=16, num_workers=4, shuffle=True):
    dataloader = DataLoader(MorphInflectionDataset(tokenized_data),
                            shuffle=shuffle,
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
        t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        config = T5Config(vocab_size=t5_tokenizer.vocab_size,
                          pad_token_id=t5_tokenizer.pad_token_id,
                          eos_token_id=t5_tokenizer.eos_token_id,
                          decoder_start_token_id=t5_tokenizer.convert_tokens_to_ids(["<pad>"])[0],
                          d_model=300)
        morph_inflection_model = T5ForConditionalGeneration(config).to(device)
    return morph_inflection_model


def get_optimizer(morph_inflection_model, learning_rate):
    optimizer = AdamW(morph_inflection_model.parameters(), lr=learning_rate)
    return optimizer


def train_loop(morph_inflection_model,
               train_dataloader,
               optimizer,
               device):
    num_train_batches = len(train_dataloader)
    train_loss_for_epoch = 0.0
    morph_inflection_model.train()

    for input_ids, attention_mask, labels in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = morph_inflection_model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
        loss = outputs.loss
        train_loss_for_epoch += loss
        loss.backward()
        optimizer.step()
    train_loss_for_epoch /= num_train_batches
    return train_loss_for_epoch.cpu().detach().numpy()


def valid_loop(morph_inflection_model,
               valid_dataloader,
               device):
    num_valid_batches = len(valid_dataloader)
    valid_loss_for_epoch = 0.0
    morph_inflection_model.eval()

    with torch.no_grad():
        for input_ids, attention_mask, labels in valid_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = morph_inflection_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             labels=labels)
            loss = outputs.loss
            valid_loss_for_epoch += loss
    valid_loss_for_epoch /= num_valid_batches
    return valid_loss_for_epoch.cpu().detach().numpy()


def train_validation_loop(morph_inflection_model,
                          train_dataloader,
                          valid_dataloader,
                          optimizer,
                          device,
                          dir_path_model,
                          num_epochs=30):
    list_train_losses = []
    list_valid_losses = []
    best_valid_loss = 0

    for epoch in range(num_epochs):
        train_loss_for_epoch = train_loop(
            morph_inflection_model, train_dataloader, optimizer, device)
        valid_loss_for_epoch = valid_loop(
            morph_inflection_model, valid_dataloader, device)

        print(f"epoch: {epoch + 1} / {num_epochs}, train loss: {train_loss_for_epoch:.4f}, validation loss: {valid_loss_for_epoch:.4f}")

        list_train_losses.append(train_loss_for_epoch)
        list_valid_losses.append(valid_loss_for_epoch)
        if epoch != 0:
            if best_valid_loss > valid_loss_for_epoch:
                best_valid_loss = valid_loss_for_epoch
                morph_inflection_model.save_pretrained(dir_path_model)
        else:
            best_valid_loss = valid_loss_for_epoch
    list_train_losses = np.array(list_train_losses)
    list_valid_losses = np.array(list_valid_losses)
    return list_train_losses, list_valid_losses


def generate(model_filepath, df_test, device, max_length=50):
    gen_model = T5ForConditionalGeneration.from_pretrained(
        model_filepath, return_dict=True)
    gen_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_filepath)

    gen_inputs = tokenizer([f"{item}" for item in df_test["inputs"]],
                           return_tensors="pt", padding=True).to(device)

    outputs = gen_model.generate(
        input_ids=gen_inputs["input_ids"],
        attention_mask=gen_inputs["attention_mask"],
        max_length=max_length,
        do_sample=False
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def comparer(df_test, gen_outputs, amount=20):
    df_generated_comparison = pd.DataFrame.from_dict(
        {"Expected": df_test["labels"], "Predicted": gen_outputs})
    print(df_generated_comparison.head(amount))
    return df_generated_comparison


def acc_score(pred: list, gold: list, dec: int=2):
    outcomes = {'correct': [], 'incorrect': []}
    for idx, i in enumerate(pred):
        if i == gold[idx]:
            outcomes['correct'].append([idx, i])
        else:
            outcomes['incorrect'].append([idx, i])

    score = round(len(outcomes['correct']) / len(gold), dec)

    print('The accuracy score is {}'.format(score))
    print('\n\nThe incorrect items are:\n')
    print('idx: pred - gold\n')
    for x, y in outcomes['incorrect']:
        print(f'{x}: {y} - {gold[x]}')
