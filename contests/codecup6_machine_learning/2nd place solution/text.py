import re
import numpy as np
import pandas as pd

import transformers
import hazm
from cleantext import clean


import gc

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt

import wandb

df = pd.read_csv("../input/code-cup-comments/comments/train.csv", low_memory=False)
test_df = pd.read_csv("../input/code-cup-comments/comments/test.csv", low_memory=False)


df['num_words'] = df['comment'].apply(lambda t: len(hazm.word_tokenize(t)))
df['num_words'].describe()


################
# CLEANING FUNCTIONS FROM https://github.com/hooshvare/parsbert
################

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def cleaning(text):
    text = text.strip()
    
    # regular cleaning
    text = clean(text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="0",
        replace_with_currency_symbol="",
    )

    # cleaning htmls
    text = cleanhtml(text)
    
    # normalizing
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)
    
    # removing wierd patterns
    wierd_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        # u"\u200c"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)
    
    text = wierd_pattern.sub(r'', text)
    
    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    
    return text



test_df['cleaned_comment'] = test_df['comment'].apply(cleaning)
test_df['cleaned_comment_num_words'] = test_df['cleaned_comment'].apply(lambda t: len(hazm.word_tokenize(t)))
test_df['cleaned_comment_num_words'].describe()

# ## Modeling


class CFG:
    batch_size = 16
    num_workers = 2
    device = torch.device("cuda")    



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=None):
        self.dataframe = dataframe
        texts = list(dataframe['cleaned_comment'].values)

        self.encodings = tokenizer(texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=max_length)        
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        
        return item
    
    def __len__(self):
        return len(self.dataframe)


def make_loaders(dataframe, tokenizer, mode="train", max_length=None):
    dataset = TextDataset(dataframe, tokenizer, max_length=max_length)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=CFG.batch_size, 
                                             shuffle=True if mode == "train" else False,
                                             num_workers=CFG.num_workers)
    return dataloader


tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
bert_model = BertForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
bert_model.to(CFG.device);
bert_model.eval();



test_loader = make_loaders(test_df, tokenizer, mode="valid", max_length=128)



all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        preds = bert_model(**batch)
        all_preds.append(preds.logits)



pos_preds = torch.cat(all_preds).softmax(dim=-1).cpu().numpy()[:, 0]


result = pd.DataFrame(
    {'prediction': pos_preds}
)
result.to_csv("output.csv", index=False)

