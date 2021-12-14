import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, pipeline
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Use pretrained model to label the train data

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

labels = classifier(list(train_data["comment"]))
labels = [1 if item["label"] == "HAPPY" else 0 for item in labels]
train_data["labels"] = labels

# Fine-tune on the labeled data

train_encodings = tokenizer(list(train_data["comment"]), return_tensors="pt", truncation=True, padding=True)
test_encodings = tokenizer(list(test_data["comment"]), return_tensors="pt", truncation=True, padding=True)

from torch.utils.data import Dataset

class SADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self):
        return len(self.data['input_ids'])

train_dataset = SADataset(train_encodings)
test_dataset = SADataset(test_encodings)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()

predictions = []

for batch in tqdm(test_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids, attention_mask=attention_mask)
        predictions += list(torch.sigmoid(output["logits"])[:, 1].cpu().numpy())

prediction = pd.DataFrame(predictions, columns=["prediction"])
prediction.to_csv('output.csv', index=False)
