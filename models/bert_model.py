import torch
# import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils.preprocessing import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_row_to_text(row):
    return (
        f"Workload demand is {row['Workload_Demand']}. "
        f"Stress level is {row['Stress_Level']}. "
        f"Mental exhaustion level is {row['Mentally_Drained']}. "
        f"Sleep quality is {row['Sleep_Quality']}. "
        f"Sleep consistency is {row['Sleep_Consistency']}. "
        f"Daytime sleepiness is {row['Daytime_Sleepiness']}. "
        f"Wake up feeling is {row['Wake_Up_Feeling']}. "
        f"Social connectedness is {row['Social_Connectedness']}. "
        f"Resting heart rate severity is {row['Resting_HR']}."
    )


class RiskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_bert():
    df = load_data()
    df = df.dropna(subset=["Risk_Level"])

    df["text"] = df.apply(convert_row_to_text, axis=1)

    label_mapping = {"Low Risk": 0, "Mid Risk": 1, "High Risk": 2}
    df["label"] = df["Risk_Level"].map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )

    model.to(device)

    train_dataset = RiskDataset(X_train, y_train, tokenizer)
    test_dataset = RiskDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)

    return {
        "accuracy": accuracy,
        "report": report
    }


if __name__ == "__main__":
    train_bert()