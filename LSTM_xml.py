import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# Custom Dataset class for XML
class XMLDataset(Dataset):
    def __init__(self, xml_texts, labels, vocab):
        self.xml_texts = xml_texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.xml_texts)

    def __getitem__(self, idx):
        xml_text = self.xml_texts[idx]
        label = self.labels[idx]
        tokenized_xml = torch.tensor([self.vocab.get(token, self.vocab["<unk>"]) for token in tokenize_xml(xml_text)], dtype=torch.long)
        return tokenized_xml, torch.tensor(label, dtype=torch.float)

# Tokenize XML (simplified tokenizer)
def tokenize_xml(xml_str):
    # Replace < and > for simplicity and split by space
    tokens = xml_str.replace('<', ' ').replace('>', ' ').split()
    return tokens

# Sample dataset (replace this with your actual dataset)
xml_texts = [
    "<plan><name>Basic</name><expiration>2025-01-01</expiration></plan>",
    "<plan><name>Premium</name><expiration>2024-06-01</expiration></plan>",
    # Add your actual XML text data here...
]

labels = [1, 0]  # 1 if file needs testing, 0 otherwise

# Build the vocabulary manually
def build_vocab(xml_texts):
    token_counter = Counter()
    for xml in xml_texts:
        tokens = tokenize_xml(xml)
        token_counter.update(tokens)
    
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(token_counter.items())}  # Starting from 2 (0 reserved for padding, 1 for unknown)
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab

vocab = build_vocab(xml_texts)

# Convert XMLs into padded sequences
def collate_fn(batch):
    xml_tensors, labels = zip(*batch)
    xml_tensors_padded = pad_sequence(xml_tensors, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.stack(labels)
    return xml_tensors_padded, labels

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(xml_texts, labels, test_size=0.2, random_state=42)

# Create Dataset and DataLoader
train_dataset = XMLDataset(X_train, y_train, vocab)
test_dataset = XMLDataset(X_test, y_test, vocab)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# LSTM Model Definition
class XMLLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(XMLLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Binary classification
        
    def forward(self, x):
        embeds = self.embedding(x)
        _, (hn, _) = self.lstm(embeds)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

# Hyperparameters
vocab_size = len(vocab)
embed_size = 128
hidden_size = 256
output_size = 1  # Binary classification: 1 or 0

# Initialize model, loss function, and optimizer
model = XMLLSTMClassifier(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xml_batch, label_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(xml_batch)
        loss = criterion(outputs.squeeze(), label_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xml_batch, label_batch in test_loader:
        outputs = model(xml_batch)
        predicted = (outputs.squeeze() > 0.5).float()
        total += label_batch.size(0)
        correct += (predicted == label_batch).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
