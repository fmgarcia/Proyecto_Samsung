import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import re

# --- Modelos DL ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden)

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text) # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) # [batch size, 1, sent len, emb dim]
        
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# --- Utilidades ---
def tokenize(text):
    return re.findall(r'\w+', text.lower())

def build_vocab(texts, max_size=20000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(str(text)))
    most_common = counter.most_common(max_size)
    vocab = {word: i+2 for i, (word, _) in enumerate(most_common)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def text_pipeline(text, vocab, max_len=100):
    tokens = tokenize(str(text))
    indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices, min(len(tokens), max_len)

class DLDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices, length = text_pipeline(text, self.vocab)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(length, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def run_pipeline(model_type, csv_path, text_col, label_col, model_dir_base="./modelos_dl", use_saved=True, output_image_dir="."):
    print(f"\n--- Iniciando Pipeline DL: {model_type} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Datos
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df[label_col])
    label_map = dict(zip(le.classes_, range(len(le.classes_))))
    
    texts = df[text_col].tolist()
    labels = df['label_encoded'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Vocabulario
    vocab = build_vocab(X_train)
    vocab_size = len(vocab)
    print(f"Vocabulario: {vocab_size} palabras")
    
    train_ds = DLDataset(X_train, y_train, vocab)
    test_ds = DLDataset(X_test, y_test, vocab)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    # Configuración Modelo
    output_dim = len(label_map)
    embed_dim = 100
    hidden_dim = 256
    
    if model_type == 'LSTM':
        model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, bidirectional=False, dropout=0.5)
    elif model_type == 'Bi-LSTM':
        model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, bidirectional=True, dropout=0.5)
    elif model_type == 'CNN':
        model = CNNClassifier(vocab_size, embed_dim, n_filters=100, filter_sizes=[3,4,5], output_dim=output_dim, dropout=0.5)
    else:
        raise ValueError("Modelo desconocido")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model_dir = os.path.join(model_dir_base, model_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "model.pt")
    
    if use_saved and os.path.exists(model_path):
        print("Cargando modelo guardado...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Entrenando...")
        for epoch in range(5): # 5 épocas demo
            model.train()
            total_loss = 0
            for text, length, label in train_loader:
                text, length, label = text.to(device), length.to(device), label.to(device)
                optimizer.zero_grad()
                if model_type == 'CNN':
                    output = model(text)
                else:
                    output = model(text, length)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), model_path)
        
    # Evaluación
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text, length, label in test_loader:
            text, length, label = text.to(device), length.to(device), label.to(device)
            if model_type == 'CNN':
                output = model(text)
            else:
                output = model(text, length)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
    print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))
    
    # Gráfico
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
    plt.title(f'Matriz de Confusión - {model_type}')
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    plt.savefig(os.path.join(output_image_dir, f"confusion_{model_type}.png"))
    plt.close()
    
    return accuracy_score(all_labels, all_preds)
