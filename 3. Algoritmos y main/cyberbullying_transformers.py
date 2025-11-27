import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Reutilizamos la clase Dataset
class CyberbullyingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def run_pipeline(model_type, csv_path, text_col, label_col, model_dir_base="./modelos_transformers", use_saved=True, output_image_dir="."):
    """
    Ejecuta pipeline para RoBERTa o BERTweet.
    model_type: 'roberta' o 'bertweet'
    """
    
    if model_type == 'roberta':
        checkpoint = 'roberta-base'
        model_name_friendly = "RoBERTa"
    elif model_type == 'bertweet':
        checkpoint = 'vinai/bertweet-base'
        model_name_friendly = "BERTweet"
    else:
        raise ValueError("Tipo de modelo desconocido")

    model_dir = os.path.join(model_dir_base, model_name_friendly)
    
    print(f"\n--- Iniciando Pipeline para {model_name_friendly} ---")
    
    # 1. Carga de Datos
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {csv_path}")
        
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df[label_col])
    label_map = dict(zip(le.classes_, range(len(le.classes_))))
    print("Mapeo:", label_map)
    
    texts = df[text_col].tolist()
    labels = df['label_encoded'].tolist()
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 2. Tokenización
    print(f"Cargando tokenizer ({checkpoint})...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = CyberbullyingDataset(train_encodings, train_labels)
    test_dataset = CyberbullyingDataset(test_encodings, test_labels)
    
    # 3. Modelo
    num_labels = len(label_map)
    model = None
    model_loaded = False
    
    if use_saved and os.path.exists(model_dir):
        print(f"Cargando modelo desde {model_dir}...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model_loaded = True
        except Exception as e:
            print(f"Error cargando: {e}")
            
    if not model_loaded:
        print(f"Entrenando {model_name_friendly}...")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
        
        training_args = TrainingArguments(
            output_dir=f'./results_{model_type}',
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs_{model_type}',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        # Verificación explícita para el linter
        if model is None:
            raise ValueError("El modelo no se ha inicializado correctamente.")
            
        training_args = TrainingArguments(
            output_dir=f'./results_{model_type}',
            per_device_eval_batch_size=64
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

    # 4. Evaluación
    print("Evaluando...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    print(classification_report(test_labels, y_pred, target_names=target_names))
    
    # 5. Gráficos
    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusión - {model_name_friendly}')
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    
    plot_filename = os.path.join(output_image_dir, f"confusion_matrix_{model_type}.png")
    plt.savefig(plot_filename)
    print(f"Gráfico guardado en {plot_filename}")
    plt.close()
    
    return accuracy_score(test_labels, y_pred)
