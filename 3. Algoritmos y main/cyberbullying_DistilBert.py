import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Clase Dataset compatible con PyTorch
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

def run_pipeline(csv_path, text_col, label_col, model_dir="./modelo_ciberbullying_distilbert", use_saved=True, output_image_dir="."):
    """
    Ejecuta el pipeline completo de entrenamiento y evaluación de DistilBERT.
    
    Args:
        csv_path (str): Ruta al archivo CSV.
        text_col (str): Nombre de la columna con el texto de los tweets.
        label_col (str): Nombre de la columna con la etiqueta (target).
        model_dir (str): Directorio donde guardar/cargar el modelo.
        use_saved (bool): Si es True, intenta cargar un modelo existente.
        output_image_dir (str): Directorio donde guardar las imágenes generadas.
    """
    print(f"\n--- Iniciando Pipeline ---")
    print(f"Archivo: {csv_path}")
    print(f"Texto: '{text_col}' | Etiqueta: '{label_col}'")
    
    # Configuración de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Carga de Datos
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {csv_path}")
        
    print("Cargando dataset...")
    df = pd.read_csv(csv_path)
    
    # Limpieza básica: eliminar nulos
    initial_len = len(df)
    df = df.dropna(subset=[text_col, label_col])
    print(f"Filas cargadas: {initial_len}. Filas tras eliminar nulos: {len(df)}")
    
    # Codificación de etiquetas
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df[label_col])
    
    # Mapeo de etiquetas
    label_map = dict(zip(le.classes_, range(len(le.classes_))))
    print("Mapeo de etiquetas detectado:", label_map)
    
    texts = df[text_col].tolist()
    labels = df['label_encoded'].tolist()
    
    # División Train/Test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train set: {len(train_texts)} | Test set: {len(test_texts)}")
    
    # 2. Tokenización
    print("Tokenizando textos...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = CyberbullyingDataset(train_encodings, train_labels)
    test_dataset = CyberbullyingDataset(test_encodings, test_labels)
    
    # 3. Configuración del Modelo
    num_labels = len(label_map)
    model_loaded = False
    model = None
    
    # Intentar cargar modelo guardado
    if use_saved and os.path.exists(model_dir):
        print(f"Buscando modelo guardado en '{model_dir}'...")
        try:
            model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            model_loaded = True
            print("✅ Modelo cargado exitosamente.")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el modelo: {e}")
    
    # Entrenar si no se cargó
    if not model_loaded:
        print("🚀 Iniciando entrenamiento de nuevo modelo...")
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
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
        
        print(f"💾 Guardando modelo en '{model_dir}'...")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        # Si cargamos, configuramos Trainer solo para evaluar
        if model is None:
            raise ValueError("Error crítico: El modelo no se ha inicializado correctamente.")

        training_args = TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=64
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

    # 4. Evaluación
    print("\n--- Evaluación ---")
    results = trainer.evaluate()
    print("Métricas:", results)
    
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = test_labels
    
    # Reporte de Clasificación
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # 5. Gráficos
    print("\nGenerando gráficos...")
    labels_indices = [v for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    labels_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels_indices)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_names, 
                yticklabels=labels_names)
    plt.title('Matriz de Confusión - DistilBERT')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    
    # Asegurar que el directorio de imágenes existe
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        print(f"Directorio creado: {output_image_dir}")

    plot_filename = os.path.join(output_image_dir, "confusion_matrix_result.png")
    plt.savefig(plot_filename)
    print(f"✅ Gráfico guardado como '{plot_filename}'")
    print("--- Proceso Finalizado ---")
    
    return results.get('eval_accuracy', 0.0)
