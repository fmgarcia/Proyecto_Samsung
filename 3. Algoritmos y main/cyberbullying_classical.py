import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_model(model_name, num_labels):
    """Retorna la instancia del modelo según el nombre."""
    if model_name == 'Naive Bayes':
        return MultinomialNB()
    elif model_name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000)
    elif model_name == 'SVM':
        return SVC(probability=True)
    elif model_name == 'Random Forest':
        # Reducimos profundidad y estimadores para evitar overfitting y lentitud en alta dimensionalidad
        return RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
    elif model_name == 'XGBoost':
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    elif model_name == 'LightGBM':
        return LGBMClassifier(n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

def run_pipeline(model_name, csv_path, text_col, label_col, model_dir_base="./modelos_clasicos", use_saved=True, output_image_dir="."):
    print(f"\n--- Iniciando Pipeline para {model_name} ---")
    
    # Directorio específico para este modelo
    model_dir = os.path.join(model_dir_base, model_name.replace(" ", "_"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "model.joblib")
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    encoder_path = os.path.join(model_dir, "encoder.joblib")

    # 1. Carga de Datos
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {csv_path}")
        
    print("Cargando dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    
    # Codificación de etiquetas
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df[label_col])
    label_map = dict(zip(le.classes_, range(len(le.classes_))))
    print("Mapeo de etiquetas:", label_map)
    
    texts = df[text_col].tolist()
    labels = df['label_encoded'].tolist()
    
    # División Train/Test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 2. Vectorización (TF-IDF)
    # Para modelos de árboles (RF, XGB, LGBM) limitamos features para controlar dimensionalidad
    max_features = 5000 if model_name in ['Random Forest', 'XGBoost', 'LightGBM'] else 10000
    
    if use_saved and os.path.exists(vectorizer_path):
        print("Cargando vectorizador guardado...")
        vectorizer = joblib.load(vectorizer_path)
        X_train = vectorizer.transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
    else:
        print(f"Entrenando vectorizador (max_features={max_features})...")
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(le, encoder_path)

    # 3. Modelo
    model = None
    if use_saved and os.path.exists(model_path):
        print(f"Cargando modelo guardado desde {model_path}...")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            
    if model is None:
        print(f"Entrenando {model_name}...")
        model = get_model(model_name, len(label_map))
        model.fit(X_train, y_train)
        print(f"Guardando modelo en {model_path}...")
        joblib.dump(model, model_path)
        
    # 4. Evaluación
    print("\n--- Evaluación ---")
    # Convertir explícitamente a array numpy para satisfacer al linter y asegurar compatibilidad
    y_pred = np.array(model.predict(X_test))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 5. Gráficos
    print("\nGenerando gráficos...")
    labels_indices = [v for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    cm = confusion_matrix(y_test, y_pred, labels=labels_indices)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        
    plot_filename = os.path.join(output_image_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(plot_filename)
    print(f"✅ Gráfico guardado como '{plot_filename}'")
    plt.close()
    
    return acc
