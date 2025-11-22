import pandas as pd
import re
from wordcloud import STOPWORDS
from nltk.stem import WordNetLemmatizer
import nltk
from googletrans import Translator # pip install googletrans==4.0.0-rc1
import time

# 1. Configuración Inicial y Descargas necesarias
nltk.download('wordnet')
nltk.download('omw-1.4')

# Definición de Stopwords y Regex
STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 'im', 'll', 'y', 've', 'u', 'ur', 'don', 't', 's'])
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# Funciones de Limpieza
def clean_text(text):
    # Conversión a minúsculas
    text = text.lower()
    # Eliminar menciones, links y caracteres especiales
    text = re.sub(TEXT_CLEANING_RE, ' ', text)
    # Eliminar stopwords
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    return text

# Función de Lematización
lemmatizer = WordNetLemmatizer()
def lemmatizer_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Función de Traducción (con manejo de errores)
translator = Translator()
def translate_batch(texts, dest='es'):
    translations = []
    try:
        # googletrans puede manejar listas
        results = translator.translate(texts, dest=dest)
        if isinstance(results, list):
            translations = [res.text for res in results]
        else:
            translations = [results.text]
    except Exception as e:
        print(f"Error en lote: {e}. Intentando uno por uno...")
        # Fallback: uno por uno si falla el lote
        for text in texts:
            try:
                res = translator.translate(text, dest=dest)
                translations.append(res.text)
            except Exception as inner_e:
                print(f"Error traduciendo '{text}': {inner_e}")
                translations.append(text) # Devolver original si falla
    return translations

# --- PROCESO ---

# Configuración de lotes
BATCH_SIZE = 20 # Tamaño del lote
DELAY_SECONDS = 2 # Segundos de espera entre lotes para evitar bloqueo
OUTPUT_FILE = 'cyberbullying_tweets_es_partial.csv'
FINAL_FILE = 'cyberbullying_tweets_es.csv'

# Cargar el dataset
print("Cargando dataset...")
df = pd.read_csv('./Dataset/cyberbullying_tweets.csv') 

# Limpieza y Lematización (esto es rápido, se puede hacer siempre)
print("Limpiando y Lematizando...")
df['tweet_text_clean'] = df['tweet_text'].apply(clean_text)
df['tweet_text_clean'] = df['tweet_text_clean'].apply(lemmatizer_words)

# Inicializar columna de traducción si no existe
if 'translated_text' not in df.columns:
    df['translated_text'] = None

# Comprobar si hay un archivo parcial para reanudar
start_index = 0
import os
if os.path.exists(OUTPUT_FILE):
    print(f"Encontrado archivo parcial '{OUTPUT_FILE}'. Reanudando...")
    df_partial = pd.read_csv(OUTPUT_FILE)
    # Asumimos que el archivo parcial tiene las mismas filas en orden o actualizamos
    # Una forma segura es actualizar el df principal con lo que ya tenemos
    # Pero para simplificar, si el parcial tiene menos filas, cogemos esas traducciones
    
    # Mejor estrategia: Ver cuántos no son nulos en el parcial y empezar desde ahí
    # Si el parcial tiene el mismo tamaño que el original:
    if len(df_partial) == len(df):
        df = df_partial
        # Buscar el primer índice donde translated_text es nulo
        missing_translations = df['translated_text'].isnull()
        if missing_translations.any():
            start_index = int(missing_translations.idxmax())
        else:
            start_index = len(df) # Todo traducido
            print("Parece que ya está todo traducido.")
    else:
        print("El archivo parcial no coincide en tamaño. Empezando de cero (o ajusta la lógica si es necesario).")

print(f"Comenzando traducción desde el índice {start_index} de {len(df)}...")

total_rows = len(df)

for i in range(start_index, total_rows, BATCH_SIZE):
    batch_end = min(i + BATCH_SIZE, total_rows)
    batch_indices = list(range(i, batch_end))
    
    # Obtener textos del lote (usamos el texto limpio o el original según prefieras, aquí uso el limpio)
    batch_texts = df.loc[batch_indices, 'tweet_text_clean'].tolist()
    
    # Traducir
    print(f"Traduciendo lote {i} - {batch_end} ({round((i/total_rows)*100, 2)}%)...")
    translated_batch = translate_batch(batch_texts)
    
    # Asignar resultados
    df.loc[batch_indices, 'translated_text'] = translated_batch
    
    # Guardar progreso cada lote (o cada X lotes)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Esperar para respetar límites de API
    time.sleep(DELAY_SECONDS)

# Guardar Resultado Final
df.to_csv(FINAL_FILE, index=False)
print(f"Proceso finalizado. Archivo guardado como '{FINAL_FILE}'")