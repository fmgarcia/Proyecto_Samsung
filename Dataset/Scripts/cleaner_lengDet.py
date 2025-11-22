import pandas as pd
import re
import emoji
import fasttext
import os

# === 1. CARGAR CSV ORIGINAL ===
ruta_csv = "cyberbullying_tweets_original.csv"  # Cambia si el nombre es distinto
df = pd.read_csv(ruta_csv)

# === 2. LIMPIEZA DE TEXTO ===
def limpiar_texto(texto):
    """Limpia menciones, URLs, emojis y puntuación, pero mantiene hashtags."""
    if not isinstance(texto, str):
        return ""
    
    texto = re.sub(r"@\w+", "", texto)                # Menciones
    texto = re.sub(r"http\S+|www\S+", "", texto)      # URLs
    texto = emoji.replace_emoji(texto, replace="")    # Emojis
    texto = re.sub(r"[^\w\s#]", "", texto)            # Puntuación (mantiene hashtags)
    texto = texto.lower()                             # Minúsculas
    texto = re.sub(r"\s+", " ", texto).strip()        # Espacios
    return texto

df["texto_limpio"] = df["tweet_text"].apply(limpiar_texto)

# === 3. CARGAR MODELO FASTTEXT ===
modelo_path = "lid.176.ftz"
if not os.path.exists(modelo_path):
    raise FileNotFoundError(
        "❌ No se encontró el modelo 'lid.176.ftz'. Descárgalo de:\n"
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    )

modelo = fasttext.load_model(modelo_path)

# === 4. DETECCIÓN DE IDIOMA ===
def detectar_idioma_fasttext(texto):
    """Detecta idioma con FastText. Asume english salvo que se identifique otro."""
    if not isinstance(texto, str) or not texto.strip():
        return "null"  # Texto vacío o nulo
    idioma = modelo.predict(texto)[0][0].replace("__label__", "")
    return "english" if idioma == "en" else "other"

df["idioma"] = df["texto_limpio"].apply(detectar_idioma_fasttext)

# === 5. CSV FINAL ===
resultado = df[["texto_limpio", "cyberbullying_type", "idioma"]]
resultado.to_csv("cyberbullying_limpio_idioma_fasttext.csv", index=False)
print("✅ Archivo generado: cyberbullying_limpio_idioma_fasttext.csv")

# === 6. ESTADÍSTICAS ===
conteo = resultado["idioma"].value_counts(dropna=False)
print("\n📊 Resumen de idiomas detectados:")
print(conteo)

# (Opcional) mostrar los primeros ejemplos
print("\n🔎 Ejemplos de las primeras filas:\n")
print(resultado.head(10))
