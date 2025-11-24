
# **Exploración y Análisis del Dataset de Ciberacoso**

## **1. Características Estadísticas del Dataset de Ciberacoso**

Este documento recoge los resultados del análisis exploratorio del dataset.

---

### **1. Estructura general del dataset**
- **Total de registros:** 47.692
- **Columnas:** 3
  - `texto_limpio`: Contenido textual del tuit, ya preprocesado.
  - `cyberbullying_type`: Tipo de ciberacoso o etiqueta de clasificación.
  - `idioma`: Idioma detectado del mensaje (predominantemente inglés).
- **Duplicados exactos:** 974 (≈2%).
- **Valores nulos o vacíos:** 266 (≈0.6%).

---

### **2. Distribución de clases**
El dataset contiene seis categorías, todas con frecuencias muy similares, lo que indica una **muestra balanceada artificialmente**.

| Categoría | Frecuencia | Porcentaje |
|------------|-------------|-------------|
| religion | 7.998 | 16.77% |
| age | 7.992 | 16.76% |
| gender | 7.973 | 16.72% |
| ethnicity | 7.961 | 16.69% |
| not_cyberbullying | 7.945 | 16.66% |
| other_cyberbullying | 7.823 | 16.40% |


---

### **3. Distribución por idioma**
| Idioma | Proporción |
|---------|--------------|
| English | 95.9% |
| Other | 4.1% |

El dataset está compuesto principalmente por tuits en inglés, lo que facilita el uso de modelos de lenguaje preentrenados en ese idioma (FastText, Word2Vec, BERT, etc.).

---

### **4. Proporción de bullying por idioma**
| Idioma | Proporción de bullying |
|---------|--------------------------|
| English | 84.7% |
| Other | 54.2% |


---

### **5. Distribución de tipos de acoso en idioma 'other'**
| Categoría | Proporción dentro de 'other' |
|-------------|-------------------------------|
| not_cyberbullying | 45.8% | (883 mensajes, el 11% del total)
| ethnicity | 27.1% |
| other_cyberbullying | 17.9% |
| gender | 7.7% |
| religion | 0.9% |


---

### **6. Textos nulos o vacíos tras la limpieza**
Durante el proceso de limpieza, 266 mensajes quedaron vacíos debido a la eliminación de menciones, URLs, emojis o puntuación. Estos textos, aunque sin contenido útil, conservaron su etiqueta original.

| Categoría | Porcentaje sobre los vacíos |
|------------|-----------------------------|
| other_cyberbullying | 59.4% |
| not_cyberbullying | 32.7% |
| gender | 7.5% |
| ethnicity | 0.4% |

> La mayoría de los textos vacíos pertenecen a la clase `other_cyberbullying`, lo que sugiere que muchos mensajes de esta categoría eran triviales o carentes de contenido textual tras la limpieza.

---

### **7. Evidencia de categorización léxica**
El análisis de similitud léxica confirma que las clases se diferencian por vocabulario.

| Categoría | not_cyberbullying | other_cyberbullying | gender | religion | ethnicity | age |
|------------|-------------------|---------------------|---------|-----------|------------|-----|
| **not_cyberbullying** | **1.00** | 0.74 | 0.33 | 0.21 | 0.18 | 0.11 |
| **other_cyberbullying** | 0.74 | **1.00** | 0.40 | 0.28 | 0.20 | 0.14 |
| **gender** | 0.33 | 0.40 | **1.00** | 0.25 | 0.13 | 0.12 |
| **religion** | 0.21 | 0.28 | 0.25 | **1.00** | 0.17 | 0.12 |
| **ethnicity** | 0.18 | 0.20 | 0.13 | 0.17 | **1.00** | 0.15 |
| **age** | 0.11 | 0.14 | 0.12 | 0.12 | 0.15 | **1.00** |

> **Conclusión:** El vocabulario fue el criterio de clasificación. Las categorías se agrupan por palabras clave temáticas, no por significado o contexto.

---

## **Resumen del análisis del dataset de ciberacoso**

A continuación se resumen las principales conclusiones del análisis exploratorio y crítico del dataset `cyberbullying_limpio_idioma_fasttext.csv`. Este resumen sintetiza los hallazgos más relevantes para orientar las decisiones del equipo en las siguientes fases del proyecto.

---

### **1. Etiquetado basado en criterios léxicos, no en significado**
El análisis de similitud léxica entre categorías muestra que las etiquetas del dataset se asignaron en función de las **palabras presentes en los tweets**, no del significado o intención del mensaje. Cada categoría (por ejemplo, `religion`, `gender`, `ethnicity`, `age`) está definida por un conjunto de vocablos específicos asociados con insultos o expresiones temáticas. No hay evidencia de que el proceso de etiquetado haya tenido en cuenta el contexto, la ironía o la relación entre usuarios.

**Conclusión:** el dataset mide la presencia de lenguaje ofensivo temático, no el acoso como fenómeno comunicativo o psicológico.

---

### **2. Etiquetado automático, no humano**
El patrón de distribución de las clases (todas con proporciones cercanas al 16%) y la homogeneidad léxica dentro de cada categoría indican que el etiquetado fue **automático o semiautomático**, probablemente mediante listas de palabras clave. En un etiquetado manual, las proporciones serían desiguales y habría más variabilidad en la forma lingüística de los mensajes.

**Conclusión:** las etiquetas fueron asignadas por un sistema de reglas o detección de términos, no por anotadores humanos que evaluaran intención o contexto.

---

### **3. Distribución de los mensajes de otros idiomas**
Los mensajes escritos en otros idiomas tienen una distribución distinta a la del dataset. Si decidieramos eliminarlos estaríamos alterando la distribución original. Especialmente hay que tener en cuenta que en proporción hay muchos más mensajes de ausencia de ciberbullying en otros idiomas que en inglés (un 11% del total), por lo que perderíamos bastantes datos del grupo control. 

**Conclusión:** hay que tener cuidado a la hora de eliminar los mensajes de la categoría other. 

---

### **4. Distribución artificial y no generalizable**
El dataset está **equilibrado de forma sintética**, es decir, cada tipo de ciberacoso tiene aproximadamente el mismo número de ejemplos. En la realidad, la proporción de mensajes con acoso es muy baja (menos del 5%), por lo que este equilibrio no representa el entorno real de las redes sociales.

**Conclusión:** el modelo entrenado con estos datos no podrá generalizar bien a escenarios reales, donde el acoso es minoritario.

---

### **5. Dificultad para aplicar técnicas no supervisadas**
El equilibrio artificial y la homogeneidad del vocabulario dificultan el uso de técnicas de aprendizaje no supervisado, como clustering o análisis de tópicos. Estas técnicas buscan patrones naturales en los datos, pero al haber sido forzada la distribución, los agrupamientos reflejarán la estructura artificial del dataset en lugar de relaciones semánticas genuinas.

**Conclusión:** los modelos no supervisados (K-Means, LDA, etc.) producirán resultados sesgados y poco interpretables con esta base de datos.

---

### **6. Precaución con el término “bullying”**
Aunque el dataset se denomina de “ciberacoso”, **nada garantiza que los tweets etiquetados describan o constituyan un caso de acoso real**. Muchos mensajes son simplemente ofensivos, provocadores o irónicos, sin evidencia de comportamiento reiterado o intención de daño. El concepto de bullying implica repetición, desequilibrio de poder y contexto interpersonal, elementos que este corpus no recoge.

**Conclusión:** los modelos desarrollados detectarán **lenguaje potencialmente abusivo o tóxico**, no ciberacoso en sentido estricto. El término “detección de lenguaje ofensivo” sería más preciso.

---

### **7. Dificultad para diferenciar bullying frente a su ausencia**
Debido a que la proporción de mensajes “no bullying” es pequeña (≈16%) y muchos de ellos comparten vocabulario con las categorías de acoso, **será difícil entrenar un modelo fiable para distinguir entre acoso y no acoso**. Esto aumenta el riesgo de falsos positivos y limita la aplicabilidad del modelo fuera del contexto del dataset.

**Conclusión:** será necesario aplicar técnicas de balanceo, ajuste de umbrales o ponderación de clases si se desea construir un modelo binario (bullying vs. no bullying).

---

### **Resumen final**
El dataset es adecuado para estudiar el lenguaje ofensivo explícito mediante aprendizaje supervisado, pero presenta **limitaciones metodológicas y conceptuales significativas** para detectar acoso real. Cualquier modelo entrenado con estos datos debe interpretarse como un detector de lenguaje abusivo, no como un identificador de comportamientos de ciberacoso.

