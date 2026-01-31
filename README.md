# Clasificación de texto (SVM + Transformers)

Este proyecto ejecuta distintos pipelines de clasificación para comentarios en español, usando:

- `SVM` con TF-IDF.
- `SVM extendido` con TF-IDF + puntuaciones de léxicos.
- `Zero-shot` con Transformers (`facebook/bart-large-mnli`).
- `Fine-tuning` de Transformers (`FacebookAI/xlm-roberta-base`).

## 1) Requisitos previos

- Python 3.10+ (recomendado).
- `pip` actualizado.
- Conexión a Internet en la primera ejecución (descarga de modelos/NLTK).
- Ejecutar siempre desde la raíz del proyecto (`playground/`).

> Nota: El preprocesador descarga recursos de NLTK automáticamente en runtime.

## 2) Instalación

Desde la carpeta raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Si no existe el directorio de salida, créalo:

```bash
mkdir -p target
```

## 3) Estructura de datos esperada

- Dataset de entrada: `data/dataset_raw.csv`
- Léxicos:
	- `data/gaming.txt`
	- `data/technology.txt`

En la primera ejecución, el programa genera automáticamente archivos procesados y particiones de train/test en `target/`.

## 4) Cómo ejecutar

El script principal es `main.py` y **requiere** el argumento `--model` (o `-m`).

### Opciones válidas de `--model`

- `svm`
- `svm_ext`
- `zs_transformers`
- `ft_transformers`
- `all`

### Ejemplos de ejecución

Ejecutar solo SVM:

```bash
python main.py -m svm
```

Ejecutar SVM extendido:

```bash
python main.py -m svm_ext
```

Ejecutar zero-shot Transformers:

```bash
python main.py -m zs_transformers
```

Ejecutar fine-tuning Transformers:

```bash
python main.py -m ft_transformers
```

Ejecutar todo en cascada:

```bash
python main.py -m all
```

También puedes pasar varios modelos en una sola ejecución:

```bash
python main.py -m svm svm_ext
```

## 5) ¿Qué hace internamente al ejecutar?

1. Genera datasets procesados si no existen.
2. Genera split train/test si no existe.
3. Entrena/evalúa el/los modelos seleccionados.
4. Muestra métricas de clasificación y matriz de confusión.

## 6) Salidas y checkpoints

- Archivos procesados y splits en `target/`.
- Checkpoints del fine-tuning en `target/ft_model/`.
	- Si existe `target/ft_model/checkpoint-600`, se reutiliza para evitar reentrenar.

## 7) Errores frecuentes

- **`error: the following arguments are required: -m/--model`**
	- Solución: ejecutar con `-m`, por ejemplo `python main.py -m svm`.

- **No encuentra `data/dataset_raw.csv`**
	- Solución: ejecutar desde la raíz del proyecto y verificar que el archivo exista en `data/`.

- **Descarga de modelos lenta o fallida**
	- Solución: revisar conexión a Internet y reintentar.

- **Sin memoria en GPU (Transformers)**
	- Solución: cerrar procesos GPU, usar solo `svm`/`svm_ext`, o ejecutar en CPU.
