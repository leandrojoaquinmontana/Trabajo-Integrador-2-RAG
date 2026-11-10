# Trabajo Integrador 2: RAG con documentos de Sanidad

RAG local para consultar los Convenios Colectivos de Trabajo 108/75, 103/75 y
122/75. El proyecto permite inspeccionar la extracción de texto, comparar
fragmentos, realizar búsqueda vectorial y generar respuestas respaldadas por
las fuentes recuperadas.

## Contenido de la entrega

- `Trabajo_Integrador_2_CCT_Sanidad_3Docs.ipynb`: análisis completo y reproducible.
- `interfaz_gradio.py`: interfaz local para consultar el RAG.
- `datos/`: los tres documentos reales utilizados por el sistema.
- `requirements.txt`: dependencias de Python.

Los índices Chroma, el entorno virtual y las cachés locales no forman parte de
la entrega. Se reconstruyen automáticamente a partir de los PDF y están
excluidos mediante `.gitignore`.

## Requisitos

- Python 3.12.
- Ollama instalado y ejecutándose localmente.
- Los modelos `granite4:latest` y `bge-m3` descargados en Ollama.

## Instalación

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
ollama pull granite4:latest
ollama pull bge-m3
```

## Uso

Para ejecutar el notebook, abrí esta carpeta en VS Code, seleccioná el entorno
`.venv` como kernel y ejecutá las celdas en orden.

Para iniciar la interfaz Gradio:

```powershell
python interfaz_gradio.py
```

La interfaz muestra por separado la respuesta del modelo y los fragmentos
recuperados, con el convenio, el nombre del archivo, la página y la distancia
de búsqueda. La primera ejecución puede tardar porque crea el índice local.

## Verificación

Las respuestas deben contrastarse con los fragmentos mostrados y con el PDF
original. La distancia de Chroma ordena los candidatos recuperados, pero no es
una medida de certeza.
