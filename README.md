# Trabajo Integrador 2: RAG con documentos de Sanidad

## Problema y destinatario

Este sistema permite que delegados, trabajadores y estudiantes consulten
preguntas concretas sobre licencias, adicionales y condiciones laborales en
los Convenios Colectivos de Trabajo 108/75, 103/75 y 122/75. La respuesta se
genera sólo a partir de fragmentos recuperados de esos tres documentos y debe
contrastarse con la evidencia mostrada y con el PDF original.

## Contenido de la entrega

- `Trabajo_Integrador_2_CCT_Sanidad_3Docs.ipynb`: análisis completo y reproducible.
- `app.py`: interfaz local para consultar el RAG.
- `datos/`: los tres documentos reales utilizados por el sistema.
- `requirements.txt`: dependencias de Python.

Los índices Chroma, el entorno virtual y las cachés locales no forman parte de
la entrega. Se reconstruyen automáticamente a partir de los PDF y están
excluidos mediante `.gitignore`.

## Corpus y procedencia

| Archivo | Organismo/procedencia | Fecha o versión | Condición de uso | Ejemplo de consulta |
|---|---|---|---|---|
| `CCT108-Texto Completo.pdf` | Federación de Asociaciones de Trabajadores de la Sanidad Argentina; copia obtenida del sitio oficial [Sanidad](https://sanidad.org.ar/accion-gremial/) | Convenio de 1975; versión publicada en el sitio consultado | Documento publicado en fuente institucional. Verificar los términos del sitio antes de redistribuir fuera de la entrega educativa. | Licencia especial por matrimonio |
| `CCT103-Texto Completo.pdf` | Federación de Asociaciones de Trabajadores de la Sanidad Argentina; copia obtenida del sitio oficial [Sanidad](https://sanidad.org.ar/accion-gremial/) | Convenio de 1975; versión publicada en el sitio consultado | Documento publicado en fuente institucional. Verificar los términos del sitio antes de redistribuir fuera de la entrega educativa. | Adicional de Terapia Intensiva |
| `CCT122-Texto Completo.pdf` | Federación de Asociaciones de Trabajadores de la Sanidad Argentina; copia obtenida del sitio oficial [Sanidad](https://sanidad.org.ar/accion-gremial/) | Convenio de 1975; versión publicada en el sitio consultado | Documento publicado en fuente institucional. Verificar los términos del sitio antes de redistribuir fuera de la entrega educativa. | Licencia especial por matrimonio |

Los PDF versionados en `datos/` son las copias exactas utilizadas para las
pruebas. No contienen datos personales incorporados por este proyecto.


## Requisitos

- Python 3.12.
- Ollama instalado y ejecutándose localmente.
- Los modelos `granite4:latest` y `bge-m3` descargados en Ollama.

## Instalación

```powershell
irm https://ollama.com/install.ps1 | iex (en caso de no tener ollama instalado para windows)
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
ollama pull granite4:latest
ollama pull bge-m3
```

## Uso

Para ejecutar el notebook, abrí esta carpeta en VS Code, seleccioná el entorno
`.venv` como kernel y ejecutá las celdas en orden. El notebook carga los PDF,
compara cortes, reconstruye la colección Chroma y ejecuta cinco consultas de
prueba.

Para iniciar la interfaz Gradio:

```powershell
python app.py
```

La interfaz presenta las consultas y respuestas en formato de conversación.
La evidencia recuperada queda disponible en la sección desplegable
`Auditoría de recuperación`, con el convenio, el nombre del archivo, la página,
los fragmentos y la distancia de búsqueda. La primera ejecución puede tardar
porque crea el índice local.

Para reconstruir el índice, eliminá `chroma_db_cct_sanidad/` y ejecutá de
nuevo el notebook o `python app.py`. El índice se crea desde los
PDF de `datos/`; no se debe versionar. El notebook y la interfaz usan el mismo
corte de `1100` caracteres con `180` de solapamiento, `bge-m3`, colección
Chroma y `k=3`.

## Arquitectura y decisiones

`PyPDFLoader` conserva el archivo y la página de cada documento. Luego
`RecursiveCharacterTextSplitter` genera fragmentos de 1100 caracteres con 180
caracteres de solapamiento: el tamaño amplio reduce el riesgo de separar un
artículo de su inciso o porcentaje. Se comparan también fragmentos de 500/80
para inspeccionar el efecto del corte. `OllamaEmbeddings` con `bge-m3` se usa
para indexar y consultar la colección Chroma. Los tres fragmentos con menor
distancia forman el contexto explícito que recibe `granite4:latest`.

La distancia sólo ordena candidatos; no es una probabilidad de verdad. Las
citas `[1]`, `[2]` y `[3]` permiten vincular la respuesta con los fragmentos
que recibió el modelo, pero no reemplazan la revisión del PDF original.

## Resultados de prueba

El notebook ejecuta y deja visibles tres casos respondibles, uno por convenio,
y dos casos sin respuesta. Los casos sin respuesta preguntan por el salario
mínimo nacional vigente en 2026 y por un hospital recomendado para una
internación, datos que no forman parte del corpus. Para cada caso se imprimen
la fuente esperada, los tres fragmentos recuperados, sus páginas y la
respuesta generada. La abstención se comprueba empíricamente y no se presenta
como una garantía del modelo.

| Tipo | Caso | Resultado esperado |
|---|---|---|
| Respondible | Licencia por matrimonio del CCT 108/75 | Recuperó CCT 108/75, página 8, y respondió 15 días. |
| Respondible | Adicional mensual de Terapia Intensiva del CCT 103/75 | Recuperó CCT 103/75, página 5, y respondió 20% para enfermeras y 10% para auxiliares. |
| Respondible | Comparación de licencia por matrimonio entre CCT 122/75 y CCT 108/75 | Recuperó CCT 108/75, página 8, y CCT 122/75, página 10; respondió 15 y 14 días respectivamente. |
| Sin respuesta | Salario mínimo nacional vigente en 2026 | Recuperó fragmentos irrelevantes de CCT 122/75 y CCT 108/75, pero respondió que no encuentra esa información. |
| Sin respuesta | Hospital recomendado para una internación | Recuperó fragmentos irrelevantes de los tres convenios, pero respondió que no encuentra esa información. |

## Límites

El sistema no es un buscador general ni una fuente legal autónoma. Puede
recuperar fragmentos parecidos pero irrelevantes y el modelo puede interpretar
mal la evidencia o no abstenerse ante una pregunta fuera del corpus. Los PDF
se procesan con extracción de texto; una tabla, columna o página escaneada
podría producir texto incompleto. La página mostrada corresponde al índice
del PDF y debe verificarse contra el documento original.

## Uso de IA

Se utilizó GitHub Copilot durante la revisión y construcción del proyecto para
organizar el código, mejorar la documentación y asegurar el cumplimiento de las consignas. Los modelos del sistema son `bge-m3` para embeddings y `granite4:latest` para generación, ambos ejecutados localmente con Ollama. Se verificaron personalmente los archivos, las páginas citadas, los parámetros del pipeline, las respuestas de prueba y la ejecución sintáctica de la interfaz.

## Verificación

Las respuestas deben contrastarse con los fragmentos mostrados y con el PDF
original. La distancia de Chroma ordena los candidatos recuperados, pero no es
una medida de certeza.
