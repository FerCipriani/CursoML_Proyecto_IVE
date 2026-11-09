# CursoML_Proyecto_IVE

Proyecto integrador de Machine Learning: **predecir el voto de un diputado a partir de cómo
argumentó en el recinto**.

Los datos provienen del debate parlamentario argentino sobre la Interrupción Voluntaria del
Embarazo (IVE) de 2020, sancionada después como
[Ley 27.610](https://servicios.infoleg.gob.ar/infolegInternet/verNorma.do?id=346231). Cada
discurso fue procesado con un modelo de tópicos y reducido a un vector de 7 proporciones. Sobre
esos vectores se entrena un clasificador.

```
discurso  →  [MALLET / LDA]  →  7 proporciones  →  [clasificador]  →  AFI / NEG
              modelo 1                                  modelo 2
```

---

## Los cuadernos

| Cuaderno | Qué hace | Abrir |
|---|---|---|
| **Parte 1 — Análisis y limpieza** | Carga, EDA, auditoría de calidad, transformaciones, selección de variables y división en entrenamiento y prueba | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FerCipriani/CursoML_Proyecto_IVE/blob/main/Curso_ML_P1_Analisis_y_limpieza.ipynb) |
| **Parte 2 — Modelos de predicción** | Validación cruzada, fuga de información, sobreajuste, cuatro modelos y comparativa final | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FerCipriani/CursoML_Proyecto_IVE/blob/main/Curso_ML_P2_Modelos_de_Prediccion.ipynb) |

**No hace falta descargar ni configurar nada.** El cuaderno lee el dataset directamente desde
este repositorio. Se abre con el botón de arriba y se ejecuta con *Entorno de ejecución →
Ejecutar todo*.

Si querés trabajar sobre él, en Colab hacé *Archivo → Guardar una copia en Drive*: vas a estar
editando tu propia copia, no la de este repositorio.

---

## Los archivos

| Archivo | Qué es |
|---|---|
| `Curso_ML_P1_Analisis_y_limpieza.ipynb` | Cuaderno de la Parte 1 |
| `Curso_ML_P2_Modelos_de_Prediccion.ipynb` | Cuaderno de la Parte 2 |
| `doc-topics_K07_it1500_oi0_ob0_a1_b0.01_s12345.xlsx` | El dataset: 171 filas × 8 columnas |
| `dataset_limpio.csv` | Salida de la Parte 1, entrada de la Parte 2. La Parte 2 lo lee directamente desde este repositorio |

### El dataset

171 discursos de la Cámara de Diputados, con dos clases:

| Clase | Casos | |
|---|---|---|
| `AFI` | 95 | voto afirmativo |
| `NEG` | 76 | voto negativo |

Las siete columnas `Topico 0` a `Topico 6` son proporciones que **suman 1 en cada fila**. El
nombre del archivo codifica los parámetros del modelo de tópicos que las generó: 7 tópicos, 1500
iteraciones, α=1, β=0,01, semilla 12345.

---

## Fuentes

Todas oficiales, de la Honorable Cámara de Diputados de la Nación:

- [Diario de Sesiones — versión taquigráfica](https://www3.hcdn.gob.ar/dependencias/dtaquigrafos/diarios/periodo-138/diario_2020121017.pdf)
- [Transcripción HTML de la sesión](https://www4.hcdn.gob.ar/sesionesxml/provisorias/138-17.htm)
- [Votación nominal — Acta 1, O.D. 352](https://votaciones.hcdn.gob.ar/votacion/4077)
- [Orden del Día Nº 352](https://www4.hcdn.gob.ar/dependencias/dcomisiones/periodo-138/138-352.pdf)

## Herramientas

- [MALLET](https://github.com/mimno/Mallet) — modelo de tópicos
  ([LDA](https://www.jmlr.org/papers/v3/blei03a.html))
- Python, pandas, scikit-learn, matplotlib, seaborn, plotly

---

## Correr los cuadernos en tu máquina

En Colab no hace falta instalar nada. Fuera de Colab:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install pandas numpy matplotlib seaborn plotly scikit-learn openpyxl jupyterlab
```

En la celda del punto **0.2.1** del cuaderno, comentá la opción A y descomentá la **D** (archivo
local).
