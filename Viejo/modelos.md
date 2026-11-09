# Cómo funciona la Parte 2B por dentro

Guía para seguir, de punta a punta, lo que le pasa a cualquier modelo del cuaderno
`Curso_ML_P2B_Los_modelos.ipynb`: qué variables existen, quién las crea, quién las usa y en qué
orden. Sirve también como receta para agregar un modelo nuevo.

---

## Índice

1. [La idea en una línea](#1-la-idea-en-una-línea)
2. [Las cuatro capas](#2-las-cuatro-capas)
3. [Capa 1: el estado (Parte 0)](#3-capa-1-el-estado-parte-0)
4. [Capa 2: la maquinaria (punto 7)](#4-capa-2-la-maquinaria-punto-7)
5. [Capa 3: la secuencia de un modelo, paso a paso](#5-capa-3-la-secuencia-de-un-modelo-paso-a-paso)
6. [Ejemplo completo: XGBoost](#6-ejemplo-completo-xgboost)
7. [Segundo ejemplo: la SVM y en qué se diferencia](#7-segundo-ejemplo-la-svm-y-en-qué-se-diferencia)
8. [Ficha de los seis modelos](#8-ficha-de-los-seis-modelos)
9. [Capa 4: la comparativa (punto 12)](#9-capa-4-la-comparativa-punto-12)
10. [Receta: agregar un modelo nuevo](#10-receta-agregar-un-modelo-nuevo)
11. [Trampas conocidas](#11-trampas-conocidas)

---

## 1. La idea en una línea

Se separa **qué modelos se evalúan** (el catálogo) de **cómo se los evalúa** (una única función),
y todo lo que se mide se acumula en una tabla que la comparativa final lee sin saber cuántos
modelos hay.

```
CATALOGO_MODELOS  ──►  evaluar_modelo()  ──►  RESULTADOS / PREDICCIONES  ──►  punto 12
 (qué se evalúa)       (cómo se evalúa)        (lo que se midió)               (lo compara)
```

---

## 2. Las cuatro capas

Cada capa deja variables globales que la siguiente usa. Ninguna capa modifica las anteriores.

```
 CAPA 1 · PARTE 0          CAPA 2 · PUNTO 7            CAPA 3 · PUNTOS 8–11c        CAPA 4 · PUNTO 12
 Estado                    Maquinaria                  Modelos                      Comparativa
 (datos y decisiones)      (cómo se evalúa)            (qué se evalúa)              (solo lee)

 SEMILLA, N_PLIEGUES ──┐   ESCALADORES                 búsqueda propia              maestra = DataFrame(RESULTADOS)
 CLASE_POSITIVA        │   METRICAS_CV                  (busqueda_xgb, …)                  │
 umbrales              ├─► CATALOGO_MODELOS ◄────────── registrar_modelo(...)              ▼
 X, y, predictoras     │   construir_pipeline()                │                    mejores_por_modelo
 cv  ──────────────────┘   calcular_metricas()                 ▼                           │
 linea_base                diagnosticar_ajuste()        evaluar_modelo(nombre) ──► RESULTADOS (filas)
 COLOR_CLASE               evaluar_modelo() ───────────────────────────────────► PREDICCIONES (vectores)
```

---

## 3. Capa 1: el estado (Parte 0)

Se crea una sola vez, al principio, y después **solo se lee**.

| Variable | Celda | Qué contiene | Quién la usa |
|---|---|---|---|
| `RUTA_DATOS`, `NOMBRE_ARCHIVO` | 0.2 | Dónde está `datos_modelado.csv` | La lectura |
| `df` | 0.2 | El CSV tal cual: `TAG`, `T0`…`T6`, `y` | 0.4 |
| `SEMILLA` | 0.3 | `65`. Fija todo el azar | `cv` y el `random_state` de cada modelo |
| `CLASE_POSITIVA` | 0.3 | `"AFI"` | Verificación del 0.4, etiquetas de reportes |
| `N_PLIEGUES` | 0.3 | `10` | `cv`, gráficos por pliegue, textos |
| `BRECHA_SOBREAJUSTE` | 0.3 | `0.10` | `diagnosticar_ajuste()`, gráficos del 12.6 y de las búsquedas |
| `DESVIO_INESTABLE` | 0.3 | `0.08` | `diagnosticar_ajuste()` |
| `SCORE_SUBAJUSTE` | 0.3 | `0.70` | `diagnosticar_ajuste()` |
| `objetivo`, `COLUMNA_CODIFICADA` | 0.4 | `"TAG"` y `"y"` | La construcción de `X` e `y` |
| `predictoras` | 0.4 | `["T0", …, "T6"]` | Todo gráfico que nombra variables |
| `X` | 0.4 | DataFrame 171 × 7, **sin escalar** | Toda evaluación y búsqueda |
| `y` | 0.4 | Serie de 0/1 (1 = `AFI`) | Toda evaluación y búsqueda |
| `clase_negativa` | 0.4 | `"NEG"` | Reportes y gráficos |
| `linea_base` | 0.4 | 0,5556 (siempre `AFI`) | 12.2 y 12.8 |
| `cv` | 0.4 | `StratifiedKFold(10, shuffle=True, random_state=SEMILLA)` | **Todas** las evaluaciones y búsquedas |
| `COLOR_CLASE` | 0.5 | `{"AFI": azul, "NEG": rojo}` | Gráficos |

### Por qué `cv` es la pieza clave

`cv` no guarda particiones: guarda **la receta** para hacerlas. Como la semilla es fija, cada vez
que alguien llama a `cv.split(X, y)` salen exactamente los mismos 10 pliegues. Consecuencia: la
búsqueda de un modelo, su evaluación y la evaluación de cualquier otro modelo se hacen sobre
**las mismas particiones**. Las diferencias entre modelos son atribuibles a los modelos, no a que a
uno le tocó una partición más fácil.

---

## 4. Capa 2: la maquinaria (punto 7)

Se define una vez y es idéntica para todos los modelos.

### 4.1 Los objetos

| Objeto | Celda | Qué es |
|---|---|---|
| `ESCALADORES` | 7.1 | `{"Sin escalar": None, "Min-Max": MinMaxScaler(), "Z-score": StandardScaler()}` |
| `METRICAS_CV` | 7.1 | `{"accuracy", "f1_macro", "roc_auc"}`: lo que se mide pliegue por pliegue |
| `CATALOGO_MODELOS` | 7.2 | `{}` al inicio. Una entrada por modelo registrado |
| `RESULTADOS` | 7.2 | `[]` al inicio. Una fila (diccionario) por cada par modelo + escalado |
| `PREDICCIONES` | 7.2 | `{}` al inicio. Clave `(modelo, escalado)`, valor: vectores de predicciones |

### 4.2 Las funciones

| Función | Celda | Entra | Sale |
|---|---|---|---|
| `registrar_modelo(nombre, estimador, escalados, color, nota, grafico_extra)` | 7.2 | La configuración | Escribe `CATALOGO_MODELOS[nombre]` |
| `construir_pipeline(nombre_modelo, nombre_escalado)` | 7.2 | Dos nombres | Un `Pipeline` nuevo, sin entrenar |
| `calcular_metricas(y_real, y_pred, y_proba)` | 7.3 | Etiquetas y probabilidades | Diccionario de 15 métricas |
| `diagnosticar_ajuste(score_entrena, score_valida, desvio)` | 7.3 | Tres números | `(brecha, lista de avisos)` |
| `evaluar_modelo(nombre, mostrar_graficos=True)` | 7.4 | El nombre de un modelo del catálogo | Llena `RESULTADOS` y `PREDICCIONES`; devuelve la tabla del modelo |

### 4.3 La forma de una entrada del catálogo

```python
CATALOGO_MODELOS["XGBoost"] = {
    "estimador":     XGBClassifier(...),   # SIN entrenar: es una plantilla
    "escalados":     ["Sin escalar", "Min-Max", "Z-score"],
    "color":         "#17becf",
    "nota":          "Boosting: árboles chicos en secuencia, ...",
    "grafico_extra": grafico_xgboost,      # la FUNCIÓN, no su resultado (o None)
}
```

Dos detalles:

- **El estimador nunca se entrena.** `construir_pipeline()` hace `clone()`, que copia los
  parámetros a un objeto nuevo y vacío. La plantilla del catálogo queda intacta.
- **`grafico_extra` guarda la función sin llamarla.** `evaluar_modelo` la llama al final, cuando ya
  tiene un modelo entrenado para pasarle. Toda función de gráfico extra recibe siempre los mismos dos
  argumentos: `(pipeline_ajustado, color)`.

### 4.4 La forma de una fila de `RESULTADOS`

```python
{
    "modelo": "XGBoost", "escalado": "Z-score", "color": "#17becf",

    # 15 métricas de calcular_metricas(), sobre las predicciones fuera de pliegue
    "accuracy", "balanced_accuracy",
    "precision", "recall", "f1",
    "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
    "roc_auc", "average_precision",
    "mcc", "kappa",
    "log_loss", "brier",

    # 15 de cross_validate: 3 métricas × 5 estadísticas
    "cv_{accuracy|f1_macro|roc_auc}_{entrena|media|desvio|min|max}",

    # 2 del diagnóstico
    "brecha_ajuste",      # f1 entrenamiento − f1 validación
    "diagnostico",        # "AJUSTE CORRECTO", "SOBREAJUSTE", "Sobreajuste leve", ...
}
```

### 4.5 La forma de una entrada de `PREDICCIONES`

```python
PREDICCIONES[("XGBoost", "Z-score")] = {
    "y_pred":      array de 171 clases (0/1),
    "y_proba":     array de 171 probabilidades de la clase 1,
    "por_pliegue": el diccionario completo que devolvió cross_validate,
    "avisos":      lista de textos de diagnosticar_ajuste(),
}
```

**Por qué hay dos estructuras.** `RESULTADOS` guarda números sueltos y se convierte en tabla.
`PREDICCIONES` guarda vectores de 171 valores, que el punto 12 necesita para dibujar curvas ROC y
cajas por pliegue. Las une la clave `(modelo, escalado)`.

---

## 5. Capa 3: la secuencia de un modelo, paso a paso

Todo modelo pasa por **tres pasos**. Solo el primero es propio de cada modelo; los otros dos son
siempre iguales.

```
PASO 1 · BÚSQUEDA (opcional, propia del modelo)
   └─► produce los hiperparámetros elegidos  +  variables para graficar la búsqueda

PASO 2 · REGISTRO
   registrar_modelo(nombre, Estimador(**fijos, **elegidos), color, nota, grafico_extra)
   └─► escribe CATALOGO_MODELOS[nombre]

PASO 3 · EVALUACIÓN
   tabla_xxx = evaluar_modelo(nombre)
   └─► llena RESULTADOS y PREDICCIONES, imprime y grafica
```

### 5.1 Paso 1: los cuatro patrones de búsqueda que usa el cuaderno

| Patrón | Modelos | Qué queda para el paso 2 |
|---|---|---|
| **Sin búsqueda**: parámetros fijos escritos a mano | Regresión Logística, Random Forest | Nada |
| **Bucle manual** con `cross_validate` sobre un `Pipeline` | KNN | `mejor_k` |
| **`GridSearchCV` sobre el modelo solo** | Árbol, XGBoost | `busqueda.best_params_`, `busqueda_xgb.best_params_` |
| **`GridSearchCV` sobre un `Pipeline`** (escalador + modelo) | SVM | `best_params_` con prefijo `modelo__` → hay que limpiarlo |

Regla para elegir: si el modelo **no necesita escalado** (árboles), se busca sobre el modelo solo.
Si **lo necesita** (KNN, SVM), se busca dentro de un `Pipeline` con `StandardScaler`, para que el
escalador se ajuste dentro de cada pliegue y no haya fuga.

Toda búsqueda usa `cv=cv`: los mismos pliegues que la evaluación. Eso es la **forma 4 de fuga** del
punto 4.3 de la Parte 2A, y está declarada en la limitación (b) del 12.8.

### 5.2 Paso 2: el registro

Solo escribe en el catálogo. No entrena nada, no mide nada.

### 5.3 Paso 3: qué hace `evaluar_modelo(nombre)` por dentro

```
configuracion = CATALOGO_MODELOS[nombre]

PARA CADA nombre_escalado EN configuracion["escalados"]:          ← normalmente 3 vueltas

    pipeline = construir_pipeline(nombre, nombre_escalado)
        └─► Pipeline([("escalador", clone(ESCALADORES[...])),     ← se omite si es "Sin escalar"
                      ("modelo",    clone(configuracion["estimador"]))])

    y_pred  = cross_val_predict(pipeline, X, y, cv=cv, method="predict")        10 ajustes
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")  10 ajustes
        └─► cada uno de los 171 casos lo predice el modelo del pliegue que NO lo vio

    metricas = calcular_metricas(y, y_pred, y_proba)                ← 15 métricas

    detalle = cross_validate(pipeline, X, y, cv=cv,
                             scoring=METRICAS_CV, return_train_score=True)      10 ajustes
        └─► agrega a metricas las 15 claves cv_*

    brecha, avisos = diagnosticar_ajuste(cv_f1_macro_entrena,
                                         cv_f1_macro_media,
                                         cv_f1_macro_desvio)

    RESULTADOS.append(fila)                                         ← fila de la sección 4.4
    PREDICCIONES[(nombre, nombre_escalado)] = {...}                 ← entrada de la sección 4.5

FIN DEL BUCLE

tabla_modelo = DataFrame con las 3 filas de este modelo
    └─► imprime "EFECTO DEL ESCALADO"
        si las 3 dan el mismo ROC-AUC (diferencia < 1e-9) → "el escalado no cambia nada"

mejor = fila de mayor roc_auc  →  escalado_elegido
    └─► lee PREDICCIONES[(nombre, escalado_elegido)] y muestra:
        · métricas detalladas en 6 bloques
        · diagnóstico de ajuste pliegue por pliegue
        · classification_report
        · 6 gráficos: matriz de confusión (casos y %), ROC, precisión-recall,
          distribución de probabilidades, entrenamiento vs. validación por pliegue

SI configuracion["grafico_extra"] no es None:
    pipeline_final = construir_pipeline(nombre, escalado_elegido)
    pipeline_final.fit(X, y)                                         1 ajuste
    configuracion["grafico_extra"](pipeline_final, configuracion["color"])

return tabla_modelo
```

**Importante sobre el último ajuste.** `pipeline_final` se entrena con las 171 filas **solo para
mirar el modelo** (importancias, coeficientes, el árbol dibujado). Ninguna métrica sale de ahí: todas
se calcularon antes, fuera de pliegue.

**Cuenta de ajustes por modelo:** 3 escalados × 30 ajustes + 1 final = **91 ajustes**, más los de su
búsqueda.

---

## 6. Ejemplo completo: XGBoost

### Paso 1: la búsqueda (primera celda de código del 11b)

```python
PARAMS_XGB_FIJOS = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=SEMILLA, n_jobs=1)

grilla_xgb = {"max_depth": [1, 2, 3, 4],
              "n_estimators": [25, 50, 100, 150, 200, 300, 400, 600]}

busqueda_xgb = GridSearchCV(XGBClassifier(**PARAMS_XGB_FIJOS), grilla_xgb,
                            scoring="f1_macro", cv=cv, return_train_score=True)
busqueda_xgb.fit(X, y)       # 32 combinaciones × 10 pliegues = 320 ajustes (+1 reajuste)
```

Variables que deja:

| Variable | Qué es | Se usa en |
|---|---|---|
| `PARAMS_XGB_FIJOS` | Lo que no se busca | El registro (paso 2) |
| `grilla_xgb` | Lo que se busca | Solo esta celda |
| `busqueda_xgb` | El objeto de búsqueda ya ajustado | `.best_params_` en el registro |
| `resultados_xgb` | `cv_results_` como DataFrame, con columna `brecha` | Las curvas de esta celda |
| `mejor_n`, `mejor_prof`, `profundidades` | Auxiliares de los gráficos | Solo esta celda |

### Paso 2: el registro (segunda celda de código del 11b)

```python
registrar_modelo(
    "XGBoost",
    XGBClassifier(**PARAMS_XGB_FIJOS, **busqueda_xgb.best_params_),   # fijos + elegidos
    color="#17becf",
    nota="Boosting: árboles chicos en secuencia, ...",
    grafico_extra=grafico_xgboost,
)
```

`**PARAMS_XGB_FIJOS, **busqueda_xgb.best_params_` desempaqueta los dos diccionarios como argumentos
con nombre. Es lo mismo que escribir
`XGBClassifier(learning_rate=0.05, ..., max_depth=1, n_estimators=400)`.

### Paso 3: la evaluación

```python
tabla_xgb = evaluar_modelo("XGBoost")
```

Lo que pasa adentro es la sección 5.3 con `nombre = "XGBoost"`. Lo particular de este modelo:

- Las tres filas de escalado dan **idénticas**: es un modelo de árboles y el escalado no mueve los
  cortes. El mensaje "el escalado NO cambia absolutamente nada" es la verificación de que la
  maquinaria funciona.
- `grafico_xgboost(pipeline_final, "#17becf")` usa el helper `importancias_xgb()` para pedirle al
  booster la importancia por ganancia y por frecuencia. Solo depende de `predictoras`, no de ninguna
  variable de la búsqueda.

---

## 7. Segundo ejemplo: la SVM y en qué se diferencia

La secuencia es la misma. Cambian tres cosas.

### 7.1 La búsqueda es sobre un `Pipeline`

```python
grilla_svm = {"modelo__C":     [0.1, 0.3, 1, 3, 10, 30, 100],
              "modelo__gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 1.0]}

busqueda_svm = GridSearchCV(
    Pipeline([("escalador", StandardScaler()),
              ("modelo",    SVC(kernel="rbf", random_state=SEMILLA))]),
    grilla_svm, scoring="f1_macro", cv=cv, return_train_score=True)
busqueda_svm.fit(X, y)       # 42 combinaciones × 10 = 420 ajustes
```

El prefijo `modelo__` le dice a `GridSearchCV` a qué paso del pipeline va cada parámetro: el
nombre del paso, dos guiones bajos y el nombre del parámetro. Antes de registrar hay que quitarlo,
porque en el catálogo va el `SVC` solo:

```python
mejores_params_svm = {clave.replace("modelo__", ""): valor
                      for clave, valor in busqueda_svm.best_params_.items()}
# {"modelo__C": 30, "modelo__gamma": 0.03}  →  {"C": 30, "gamma": 0.03}
```

### 7.2 `probability` cambia entre la búsqueda y el catálogo

| Dónde | `probability` | Por qué |
|---|---|---|
| Búsqueda | `False` (por defecto) | El puntaje es F1, que usa `predict()`. Así cada ajuste es más rápido |
| Catálogo | `True` | `evaluar_modelo` llama a `predict_proba()`. Sin esto, falla |

Con `probability=True`, cada ajuste entrena internamente 5 SVM más para calibrar la sigmoide de
Platt.

### 7.3 Las tres filas de escalado dan distinto

A diferencia de XGBoost, el escalado sí importa. Y hay una sutileza: `gamma` se eligió con Z-score,
así que en "Sin escalar" ese mismo número significa otra cosa (distancias más chicas, un `gamma`
efectivo más bajo). La diferencia se ve en la tabla "EFECTO DEL ESCALADO" y después en el 12.4.

### 7.4 El gráfico extra depende de la búsqueda

`grafico_svm` lee **variables globales de la celda de búsqueda**: `mapa_valida`, `orden_gamma` y
`busqueda_svm`, para dibujar la grilla con el recuadro del valor elegido. Funciona porque esa celda
siempre corre antes. Ver la sección 11.

---

## 8. Ficha de los seis modelos

| | Regresión Logística | KNN | Árbol | Random Forest | XGBoost | SVM |
|---|---|---|---|---|---|---|
| **Punto** | 8 | 9 | 10 | 11 | 11b | 11c |
| **Nombre en el catálogo** | `"Regresión Logística"` | `f"KNN (k={mejor_k})"` → `nombre_knn` | `"Árbol de Decisión"` | `"Random Forest"` | `"XGBoost"` | `"SVM (RBF)"` → `nombre_svm` |
| **Patrón de búsqueda** | Ninguna | Bucle manual | `GridSearchCV` | Ninguna | `GridSearchCV` | `GridSearchCV` sobre `Pipeline` |
| **Qué se busca** | — | `k` (1 a 31, impares) | `max_depth`, `min_samples_leaf`, `criterion` | — | `max_depth`, `n_estimators` | `C`, `gamma` |
| **Variables de la búsqueda** | — | `valores_k`, `resultados_k`, `mejor_k` | `grilla_arbol`, `busqueda`, `resultados_grilla` | — | `PARAMS_XGB_FIJOS`, `grilla_xgb`, `busqueda_xgb`, `resultados_xgb` | `grilla_svm`, `busqueda_svm`, `mejores_params_svm`, `resultados_svm`, `orden_gamma`, `mapa_valida`, `mapa_brecha` |
| **Estimador registrado** | `LogisticRegression(max_iter=5000, random_state=SEMILLA)` | `KNeighborsClassifier(n_neighbors=mejor_k)` | `DecisionTreeClassifier(random_state=SEMILLA, **busqueda.best_params_)` | `RandomForestClassifier(n_estimators=400, random_state=SEMILLA, n_jobs=1)` | `XGBClassifier(**PARAMS_XGB_FIJOS, **busqueda_xgb.best_params_)` | `SVC(kernel="rbf", probability=True, random_state=SEMILLA, **mejores_params_svm)` |
| **Color** | `#1f77b4` | `#ff7f0e` | `#2ca02c` | `#9467bd` | `#17becf` | `#e377c2` |
| **Gráfico extra** | `grafico_coeficientes` | `grafico_curva_k` | `grafico_arbol` | `grafico_bosque` | `grafico_xgboost` | `grafico_svm` |
| **El gráfico extra lee globales de la búsqueda** | No | **Sí**: `resultados_k`, `valores_k` | No | No | No | **Sí**: `mapa_valida`, `orden_gamma`, `busqueda_svm` |
| **¿Le importa el escalado?** | Sí | Sí | No | No | No | Sí |
| **Tabla devuelta** | `tabla_logistica` | `tabla_knn` | `tabla_arbol` | `tabla_bosque` | `tabla_xgb` | `tabla_svm` |
| **Hiperparámetros elegidos sobre los mismos datos** | No | Sí | Sí | No | Sí | Sí |

Ojo con el árbol: su búsqueda se llama `busqueda`, a secas. Si algún día agregás otra búsqueda,
no la llames igual, porque pisarías la del árbol.

---

## 9. Capa 4: la comparativa (punto 12)

El punto 12 **solo lee**. Ninguna celda nombra un modelo: todo sale de `RESULTADOS` y
`PREDICCIONES`.

```
RESULTADOS
   └─► maestra = DataFrame(RESULTADOS)                         12.1   las 18 filas
          ├─► mejores_por_modelo                                12.2   1 fila por modelo (la de mayor AUC)
          │      ├─► + PREDICCIONES[(m, e)]["y_proba"]          12.3   curvas ROC y precisión-recall
          │      ├─► estabilidad                                12.5   + PREDICCIONES[...]["por_pliegue"]
          │      ├─► ajuste                                     12.6   + PREDICCIONES[...]["avisos"]
          │      └─► ranking  (con PESOS_RANKING)               12.7
          └─► pivote_auc, pivote_f1 → sensibilidad              12.4

12.8 (veredicto) lee: ranking, linea_base, sensibilidad (del 12.4), estabilidad (del 12.5)
Cierre lee: maestra, ranking
```

| Variable | Celda | Qué es |
|---|---|---|
| `maestra` | 12.1 | Todas las filas de `RESULTADOS` como DataFrame |
| `mejores_por_modelo` | 12.2 | Una fila por modelo: su mejor escalado por ROC-AUC |
| `pivote_auc`, `pivote_f1` | 12.4 | Modelos × escalados |
| `sensibilidad` | 12.4 | Cuánto cambia el AUC entre el mejor y el peor escalado de cada modelo |
| `estabilidad` | 12.5 | Media, desvío, mínimo y máximo de F1 entre pliegues |
| `ajuste` | 12.6 | Brecha entrenamiento − validación |
| `PESOS_RANKING` | 12.7 | 35 % AUC, 30 % F1 macro, 20 % MCC, 15 % estabilidad |
| `ranking` | 12.7 | `mejores_por_modelo` + `puntaje` + `puesto` |

**Cómo llega el color a todos los gráficos:** viaja dentro de cada fila de `RESULTADOS` (columna
`color`), así que cualquier tabla derivada lo trae consigo.

**La única referencia a un modelo por nombre** está en el 12.8:
`ranking["modelo"].str.contains("Logística")`, para comparar al ganador contra el modelo más simple.

---

## 10. Receta: agregar un modelo nuevo

### Lo mínimo: dos líneas

```python
from sklearn.naive_bayes import GaussianNB

registrar_modelo("Naive Bayes", GaussianNB(), color="#8c564b",
                 nota="Supone independencia entre variables; muy rápido.")
tabla_nb = evaluar_modelo("Naive Bayes")
```

### La versión completa, con búsqueda y gráfico propio: tres celdas de código

```python
# ---- CELDA A: búsqueda ----
from sklearn.algo import MiModelo

PARAMS_MIO_FIJOS = dict(random_state=SEMILLA)
grilla_mio = {"param1": [...], "param2": [...]}
# Si el modelo necesita escalado, buscar sobre Pipeline y prefijar con "modelo__"

busqueda_mio = GridSearchCV(MiModelo(**PARAMS_MIO_FIJOS), grilla_mio,
                            scoring="f1_macro", cv=cv, n_jobs=1, return_train_score=True)
busqueda_mio.fit(X, y)
resultados_mio = pd.DataFrame(busqueda_mio.cv_results_)
# ... gráficos de la búsqueda ...


# ---- CELDA B: gráfico propio ----
def grafico_mio(pipeline_ajustado, color):          # siempre esta firma
    modelo = pipeline_ajustado.named_steps["modelo"]  # el estimador ya entrenado
    # ... dibujar algo propio del modelo ...


# ---- CELDA C: registro y evaluación ----
registrar_modelo("Mi Modelo",
                 MiModelo(**PARAMS_MIO_FIJOS, **busqueda_mio.best_params_),
                 color="#bcbd22",
                 nota="Una línea que recuerde qué es.",
                 grafico_extra=grafico_mio)
tabla_mio = evaluar_modelo("Mi Modelo")
```

### Lista de control

- [ ] El estimador tiene `predict_proba()`. Si es una SVM, `probability=True`.
- [ ] La etiqueta que espera es 0/1. `y` ya lo es.
- [ ] Le pasé `random_state=SEMILLA` si tiene azar.
- [ ] La búsqueda usa `cv=cv`, no un número de pliegues propio.
- [ ] Si busqué sobre un `Pipeline`, saqué el prefijo `modelo__` antes de registrar.
- [ ] El color no repite ninguno de la ficha (sección 8) ni el rojo de `NEG`.
- [ ] El nombre de la búsqueda no pisa una existente (`busqueda` es del árbol).
- [ ] Si no necesita escalado, puedo pasar `escalados=["Sin escalar"]` para ahorrar tiempo, o dejar
      los tres como verificación.
- [ ] Si busqué hiperparámetros, lo agregué a la limitación (b) del 12.8, que es texto fijo.
- [ ] Agregué el modelo al índice y a la tabla del cierre.
- [ ] Volví a correr el punto 12 completo, en orden.

---

## 11. Trampas conocidas

### 11.1 `RESULTADOS` solo crece

`registrar_modelo` **pisa** la entrada del catálogo, pero `evaluar_modelo` **agrega** filas. Si se
vuelve a correr la celda de un modelo, quedan filas duplicadas:

| Qué se ve afectado | Efecto |
|---|---|
| Ranking, 12.2 a 12.8 | Nada: se quedan con una fila por modelo |
| Tabla maestra (12.1) | Filas repetidas |
| "Combinaciones totales" y "Ajustes de modelo" del cierre | Inflados |

Solución: una línea al principio de `evaluar_modelo`, antes del bucle:

```python
RESULTADOS[:] = [f for f in RESULTADOS if f["modelo"] != nombre]
```

(`RESULTADOS[:] = ...` reemplaza el contenido **de la misma lista**. Con `RESULTADOS = ...` se
crearía una variable local nueva dentro de la función y la global no cambiaría.)

### 11.2 Gráficos extra que dependen de la celda de búsqueda

`grafico_curva_k` (KNN) y `grafico_svm` (SVM) leen variables globales creadas en su celda de
búsqueda. Si reiniciás el entorno y corrés solo la celda de registro, fallan con `NameError`.
Siempre hay que correr la búsqueda antes.

### 11.3 El punto 12 tiene orden interno

El 12.8 usa `sensibilidad` (creada en el 12.4) y `estabilidad` (creada en el 12.5). Hay que correr
el punto 12 entero y de arriba a abajo.

### 11.4 Textos fijos que no se actualizan solos

Casi todo el cuaderno se adapta solo a la cantidad de modelos, salvo:

- La limitación (b) del 12.8: lista a mano qué modelos tuvieron búsqueda.
- El índice, la introducción y la tabla del cierre.
- El 12.4 (markdown): las expectativas de qué modelos deberían ser sensibles al escalado.

### 11.5 El escalado "ganador" se elige por ROC-AUC

`evaluar_modelo` y el 12.2 eligen la mejor variante de cada modelo por ROC-AUC, no por F1. Puede
pasar que la variante elegida no sea la de mayor F1, como en la SVM, donde el AUC puede favorecer a
Min-Max y el F1 a Z-score. Es una decisión de diseño: el AUC no depende del umbral de 0,5.
