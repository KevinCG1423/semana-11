from __future__ import annotations

import gc
import json
import sqlite3
import time
import tracemalloc
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def medir_etapa(mediciones: dict, nombre: str, funcion):
    gc.collect()
    tracemalloc.start()
    inicio = time.perf_counter()
    try:
        resultado = funcion()
    finally:
        segundos = time.perf_counter() - inicio
        _, pico_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mediciones[nombre] = {
            "tiempo_segundos": round(segundos, 6),
            "pico_memoria_python_bytes": int(pico_bytes),
            "pico_memoria_python_mib": round(pico_bytes / (1024**2), 4),
        }
    return resultado


def validar_configuracion(config: dict) -> None:
    claves = {
        "archivo_entrada",
        "archivo_reporte",
        "columnas_requeridas",
        "columna_objetivo",
        "columnas_modelo",
        "planes_validos",
        "rangos",
        "test_size",
        "random_state",
        "umbral_accuracy",
    }
    faltantes = sorted(claves - config.keys())
    if faltantes:
        raise ValueError(f"Faltan claves en config_pipeline.json: {faltantes}")

    requeridas = config["columnas_requeridas"]
    modelos = config["columnas_modelo"]
    objetivo = config["columna_objetivo"]
    if not isinstance(requeridas, list) or not requeridas:
        raise ValueError("columnas_requeridas debe ser una lista no vacía.")
    if len(requeridas) != len(set(requeridas)):
        raise ValueError("columnas_requeridas contiene columnas repetidas.")
    if "id_cliente" not in requeridas or objetivo not in requeridas:
        raise ValueError("Se requieren id_cliente y la columna objetivo.")
    if not isinstance(modelos, list) or not modelos or not set(modelos).issubset(requeridas):
        raise ValueError("columnas_modelo debe ser una lista de columnas requeridas.")
    if objetivo in modelos:
        raise ValueError("La columna objetivo no puede usarse como variable del modelo.")
    if not config["planes_validos"]:
        raise ValueError("planes_validos no puede estar vacío.")
    if not set(modelos).issubset(config["rangos"]):
        raise ValueError("Cada columna del modelo debe tener un rango configurado.")
    for columna, limites in config["rangos"].items():
        if (
            not isinstance(limites, list)
            or len(limites) != 2
            or limites[0] > limites[1]
        ):
            raise ValueError(f"Rango inválido para {columna}: {limites}")
    if not 0 < float(config["test_size"]) < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")
    if not 0 <= float(config["umbral_accuracy"]) <= 1:
        raise ValueError("umbral_accuracy debe estar entre 0 y 1.")


def preparar_y_validar(datos: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    requeridas = config["columnas_requeridas"]
    faltantes = sorted(set(requeridas) - set(datos.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    trabajo = datos.loc[:, requeridas].copy()
    trabajo["id_cliente"] = trabajo["id_cliente"].astype("string").str.strip()
    trabajo["id_cliente"] = trabajo["id_cliente"].replace("", pd.NA)
    trabajo["plan"] = trabajo["plan"].astype("string").str.strip()

    columnas_numericas = list(dict.fromkeys(config["columnas_modelo"] + [config["columna_objetivo"]]))
    conversiones_invalidas = {}
    for columna in columnas_numericas:
        original = trabajo[columna]
        convertido = pd.to_numeric(original, errors="coerce")
        conversiones_invalidas[columna] = int((original.notna() & convertido.isna()).sum())
        trabajo[columna] = convertido

    nulos = {columna: int(cantidad) for columna, cantidad in trabajo.isna().sum().items()}
    rangos_invalidos = {}
    for columna, (minimo, maximo) in config["rangos"].items():
        if columna not in trabajo:
            raise ValueError(f"El rango configurado refiere a una columna ausente: {columna}")
        valores = trabajo[columna]
        fuera = valores.notna() & ~valores.between(minimo, maximo)
        rangos_invalidos[columna] = int(fuera.sum())

    planes_invalidos = trabajo["plan"].notna() & ~trabajo["plan"].isin(config["planes_validos"])
    objetivo = trabajo[config["columna_objetivo"]]
    objetivo_invalido = objetivo.notna() & ~objetivo.isin([0, 1])
    auditoria = {
        "filas": int(len(trabajo)),
        "columnas_encontradas": list(datos.columns),
        "columnas_faltantes": [],
        "columnas_adicionales": sorted(set(datos.columns) - set(requeridas)),
        "nulos_por_columna": nulos,
        "duplicados_exactos": int(trabajo.duplicated().sum()),
        "ids_duplicados_excedentes": int(trabajo["id_cliente"].duplicated().sum()),
        "planes_invalidos": int(planes_invalidos.sum()),
        "valores_no_numericos_por_columna": conversiones_invalidas,
        "valores_fuera_de_rango_por_columna": rangos_invalidos,
        "objetivos_invalidos": int(objetivo_invalido.sum()),
    }
    return trabajo, auditoria


def limpiar_datos(datos: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    filas_entrada = len(datos)
    sin_duplicados = datos.drop_duplicates()
    duplicados_exactos = filas_entrada - len(sin_duplicados)
    ids_unicos = sin_duplicados.drop_duplicates(subset=["id_cliente"], keep="first")
    ids_duplicados = len(sin_duplicados) - len(ids_unicos)

    invalidas = ids_unicos.isna().any(axis=1)
    invalidas |= ~ids_unicos["plan"].isin(config["planes_validos"])
    objetivo = ids_unicos[config["columna_objetivo"]]
    invalidas |= ~objetivo.isin([0, 1])
    for columna, (minimo, maximo) in config["rangos"].items():
        invalidas |= ~ids_unicos[columna].between(minimo, maximo)

    filas_invalidas = int(invalidas.sum())
    limpias = ids_unicos.loc[~invalidas].reset_index(drop=True)
    resumen = {
        "filas_entrada": int(filas_entrada),
        "duplicados_exactos_eliminados": int(duplicados_exactos),
        "ids_duplicados_adicionales_eliminados": int(ids_duplicados),
        "filas_con_datos_invalidos_eliminadas": filas_invalidas,
        "filas_salida": int(len(limpias)),
        "filas_eliminadas_total": int(filas_entrada - len(limpias)),
    }
    return limpias, resumen


def tasa_abandono_pandas(datos: pd.DataFrame) -> dict:
    resumen = datos.groupby("plan", sort=True)["abandono"].agg(
        clientes="size", tasa_abandono="mean"
    )
    return {
        str(plan): {
            "clientes": int(fila["clientes"]),
            "tasa_abandono": float(fila["tasa_abandono"]),
        }
        for plan, fila in resumen.iterrows()
    }


def tasa_abandono_sqlite(datos: pd.DataFrame) -> dict:
    conexion = sqlite3.connect(":memory:")
    try:
        conexion.execute(
            "CREATE TABLE clientes (plan TEXT NOT NULL, abandono INTEGER NOT NULL)"
        )
        conexion.executemany(
            "INSERT INTO clientes (plan, abandono) VALUES (?, ?)",
            (
                (str(plan), int(abandono))
                for plan, abandono in datos[["plan", "abandono"]].itertuples(index=False, name=None)
            ),
        )
        filas = conexion.execute(
            "SELECT plan, COUNT(*) AS clientes, AVG(CAST(abandono AS REAL)) AS tasa_abandono "
            "FROM clientes GROUP BY plan ORDER BY plan"
        ).fetchall()
        return {
            plan: {"clientes": int(clientes), "tasa_abandono": float(tasa)}
            for plan, clientes, tasa in filas
        }
    finally:
        conexion.close()


def seleccionar_umbral_f1(
    modelo, entrenamiento_x, entrenamiento_y, test_size: float, random_state: int
) -> tuple[float, int, int]:
    ajuste_x, validacion_x, ajuste_y, validacion_y = train_test_split(
        entrenamiento_x,
        entrenamiento_y,
        test_size=test_size,
        random_state=random_state,
        stratify=entrenamiento_y,
    )
    modelo.fit(ajuste_x, ajuste_y)
    probabilidades = modelo.predict_proba(validacion_x)[:, 1]
    precision, recall, umbrales = precision_recall_curve(validacion_y, probabilidades)
    if len(umbrales) == 0:
        raise ValueError("No se pudo seleccionar un umbral con los datos de validación.")

    puntajes_f1 = [
        2 * precision[indice] * recall[indice]
        / (precision[indice] + recall[indice])
        if precision[indice] + recall[indice] > 0
        else 0.0
        for indice in range(len(umbrales))
    ]
    mejor_indice = max(
        range(len(umbrales)),
        key=lambda indice: (puntajes_f1[indice], umbrales[indice]),
    )
    return float(umbrales[mejor_indice]), len(ajuste_y), len(validacion_y)


def evaluar_modelo(
    modelo, entrenamiento_x, prueba_x, entrenamiento_y, prueba_y, umbral: float | None = None
) -> dict:
    modelo.fit(entrenamiento_x, entrenamiento_y)
    if umbral is None:
        predicciones = modelo.predict(prueba_x)
    else:
        predicciones = modelo.predict_proba(prueba_x)[:, 1] >= umbral
    return {
        "accuracy": float(accuracy_score(prueba_y, predicciones)),
        "precision": float(precision_score(prueba_y, predicciones, zero_division=0)),
        "recall": float(recall_score(prueba_y, predicciones, zero_division=0)),
        "f1": float(f1_score(prueba_y, predicciones, zero_division=0)),
    }


def guardar_reporte(ruta: Path, reporte: dict, mediciones: dict) -> None:
    inicio = time.perf_counter()
    tracemalloc.start()
    ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    segundos = time.perf_counter() - inicio
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mediciones["guardar_reporte"] = {
        "tiempo_segundos": round(segundos, 6),
        "pico_memoria_python_bytes": int(pico_bytes),
        "pico_memoria_python_mib": round(pico_bytes / (1024**2), 4),
    }
    reporte["mediciones_por_etapa"] = mediciones
    ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    carpeta = Path(__file__).resolve().parent
    mediciones = {}
    config = medir_etapa(
        mediciones,
        "cargar_y_validar_configuracion",
        lambda: json.loads((carpeta / "config_pipeline.json").read_text(encoding="utf-8")),
    )
    validar_configuracion(config)

    ruta_csv = carpeta / config["archivo_entrada"]
    datos_originales = medir_etapa(
        mediciones,
        "cargar_csv",
        lambda: pd.read_csv(ruta_csv),
    )
    datos_preparados, validacion_antes = medir_etapa(
        mediciones,
        "validar_antes",
        lambda: preparar_y_validar(datos_originales, config),
    )
    datos_limpios, limpieza = medir_etapa(
        mediciones,
        "limpiar_datos",
        lambda: limpiar_datos(datos_preparados, config),
    )
    _, validacion_despues = medir_etapa(
        mediciones,
        "validar_despues",
        lambda: preparar_y_validar(datos_limpios, config),
    )

    validacion_despues["valido"] = (
        not any(validacion_despues["nulos_por_columna"].values())
        and validacion_despues["duplicados_exactos"] == 0
        and validacion_despues["ids_duplicados_excedentes"] == 0
        and validacion_despues["planes_invalidos"] == 0
        and validacion_despues["objetivos_invalidos"] == 0
        and not any(validacion_despues["valores_no_numericos_por_columna"].values())
        and not any(validacion_despues["valores_fuera_de_rango_por_columna"].values())
    )
    if not validacion_despues["valido"]:
        raise ValueError("La validación posterior a la limpieza encontró errores.")
    if datos_limpios.empty or datos_limpios[config["columna_objetivo"]].nunique() < 2:
        raise ValueError("Se necesitan datos limpios de ambas clases para entrenar los modelos.")

    churn_pandas = medir_etapa(
        mediciones,
        "tasa_abandono_pandas",
        lambda: tasa_abandono_pandas(datos_limpios),
    )
    churn_sqlite = medir_etapa(
        mediciones,
        "tasa_abandono_sqlite3",
        lambda: tasa_abandono_sqlite(datos_limpios),
    )
    resultados_coinciden = churn_pandas.keys() == churn_sqlite.keys() and all(
        churn_pandas[plan]["clientes"] == churn_sqlite[plan]["clientes"]
        and abs(churn_pandas[plan]["tasa_abandono"] - churn_sqlite[plan]["tasa_abandono"]) < 1e-12
        for plan in churn_pandas
    )
    if not resultados_coinciden:
        raise AssertionError("Pandas y SQLite produjeron tasas de abandono distintas.")

    columnas_modelo = config["columnas_modelo"]
    objetivo = config["columna_objetivo"]
    entrenamiento_x, prueba_x, entrenamiento_y, prueba_y = medir_etapa(
        mediciones,
        "particion_entrenamiento_prueba",
        lambda: train_test_split(
            datos_limpios[columnas_modelo],
            datos_limpios[objetivo],
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=datos_limpios[objetivo],
        ),
    )
    dummy = DummyClassifier(strategy="most_frequent")
    resultado_dummy = medir_etapa(
        mediciones,
        "entrenar_dummy_classifier",
        lambda: evaluar_modelo(dummy, entrenamiento_x, prueba_x, entrenamiento_y, prueba_y),
    )
    pipeline = Pipeline(
        [
            ("escalador", StandardScaler()),
            ("regresion_logistica", LogisticRegression(max_iter=1000, random_state=config["random_state"])),
        ]
    )
    umbral_prediccion, filas_ajuste_umbral, filas_validacion_umbral = medir_etapa(
        mediciones,
        "seleccionar_umbral_f1",
        lambda: seleccionar_umbral_f1(
            Pipeline(
                [
                    ("escalador", StandardScaler()),
                    (
                        "regresion_logistica",
                        LogisticRegression(max_iter=1000, random_state=config["random_state"]),
                    ),
                ]
            ),
            entrenamiento_x,
            entrenamiento_y,
            test_size=0.25,
            random_state=config["random_state"] + 1,
        ),
    )
    resultado_logistico = medir_etapa(
        mediciones,
        "entrenar_pipeline_regresion_logistica",
        lambda: evaluar_modelo(
            pipeline,
            entrenamiento_x,
            prueba_x,
            entrenamiento_y,
            prueba_y,
            umbral=umbral_prediccion,
        ),
    )

    etapa_mas_costosa = max(mediciones, key=lambda etapa: mediciones[etapa]["tiempo_segundos"])
    reporte = {
        "archivo_entrada": config["archivo_entrada"],
        "validacion_antes": validacion_antes,
        "limpieza": limpieza,
        "validacion_despues": validacion_despues,
        "abandono_por_plan": {
            "pandas": churn_pandas,
            "sqlite3": churn_sqlite,
            "resultados_coinciden": resultados_coinciden,
            "comparacion_tiempo_segundos": {
                "pandas": mediciones["tasa_abandono_pandas"]["tiempo_segundos"],
                "sqlite3": mediciones["tasa_abandono_sqlite3"]["tiempo_segundos"],
            },
        },
        "modelos": {
            "particion": {
                "test_size": config["test_size"],
                "random_state": config["random_state"],
                "filas_entrenamiento": int(len(entrenamiento_y)),
                "filas_prueba": int(len(prueba_y)),
            },
            "DummyClassifier": resultado_dummy,
            "Pipeline_RegresionLogistica": resultado_logistico,
            "umbral_accuracy": config["umbral_accuracy"],
            "logistica_supera_umbral_accuracy": resultado_logistico["accuracy"] >= config["umbral_accuracy"],
            "umbral_prediccion_logistica": umbral_prediccion,
            "filas_ajuste_umbral": filas_ajuste_umbral,
            "filas_validacion_umbral": filas_validacion_umbral,
            "criterio_seleccion_umbral": "maximizar F1 en validación, usando solo los datos de entrenamiento",
        },
        "mediciones_por_etapa": mediciones,
        "etapa_mas_costosa_por_tiempo": {
            "nombre": etapa_mas_costosa,
            "tiempo_segundos": mediciones[etapa_mas_costosa]["tiempo_segundos"],
        },
        "nota_memoria": (
            "Pico de asignaciones Python medido con tracemalloc; no incluye toda la memoria nativa "
            "reservada por NumPy o scikit-learn."
        ),
    }
    ruta_reporte = carpeta / config["archivo_reporte"]
    guardar_reporte(ruta_reporte, reporte, mediciones)
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    print(f"\nReporte guardado en: {ruta_reporte}")


if __name__ == "__main__":
    main()