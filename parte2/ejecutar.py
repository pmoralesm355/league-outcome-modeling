import argparse
import pandas as pd
from carga_datos import cargar_dataset_poker, muestrear_test
from caracteristicas import construir_matriz_caracteristicas
from entrenamiento_modelos import crear_modelos, entrenar_modelos
from evaluacion import evaluar_modelo, resumen_metricas_modelos
from visualizaciones import (
    graficar_distribucion_clases,
    graficar_confusion_apilada,
    graficar_metricas_por_clase,
    graficar_resumen_modelos,
    graficar_importancias_atributos,
)

def main():
    parser = argparse.ArgumentParser(description="Poker Hand: EDA, features, modelos y evaluación")
    parser.add_argument("--test-n", type=int, default=100_000, help="Tamaño de muestra de test")
    parser.add_argument("--no-plots", action="store_true", help="No mostrar gráficas")
    args = parser.parse_args()

    train_df, test_df = cargar_dataset_poker()
    test_sample = muestrear_test(test_df, args.test_n)
    print("Tamaño train:", train_df.shape, " - Tamaño test (muestra):", test_sample.shape)

    #  EDA mínima requerida (variables crudas) 
    print("\nDistribución de clases (train):")
    print(train_df["label"].value_counts(normalize=True).sort_index())

    print("\nEDA rápida de rangos (R1..R5): top-5 proporciones por columna")
    for c in [f"R{i}" for i in range(1, 6)]:
        vc = train_df[c].value_counts(normalize=True).round(3).head(5)
        print(f"{c}: {vc.to_dict()}")

    print("\nEDA rápida de palos (S1..S5): proporciones por columna")
    for c in [f"S{i}" for i in range(1, 6)]:
        vc = train_df[c].value_counts(normalize=True).round(3)
        print(f"{c}: {vc.to_dict()}")

    #  Features 
    X_train, y_train = construir_matriz_caracteristicas(train_df)
    X_test,  y_test  = construir_matriz_caracteristicas(test_sample)
    print("\nAtributos generados:", X_train.shape[1])

    # correlación simple de features derivadas
    try:
        corr = pd.concat([X_train, pd.Series(y_train, name="label")], axis=1)\
                 .corr(numeric_only=True)["label"]\
                 .sort_values(ascending=False).head(15)
        print("\nTop-15 correlaciones (features vs. label):")
        print(corr)
    except Exception as e:
        print("\n[Info] No se pudo calcular correlaciones:", e)

    # Modelos
    rf, hgb, pesos = crear_modelos(y_train)
    rf, hgb = entrenar_modelos(rf, hgb, X_train, y_train, pesos)

    pred_rf, cm_rf, f1_rf, ba_rf = evaluar_modelo(rf, X_test, y_test, "RandomForest")
    pred_hg, cm_hg, f1_hg, ba_hg = evaluar_modelo(hgb, X_test, y_test, "HistGradientBoosting")

    print("\nResumen de métricas (test):")
    resumen = resumen_metricas_modelos([
        ("RandomForest", f1_rf, ba_rf),
        ("HistGradientBoosting", f1_hg, ba_hg),
    ])
    print(resumen)

    if not args.no_plots:
        graficar_distribucion_clases(y_train, "Distribución de clases (train)")
        graficar_distribucion_clases(y_test,  "Distribución de clases (test)")

        graficar_confusion_apilada(cm_rf, "Confusión por clase (100%) - RandomForest")
        graficar_confusion_apilada(cm_hg, "Confusión por clase (100%) - HistGradientBoosting")

        graficar_metricas_por_clase(y_test, pred_rf, "Precision / Recall / F1 por clase - RandomForest")
        graficar_metricas_por_clase(y_test, pred_hg, "Precision / Recall / F1 por clase - HistGradientBoosting")

        graficar_resumen_modelos(
            macro_f1=[f1_rf, f1_hg],
            bal_acc=[ba_rf, ba_hg],
            nombres_modelos=["RandomForest", "HistGB"]
        )

        graficar_importancias_atributos(rf, nombres_features=list(X_train.columns), top_k=15)

if __name__ == "__main__":
    main()
