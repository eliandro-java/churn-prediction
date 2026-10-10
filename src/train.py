
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # 1. Carregar os dados
    BASE_DIR = Path(__file__).resolve().parent.parent  # raiz do projeto (churn-prediction)
    df = pd.read_csv(BASE_DIR / "data" / "clientes.csv")

    # 2. Separar características (X) e alvo (y)
    X = df[
        ["idade", "tempo_cliente", "valor_mensal", "reclamacoes"]
    ]
    y = df["cancelou"]

    # 3. Separar treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # 4. Criar o pipeline
    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classificador", DecisionTreeClassifier(
            max_depth=3,
            random_state=42,
        )),
    ])

    # 5. Treinar
    modelo.fit(X_treino, y_treino)

    # 6. Avaliar
    previsoes = modelo.predict(X_teste)

    print(f"Acurácia: {accuracy_score(y_teste, previsoes):.2f}")
    print("\nRelatório de classificação:")
    print(classification_report(
        y_teste,
        previsoes,
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
