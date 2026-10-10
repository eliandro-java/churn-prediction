
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = BASE_DIR / "models" / "modelo_churn.joblib"


def test_modelo_produz_previsao_valida():
    modelo = joblib.load(CAMINHO_MODELO)

    cliente = pd.DataFrame([{
        "idade": 35,
        "tempo_cliente": 24,
        "valor_mensal": 149.90,
        "reclamacoes": 2,
    }])

    previsao = modelo.predict(cliente)

    assert len(previsao) == 1
    assert previsao[0] in [0, 1]


def test_modelo_produz_probabilidades_validas():
    modelo = joblib.load(CAMINHO_MODELO)

    cliente = pd.DataFrame([{
        "idade": 35,
        "tempo_cliente": 24,
        "valor_mensal": 149.90,
        "reclamacoes": 2,
    }])

    probabilidades = modelo.predict_proba(cliente)

    assert probabilidades.shape == (1, 2)
    assert abs(probabilidades.sum() - 1.0) < 1e-9
