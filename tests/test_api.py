
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_endpoint_inicial():
    resposta = client.get("/")

    assert resposta.status_code == 200
    assert resposta.json()["mensagem"] == "Churn Prediction API funcionando"


def test_previsao_de_cliente():
    cliente = {
        "idade": 35,
        "tempo_cliente": 24,
        "valor_mensal": 149.90,
        "reclamacoes": 2,
    }

    resposta = client.post("/prever", json=cliente)

    assert resposta.status_code == 200

    resultado = resposta.json()

    assert resultado["previsao"] in [0, 1]
    assert resultado["resultado"] in [
        "cancelamento",
        "nao_cancelamento",
    ]

    probabilidade = resultado["probabilidade_cancelamento"]
    assert 0 <= probabilidade <= 1


def test_api_rejeita_idade_invalida():
    cliente = {
        "idade": 10,
        "tempo_cliente": 24,
        "valor_mensal": 149.90,
        "reclamacoes": 2,
    }

    resposta = client.post("/prever", json=cliente)

    assert resposta.status_code == 422
