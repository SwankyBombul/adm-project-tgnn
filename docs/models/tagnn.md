# TAGNN

**TAGNN** (Target Attentive Graph Neural Network) reprezentuje sesję jako **graf skierowany** (kolejne kliknięcia → krawędzie) i dekoduje następny item z **target attention** względem ostatniego węzła. W projekcie to grafowy baseline między GRU4Rec a TGN.

Port z [CRIPAC-DIG/TAGNN](https://github.com/CRIPAC-DIG/TAGNN) (SIGIR 2020, [arxiv:2005.02844](https://arxiv.org/pdf/2005.02844)).

---

## Idea

| Element | W projekcie |
|---------|-------------|
| Graf sesji | Węzły = pozycje w sesji; krawędzie = przejścia klik→klik |
| Encoder | GGNN (`GNN` class) — kilka kroków propagacji |
| Decoder | Target attention + softmax nad katalogiem |
| Loss | Cross-entropy (full catalog), jak GRU4Rec |

Wybór TAGNN zamiast SR-GNN: mechanizm attention na target item daje silniejszy sygnał przy next-item prediction (decyzja zespołu — [`data-and-eda.md`](../data-and-eda.md)).

---

## Uruchomienie

```powershell
uv run python -m src.main fit `
  -c config/data/tagnn_yoochoose.yaml `
  -c config/model/tagnn.yaml `
  -c config/experiments/tagnn_baseline.yaml
```

---

## Konfiguracja

Plik [`config/model/tagnn.yaml`](../config/model/tagnn.yaml):

| Parametr | Opis |
|----------|------|
| `hidden_dim` | Wymiar ukryty węzłów grafu |
| `gnn_steps` | Liczba kroków propagacji GGNN |
| `nonhybrid` | Czy wyłączyć hybrydę z ostatnim kliknięciem |
| `max_seq_len` | Maks. długość sesji (obcięcie) |
| `learning_rate` | SGD/Adam lr |
| `weight_decay` | Regularyzacja |

---

## Moduły

| Plik | Rola |
|------|------|
| [`model.py`](../src/models/tagnn/model.py) | `GNN`, `TAGNN` |
| [`graph_batch.py`](../src/models/tagnn/graph_batch.py) | Batchowanie grafów sesyjnych |
| [`batch_sampler.py`](../src/models/tagnn/batch_sampler.py) | Sampler z `max_seq_len` |
| [`dataset.py`](../src/models/tagnn/dataset.py) | Pickle examples z preprocessingu |
| [`module.py`](../src/models/tagnn/module.py) | `TAGNNLitModule` |
| [`src/data_modules/tagnn.py`](../src/data_modules/tagnn.py) | DataModule |

---

## Dane

- Źródło: `*/tagnn_examples.pkl` (lista grafów per sesja)
- Ten sam vocab itemów co GRU4Rec (`num_embeddings` z meta)
- Preprocessing: sliding window na trainie, last click na val/test

---

## Metryki

Identyczne jak GRU4Rec: full-catalog val, sampled test, POP baseline. Monitor: `val/recall@20`.

---

## Testy

| Plik | Zakres |
|------|--------|
| [`tests/test_tagnn_lit.py`](../tests/test_tagnn_lit.py) | LightningModule |
| [`tests/test_tagnn_dataset.py`](../tests/test_tagnn_dataset.py) | Dataset |
| [`tests/test_tagnn_graph_batch.py`](../tests/test_tagnn_graph_batch.py) | Batchowanie grafów |
| [`tests/test_tagnn_batch_sampler.py`](../tests/test_tagnn_batch_sampler.py) | Sampler |
| [`tests/test_tagnn_datamodule.py`](../tests/test_tagnn_datamodule.py) | DataModule |
