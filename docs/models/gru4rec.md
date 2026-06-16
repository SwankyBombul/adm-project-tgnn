# GRU4Rec

**GRU4Rec** to sekwencyjny baseline sesyjny: sesja = sekwencja ID itemów, model przewiduje następny klik na podstawie ostatniego stanu ukrytego GRU. Implementacja w [`src/models/gru4rec/`](../src/models/gru4rec/).

---

## Idea

| Element | W projekcie |
|---------|-------------|
| Wejście | Paddingowana sekwencja itemów + długości |
| Encoder | `Embedding` → `GRU` (packed sequence) |
| Wyjście | Linear → logity dla **całego katalogu** |
| Loss | Cross-entropy (jeden poprawny next-item) |

Literatura: Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks* (ICLR 2016).

---

## Uruchomienie

```powershell
uv run python -m src.main fit `
  -c config/data/gru4rec_yoochoose.yaml `
  -c config/model/gru4rec.yaml `
  -c config/experiments/gru4rec_baseline.yaml
```

---

## Konfiguracja

Plik [`config/model/gru4rec.yaml`](../config/model/gru4rec.yaml):

| Parametr | Typowy baseline |
|----------|-----------------|
| `embedding_dim` | 128 |
| `hidden_dim` | 128 |
| `num_layers` | 1 |
| `dropout` | 0.0 |
| `learning_rate` | 1e-3 |
| `pad_idx` | 0 |

`num_embeddings` — ustawiane automatycznie z `meta.json` (CLI).

---

## Moduły

| Plik | Rola |
|------|------|
| [`model.py`](../src/models/gru4rec/model.py) | `GRU4Rec`: encode + forward + `score_candidates` |
| [`dataset.py`](../src/models/gru4rec/dataset.py) | Parquet examples → tensory |
| [`module.py`](../src/models/gru4rec/module.py) | `GRU4RecLitModule` |
| [`src/data_modules/gru4rec.py`](../src/data_modules/gru4rec.py) | Lightning DataModule |

### Forward pass

```text
item_ids (B, L) → Embedding → GRU → hidden (B, H) → Linear → logits (B, num_items)
```

Na ewaluacji sampled: `score_candidates` indeksuje wagi warstwy wyjściowej tylko dla kolumn kandydatów — bez pełnego mnożenia macierzy.

---

## Kontrakt danych

- Źródło: `train/val/test_internal/challenge_test/gru4rec_examples.parquet`
- Kolumny: sekwencja itemów (indeksy GRU), target (następny item)
- PAD = 0; rzeczywiste itemy od indeksu 1

Szczegóły preprocessingu: [`artifacts.md`](artifacts.md).

---

## Metryki

| Etap | Metryka |
|------|---------|
| Val (`fit`) | Full-catalog `recall@20`, `mrr@20`, `ndcg@20` |
| Test (`evaluate`) | Sampled `recall@K` + POP baseline |

Monitor checkpointu: `val/recall@20`.

---

## Testy

| Plik | Zakres |
|------|--------|
| [`tests/test_gru4rec_lit.py`](../tests/test_gru4rec_lit.py) | LightningModule, forward, sampled eval |
| [`tests/test_datamodule.py`](../tests/test_datamodule.py) | DataModule GRU4Rec |
