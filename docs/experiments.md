# Eksperymenty i wyniki

Macierz uruchomień baseline’ów oraz odnośniki do logów **Weights & Biases**. Konfiguracje YAML opisane w [`configuration.md`](configuration.md); interpretacja metryk w [`evaluation.md`](evaluation.md).

**Dashboard zespołu:** [https://wandb.ai/project-nn/adm-project-tgnn](https://wandb.ai/project-nn/adm-project-tgnn)

**Środowisko treningu baseline’ów:** NVIDIA **T4** w [Lightning AI Studio](https://lightning.ai/).

---

## Macierz eksperymentów

| Model | Experiment YAML | W&B run name | Epoki | Monitor checkpoint | Val metryka | Test metryka |
|-------|-----------------|--------------|-------|-------------------|-------------|--------------|
| GRU4Rec | `gru4rec_baseline.yaml` | `gru4rec-baseline` | 10 | `val/recall@20` | Full-catalog Recall@K | Sampled Recall@K |
| TAGNN | `tagnn_baseline.yaml` | `tagnn-baseline` | 10 | `val/recall@20` | Full-catalog Recall@K | Sampled Recall@K |
| TGN | `tgn_bce_baseline.yaml` | `tgn-bce-baseline` | 10 | `val/sampled_recall@20` | Sampled Recall@K | Sampled Recall@K |

Alias TGN: `tgn_baseline.yaml` → ta sama nazwa runu `tgn-bce-baseline`.

---

## Smoke testy (weryfikacja kodu)

| Model | YAML | Cel |
|-------|------|-----|
| GRU4Rec | `gru4rec_smoke.yaml` | CPU, 1 epoka, bez W&B |
| TAGNN | `tagnn_smoke.yaml` | CPU, 1 epoka, bez W&B |
| TGN | `tgn_smoke.yaml` | CPU, limit batchy treningu |

Smoke służą szybkiej weryfikacji pipeline’u — **nie** są baseline’ami do porównania w raporcie.

---

## Zbiory ewaluacji

| Split | Opis | Prefiks metryk W&B |
|-------|------|-------------------|
| `val` | Hold-out 15% (podczas `fit`) | `val/` |
| `test_internal` | Wewnętrzny test 15% | `test_internal/sampled_*` |
| `challenge_test` | Pełny Yoochoose test challenge | `challenge_test/sampled_*` |

Dla każdego splitu logowany jest też baseline **POP@K** (`recall@20_pop`).

---

## Hyperparameter search (W&B Sweep)

| Sweep | Plik | Parametry |
|-------|------|-----------|
| GRU4Rec | `config/sweeps/gru4rec.yaml` | lr, embedding/hidden dim, layers, dropout, batch_size |
| TAGNN | `config/sweeps/tagnn.yaml` | lr, hidden_dim, gnn_steps, nonhybrid, weight_decay, batch_size, max_seq_len |

```powershell
wandb sweep config/sweeps/gru4rec.yaml
wandb agent <sweep_id>
```

Wyniki trial-i w zakładce **Sweeps** na W&B.

---

## Dane wejściowe (wspólne)

Wszystkie baseline’y (GRU4Rec, TAGNN, TGN clicks-only) korzystają z:

```text
data/processed/subsample_1_32_clicks_only/
```

- Subsample **1/32** sesji (chronologicznie)
- Split **70/15/15** po czasie
- Train: sliding window; val/test: ostatni klik w sesji

Decyzje EDA: [`data-and-eda.md`](data-and-eda.md), notebook [`notebooks/eda_yoochose.ipynb`](../notebooks/eda_yoochose.ipynb).

---

## Artefakty po treningu

```text
saved_models/<model>/<run_name>/
├── best.ckpt
└── config.yaml          # zapisana konfiguracja LightningCLI
```

Checkpointi **nie** są w repozytorium (`.gitignore`). Do reprodukcji metryk: W&B + lokalny `saved_models/` po `fit`.

---

## Porównanie modeli (oś projektu)

```text
GRU4Rec (sekwencja)  →  TAGNN (graf sesyjny)  →  TGN (graf dynamiczny, czas ciągły)
```

Szczegóły architektur: [`models/gru4rec.md`](models/gru4rec.md), [`models/tagnn.md`](models/tagnn.md), [`models/tgn.md`](models/tgn.md).
