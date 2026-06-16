# Konfiguracja YAML

Eksperymenty są w pełni sterowane plikami YAML w katalogu [`config/`](../config/). **LightningCLI** (`jsonargparse`) łączy wiele plików przez flagę `-c` — kolejność ma znaczenie: późniejsze pliki nadpisują wcześniejsze.

---

## Uruchomienie

Typowy baseline składa się z **czterech warstw**:

```powershell
uv run python -m src.main fit `
  -c config/default.yaml `
  -c config/data/gru4rec_yoochoose.yaml `
  -c config/model/gru4rec.yaml `
  -c config/experiments/gru4rec_baseline.yaml
```

W praktyce `config/default.yaml` jest ładowany automatycznie jako domyślny dla subkomend `fit` i `test` (zob. [`src/utils/cli.py`](../src/utils/cli.py)), więc często wystarczą trzy pliki `data` + `model` + `experiments`.

---

## Układ katalogu `config/`

```text
config/
├── default.yaml              # trainer, callbacks, W&B (wspólne)
├── preprocessing.yaml        # pipeline danych (osobny CLI)
├── data/                     # LightningDataModule per model
├── model/                    # LightningModule (architektura, lr)
├── experiments/              # nadpisania runu (epoki, nazwa W&B, smoke)
└── sweeps/                   # definicje W&B Sweep
```

---

## Warstwy konfiguracji

| Warstwa | Przykład | Odpowiedzialność |
|---------|----------|------------------|
| `default.yaml` | `trainer.max_epochs`, `accelerator: gpu` | Trainer, logger W&B, ModelCheckpoint, EarlyStopping |
| `data/*.yaml` | `gru4rec_yoochoose.yaml` | `class_path` DataModule, `processed_dir`, `batch_size` |
| `model/*.yaml` | `gru4rec.yaml` | `class_path` LightningModule, hiperparametry architektury |
| `experiments/*.yaml` | `gru4rec_baseline.yaml` | Nazwa runu W&B, tagi, ewentualne nadpisania epok / monitora |

### Preprocessing (osobny entry point)

| Plik | CLI |
|------|-----|
| [`config/preprocessing.yaml`](../config/preprocessing.yaml) | `uv run python -m src.preprocessing --config config/preprocessing.yaml` |

Szczegóły: [`preprocessing.md`](preprocessing.md).

---

## Pliki data / model / experiments

### GRU4Rec

| Plik | Opis |
|------|------|
| `config/data/gru4rec_yoochoose.yaml` | DataModule, ścieżka do `subsample_1_32_clicks_only` |
| `config/model/gru4rec.yaml` | `GRU4RecLitModule`, `embedding_dim`, `hidden_dim`, `learning_rate` |
| `config/experiments/gru4rec_baseline.yaml` | 10 epok, W&B name `gru4rec-baseline` |
| `config/experiments/gru4rec_smoke.yaml` | CPU, 1 epoka, logger wyłączony |

### TAGNN

| Plik | Opis |
|------|------|
| `config/data/tagnn_yoochoose.yaml` | DataModule TAGNN |
| `config/model/tagnn.yaml` | `TAGNNLitModule`, `gnn_steps`, `nonhybrid` |
| `config/experiments/tagnn_baseline.yaml` | baseline W&B |
| `config/experiments/tagnn_smoke.yaml` | smoke CPU |

### TGN

| Plik | Opis |
|------|------|
| `config/data/tgn_yoochoose.yaml` | DataModule TGN, `event_batch_size` |
| `config/model/tgn.yaml` | `TGNLitModule`, wymiary pamięci, `num_negatives` |
| `config/experiments/tgn_bce_baseline.yaml` | monitor `val/sampled_recall@20` |
| `config/experiments/tgn_baseline.yaml` | alias nazwy runu `tgn-bce-baseline` |
| `config/experiments/tgn_smoke.yaml` | limit batchy treningu |

---

## W&B Sweep

Definicje w [`config/sweeps/`](../config/sweeps/):

```powershell
wandb sweep config/sweeps/gru4rec.yaml
wandb agent <sweep_id>
```

Mechanizm `${args}` w pliku sweep rozwija wybrane hiperparametry na override’y CLI (`model.init_args.learning_rate=...`). Parametry muszą odpowiadać ścieżkom `jsonargparse` w YAML modelu/danych.

| Sweep | Optymalizowany monitor |
|-------|------------------------|
| `gru4rec.yaml` | `val/recall@20` |
| `tagnn.yaml` | `val/recall@20` |

---

## Nadpisywanie z linii poleceń

Dowolny klucz YAML można nadpisać bez edycji pliku:

```powershell
uv run python -m src.main fit `
  -c config/data/gru4rec_yoochoose.yaml `
  -c config/model/gru4rec.yaml `
  -c config/experiments/gru4rec_baseline.yaml `
  --trainer.max_epochs 5 `
  --model.init_args.learning_rate 0.001
```

---

## Konwencje

- Ścieżki w YAML są **względne do roota repozytorium** (`data/processed/...`).
- `num_embeddings` / `num_items` — ustawiane automatycznie z `meta.json` (nie wpisuj ręcznie po preprocessingu).
- Nazwa runu W&B (`trainer.logger.init_args.name`) determinuje podkatalog w `saved_models/`.

---

## Testy

| Plik | Zakres |
|------|--------|
| [`tests/test_yaml_config.py`](../tests/test_yaml_config.py) | Ładowanie i spójność YAML |
| [`tests/test_cli.py`](../tests/test_cli.py) | Składanie konfiguracji w CLI |
