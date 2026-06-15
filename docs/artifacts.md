# Artefakty preprocessingu

Pakiet [`src/artifacts/`](../src/artifacts/) zapewnia **read-only** dostęp do danych zapisanych przez [`src/preprocessing/`](../src/preprocessing/). Kod treningowy **nie importuje** modułów preprocessingu — tylko czyta pliki z dysku (konwencja w [`src/common/__init__.py`](../src/common/__init__.py)).

Wstecz: jak powstają artefakty → [`preprocessing.md`](preprocessing.md).  
Decyzje EDA → [`data-and-eda.md`](data-and-eda.md).

---

## Katalog wyjściowy

Wzorzec nazwy (logika w [`PreprocessConfig.output_dir()`](../src/preprocessing/config.py)):

```text
{output_root}/subsample_{1_N|full}_{clicks_only|with_buys}/
```

Domyślnie: `data/processed/subsample_1_32_clicks_only/`.

```text
data/processed/subsample_1_32_clicks_only/
├── meta.json
├── vocab/
│   ├── item_vocab.json
│   └── cat_bucket2idx.json
├── train/
│   ├── gru4rec_examples.parquet
│   ├── tagnn_examples.pkl
│   └── tgn/
│       ├── events.parquet
│       └── examples.parquet
├── val/                    # ta sama struktura
├── test_internal/
└── challenge_test/         # pełny yoochoose-test.dat (bez subsample)
```

Wariant `with_buys` zapisuje strumień TGN z zakupami i **pomija** pliki GRU4Rec/TAGNN (`exports.sequence_models=false` w `meta.json`).

---

## `meta.json` — schemat

Plik zapisywany przez [`write_meta()`](../src/preprocessing/export.py) na końcu pipeline’u ([`pipeline.py`](../src/preprocessing/pipeline.py)). Odczyt: [`load_meta()`](../src/artifacts/meta.py).

```133:174:src/preprocessing/export.py
def write_meta(
    output_dir: Path,
    config: PreprocessConfig,
    vocab: ItemVocab,
    boundaries: dict[str, str],
    stats: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": config.to_dict(),
        "boundaries": boundaries,
        "stats": stats,
        "index_conventions": {
            "gru4rec": {
                "pad_idx": 0,
                "item_idx_range": f"1..{vocab.n_items}",
                "unk_idx": vocab.n_items + 1,
                "embedding_num_embeddings": vocab.gru_vocab_size,
            },
            "tagnn": {
                "node_per_click": True,
                "edges": "consecutive (i -> i+1) built in training code",
                "item_idx_scheme": "same as gru4rec",
            },
            "tgn": {
                "item_idx_range": f"0..{vocab.n_items - 1}",
                "unk_idx": vocab.n_items,
                "bipartite": "session_idx -> item_idx",
                "edge_attr": [
                    "cat_bucket_idx",
                    "price_log",
                    "quantity",
                    "event_type",
                ],
                "positive_edges": "next click only (is_click=True)",
            },
        },
        "exports": {
            "sequence_models": not config.include_buys,
            "tgn_events_include_buys": config.include_buys,
        },
    }
```

| Pole | Zawartość |
|------|-----------|
| `config` | Pełna konfiguracja preprocessingu (`PreprocessConfig.to_dict()`) |
| `boundaries` | Granice czasowe splitu (`train_end`, `val_end` jako ISO) |
| `stats` | Liczby sesji, kliknięć, `cold_start_items` itd. |
| `index_conventions` | Konwencje indeksów per model |
| `exports` | Flagi: czy wyeksportowano GRU4Rec/TAGNN, czy TGN ma buys |

Helpery na meta: [`src/artifacts/meta.py`](../src/artifacts/meta.py) — `gru4rec_vocab_size()`, `tgn_num_items()`, `tgn_num_sessions()`.

---

## Kontrakty kolumn modeli

Źródło prawdy — docstring w [`src/preprocessing/config.py`](../src/preprocessing/config.py):

```63:87:src/preprocessing/config.py
# --- Model input contracts (for downstream training code) ---
#
# GRU4Rec
#   - File: {split}/gru4rec_examples.parquet
#   - Columns: example_id, session_id, history_item_idx (list[int]), target_item_idx (int)
#   - history uses PAD_IDX=0 only in batching; stored lists have no padding
#   - item_idx: 1..n_items known, n_items+1 UNK
#   - Train: all next-click steps; val/test: last click only
#
# TAGNN (session graph, per-click nodes — same as SR-GNN/TAGNN papers)
#   - File: {split}/tagnn_sessions.pkl list[dict]
#   - Each session: item_ids (list[int], one per click), target_item_idx, session_id
#   - Graph built in model: nodes 0..len-1, edges (i -> i+1)
#   - Train: one record per next-click step; val/test: last click only
#
# TGN (PyG TemporalData-compatible export)
#   - File: {split}/tgn/events.parquet
#   - Columns: event_id, session_idx, item_idx, t_sec, event_type,
#              cat_bucket_idx, price_log, quantity, is_click
#   - Bipartite link: session_idx -> item_idx at time t_sec
#   - item_idx: 0..n_items-1 known, n_items UNK (no padding)
#   - include_buys=False: only clicks, event_type=0, price/quantity=0
#   - include_buys=True: clicks + buys; positives for next-click only on is_click rows
#   - File: {split}/tgn/examples.parquet — next-click supervision rows
#
# Challenge test (clicks only, no subsample): challenge_test/
```

---

## Konwencje indeksów

| Model | Znane itemy | UNK | Padding |
|-------|-------------|-----|---------|
| GRU4Rec / TAGNN | `1 … n_items` | `n_items + 1` | `PAD_IDX = 0` tylko przy batchowaniu |
| TGN | `0 … n_items - 1` | `n_items` | brak paddingu |

Stała paddingu: [`src/common/constants.py`](../src/common/constants.py):

```3:5:src/common/constants.py
# GRU4Rec / TAGNN index conventions (also documented in meta.json).
PAD_IDX = 0
# Known items use 1..n_items; UNK uses n_items + 1.
```

Mapowanie raw `item_id` → indeksy: [`src/preprocessing/vocab.py`](../src/preprocessing/vocab.py) (`gru_index`, `tgn_index`). Odczyt w treningu: [`load_gru_item2idx()`](../src/artifacts/vocab.py), [`load_tgn_item2idx()`](../src/artifacts/vocab.py).

---

## Pliki `vocab/`

| Plik | Zawartość | Zapis | Odczyt |
|------|-----------|-------|--------|
| `item_vocab.json` | `item2idx`, `n_items`, indeksy UNK | [`export.py`](../src/preprocessing/export.py) | [`src/artifacts/vocab.py`](../src/artifacts/vocab.py) |
| `cat_bucket2idx.json` | mapowanie bucket → int | [`export.py`](../src/preprocessing/export.py) | DataModule TGN |

---

## Ścieżki per split

Typy splitów: `train`, `val`, `test_internal`, `challenge_test` — [`SplitName`](../src/artifacts/paths.py) w [`src/artifacts/paths.py`](../src/artifacts/paths.py).

| Model | Ścieżka (względem `processed_dir`) | Resolver |
|-------|-----------------------------------|----------|
| GRU4Rec | `{split}/gru4rec_examples.parquet` | `split_examples_path(..., "gru4rec")` |
| TAGNN | `{split}/tagnn_examples.pkl` (fallback `.parquet`) | `split_examples_path(..., "tagnn")` |
| TGN examples | `{split}/tgn/examples.parquet` | `split_examples_path(..., "tgn")` |
| TGN events | `{split}/tgn/events.parquet` | `split_events_path(...)` |

```12:35:src/artifacts/paths.py
def split_examples_path(
    processed_dir: Path,
    split: SplitName,
    model: ModelFormat,
) -> Path:
    split_dir = processed_dir / split
    if model == "gru4rec":
        return split_dir / "gru4rec_examples.parquet"
    if model == "tagnn":
        pkl_path = split_dir / "tagnn_examples.pkl"
        if pkl_path.is_file():
            return pkl_path
        return split_dir / "tagnn_examples.parquet"
    if model == "tgn":
        return split_dir / "tgn" / "examples.parquet"
    raise ValueError(f"Unsupported model format: {model}")


def split_events_path(
    processed_dir: Path,
    split: SplitName,
) -> Path:
    """Path to TGN temporal event stream for a split."""
    return processed_dir / split / "tgn" / "events.parquet"
```

---

## API `src/artifacts/`

Eksport z [`src/artifacts/__init__.py`](../src/artifacts/__init__.py):

| Funkcja | Użycie |
|---------|--------|
| `load_meta(processed_dir)` | Wszystkie DataModule, baselines, CLI |
| `gru4rec_vocab_size(meta)` | Rozmiar `nn.Embedding` GRU4Rec/TAGNN |
| `tgn_num_items(meta)` | Liczba węzłów item w TGN (known + UNK) |
| `tgn_num_sessions(meta, split)` | Liczba sesji w splicie |
| `load_gru_item2idx(vocab_dir)` | Mapowanie raw → GRU4Rec |
| `load_tgn_item2idx(vocab_dir)` | Mapowanie raw → TGN |
| `split_examples_path(...)` | Ścieżka parquet/pkl przykładów |
| `split_events_path(...)` | Ścieżka strumienia zdarzeń TGN |

---

## Kto czyta artefakty

| Komponent | Plik | Co ładuje |
|-----------|------|-----------|
| GRU4Rec DataModule | [`src/data_modules/gru4rec.py`](../src/data_modules/gru4rec.py) | `load_meta`, `split_examples_path`, `gru4rec_vocab_size` |
| GRU4Rec Dataset | [`src/models/gru4rec/dataset.py`](../src/models/gru4rec/dataset.py) | parquet z historią i targetem |
| TAGNN DataModule | [`src/data_modules/tagnn.py`](../src/data_modules/tagnn.py) | `split_examples_path` (pkl) |
| TAGNN Dataset | [`src/models/tagnn/dataset.py`](../src/models/tagnn/dataset.py) | graf `(i → i+1)` z listy kliknięć |
| TGN DataModule | [`src/data_modules/tgn.py`](../src/data_modules/tgn.py) | `split_events_path`, `split_examples_path`, `tgn_num_items` |
| TGN Dataset | [`src/models/tgn/dataset.py`](../src/models/tgn/dataset.py) | `events.parquet` → `TemporalData` |
| Lightning CLI | [`src/utils/cli.py`](../src/utils/cli.py) | auto `num_embeddings` / `num_items` z meta |
| POP baseline | [`src/evaluation/baselines.py`](../src/evaluation/baselines.py) | `load_meta` (statystyki popularności) |

---

## Konfiguracja treningu (`processed_dir`)

Pole `processed_dir` w YAML **musi** wskazywać ten sam katalog, który wygenerował preprocessing:

| Model | Plik config |
|-------|-------------|
| GRU4Rec | [`config/data/gru4rec_yoochoose.yaml`](../config/data/gru4rec_yoochoose.yaml) |
| TAGNN | [`config/data/tagnn_yoochoose.yaml`](../config/data/tagnn_yoochoose.yaml) |
| TGN | [`config/data/tgn_yoochoose.yaml`](../config/data/tgn_yoochoose.yaml) |

Domyślna wartość we wszystkich trzech: `data/processed/subsample_1_32_clicks_only`.

---

## Testy i walidacja

| Plik testu | Zakres |
|------------|--------|
| [`tests/test_training_paths.py`](../tests/test_training_paths.py) | Ścieżki do artefaktów |
| [`tests/test_datamodule.py`](../tests/test_datamodule.py) | GRU4Rec DataModule |
| [`tests/test_tagnn_datamodule.py`](../tests/test_tagnn_datamodule.py) | TAGNN DataModule |
| [`tests/test_tgn_datamodule.py`](../tests/test_tgn_datamodule.py) | TGN DataModule |

Notebook: [`notebooks/validate_preprocessing.ipynb`](../notebooks/validate_preprocessing.ipynb) — porównanie wyjścia z oczekiwaniami EDA.

---

## Powiązane

- [`preprocessing.md`](preprocessing.md) — jak powstają artefakty
- [`overview.md`](overview.md) — mapa modeli i eksperymentów
- [`README.md`](README.md) — spis dokumentacji
- Planowane: `training.md` — LightningCLI, checkpointy, W&B
