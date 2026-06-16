# adm-project-tgnn

Projekt z przedmiotu **Advanced Data Mining (ADM)** — rekomendacje sesyjne z wykorzystaniem **Temporal Graph Networks (TGN)** na zbiorze **Yoochoose** (RecSys Challenge 2015).

**Zespół:** Wiktor Małysa (349105), Mikołaj Orzechowski (363917)

| Link | Opis |
|------|------|
| [Weights & Biases](https://wandb.ai/project-nn/adm-project-tgnn) | logi treningów, metryki, porównanie runów |
| [HackMD — notatki](https://hackmd.io/56eeHBjMQfmq4Wh2M82m4A) | prezentacja / notatki zespołu |
| [`docs/first_presentation.md`](docs/first_presentation.md) | opis problemu, danych i planu eksperymentów |
| [Yoochoose na Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) | surowe dane |

---

## Spis treści

- [O projekcie](#o-projekcie)
- [Pipeline w skrócie](#pipeline-w-skrocie)
- [Wymagania](#wymagania)
- [Szybki start (lokalnie)](#szybki-start-lokalnie)
  - [Pobranie danych](#1-pobranie-surowych-danych)
  - [EDA](#2-eksploracja-danych-eda)
  - [Preprocessing](#3-preprocessing)
- [Trening i eksperymenty](#trening-i-eksperymenty)
- [Struktura repozytorium](#struktura-repozytorium)
- [Środowisko i zależności](#środowisko-i-zależności)
- [Konfiguracja YAML](#konfiguracja-yaml-config)
- [Stan prac](#stan-prac)
- [Licencja i dane](#licencja-i-dane)

---

## O projekcie

- **Zadanie:** next-item prediction w sesji e-commerce (na podstawie dotychczasowych kliknięć przewidzieć następny produkt).
- **Model główny (docelowy):** TGN na grafie dynamicznym w czasie ciągłym (sesje ↔ produkty).
- **Baseline:** modele sekwencyjne z grafowej reprezentacji sesji — na start **GRU4Rec**, później **TAGNN** (szczegóły w `docs/first_presentation.md`).
- **Dane:** kliknięcia, zakupy i test challenge'u Yoochoose; po EDA pracujemy na chronologicznym subsample **1/32** pełnych sesji.

## Pipeline w skrócie

Krótki przegląd dla kogoś, kto wchodzi w projekt po raz pierwszy:

```text
Kaggle (.dat)  →  download_raw_data.py  →  data/raw/
       ↓
notebooks/eda_yoochose.ipynb  →  decyzje o subsample, splicie, metrykach
       ↓
src/preprocessing/  →  data/processed/  (GRU4Rec, TAGNN, TGN w jednym przebiegu)
       ↓
src/main.py (LightningCLI)  →  trening GRU4Rec  →  saved_models/ + W&B
       ↓
(planowane) TAGNN → TGN
```

1. **Surowe dane** pobieramy skryptem z Kaggle (`scripts/download_raw_data.py`) — nie są w gicie.
2. **EDA** w notebooku ustaliliśmy m.in. subsample 1/32, split 70/15/15 po czasie, sliding window na trainie i ostatni klik na val/test.
3. **Preprocessing** (`uv run python -m src.preprocessing`) zapisuje gotowe pliki pod trzy modele; kontrakt jest w `meta.json`.
4. **Trening** idzie przez PyTorch Lightning CLI (`src/main.py`) z konfiguracją YAML; logi trafiają na [W&B](https://wandb.ai/project-nn/adm-project-tgnn).
5. **Modele:** GRU4Rec (baseline sekwencyjny) i **TAGNN** (graf sesyjny + target attention); docelowo TGN.

---

## Wymagania

- Python **3.12** (pin w `.python-version`)
- [uv](https://docs.astral.sh/uv/) — środowisko wirtualne i zależności
- Konto Kaggle + klucz API (do pobrania danych lokalnie)
- Konto w zespole W&B **project-nn** (do logowania eksperymentów; poproś o zaproszenie, jeśli go nie masz)

## Szybki start (lokalnie)

```powershell
cd adm-project-tgnn
uv sync
```

Utwórz plik `.env` w katalogu projektu (nie commitujemy go do gita):

```env
KAGGLE_USERNAME=twoj_login
KAGGLE_KEY=twoj_api_key
```

Klucz wygenerujesz w [ustawieniach Kaggle → API](https://www.kaggle.com/settings).

### 1. Pobranie surowych danych

```powershell
uv run python scripts/download_raw_data.py
```

Skrypt pobiera archiwum z Kaggle, rozpakowuje je i przenosi pliki z podfolderu `yoochoose-data/` bezpośrednio do `data/raw/`.

Po pobraniu w `data/raw/` powinny być:

| Plik | Opis |
|------|------|
| `yoochoose-clicks.dat` | kliknięcia (train) |
| `yoochoose-buys.dat` | zakupy (train) |
| `yoochoose-test.dat` | kliknięcia (test) |
| `dataset-README.txt` | opis formatu od organizatorów |

Katalog `data/` jest w `.gitignore` — duże pliki `.dat` nie trafiają do repozytorium.

### 2. Eksploracja danych (EDA)

Otwórz notebook [`notebooks/eda_yoochose.ipynb`](notebooks/eda_yoochose.ipynb).

Zakres analizy:

- jakość danych (braki, spójność `clicks` / `buys` / `test`),
- skala zbioru, długość sesji, popularność itemów,
- oś czasu, propozycja splitu 70/15/15, podpróbkowanie chronologiczne,
- pary next-item, powtórzenia itemów w sesji, kaskadowość przejść,
- `category` (buckety, niespójność per item),
- `buys` vs kliknięcia,
- cold start w `test.dat`, struktura pod grafy (TGN, SR-GNN/TAGNN),
- metryki baseline (POP@20),
- **wnioski i rekomendacje preprocessingu** (ostatnia sekcja notebooka).

Walidację wyjścia preprocessingu można sprawdzić w [`notebooks/validate_preprocessing.ipynb`](notebooks/validate_preprocessing.ipynb).

### 3. Preprocessing

Pipeline w `src/preprocessing/` przygotowuje dane pod **GRU4Rec**, **TAGNN** i **TGN** z jednego przebiegu. Domyślne ustawienia wynikają z EDA:

| Decyzja | Wartość |
|---------|---------|
| Subsample | **1/32** sesji (chronologicznie, pełne sesje) |
| Split wewnętrzny | **70/15/15** po czasie; sesje przypisane w całości |
| Powtórzenia `A→A` w sesji | **zostają** |
| Trening | sliding window — każdy krok next-click |
| Val / test | **tylko ostatni klik** w sesji |
| GRU4Rec / TAGNN | tylko **kliknięcia** |
| TGN | wariant **clicks-only** lub **clicks + buys** (osobne katalogi wyjściowe) |

Parametry preprocessingu trzymamy w **`config/preprocessing.yaml`** (sekcje `paths`, `preprocessing`). CLI nadpisuje YAML.

**Wariant clicks-only** (GRU4Rec, TAGNN, TGN bez zakupów):

```powershell
uv run python -m src.preprocessing --config config/preprocessing.yaml --force
```

**Wariant z buys** (tylko strumień TGN; GRU4Rec/TAGNN korzystają z `clicks_only`):

```powershell
uv run python -m src.preprocessing --include-buys --force
```

Opcjonalnie mniejszy subsample (smoke test):

```powershell
uv run python -m src.preprocessing --fraction 0.001 --force
```

Wyjście trafia do `data/processed/` (w `.gitignore`):

```text
data/processed/
├── subsample_1_32_clicks_only/       # GRU4Rec, TAGNN, TGN (same kliknięcia)
│   ├── meta.json
│   ├── vocab/
│   ├── train/   val/   test_internal/
│   │   ├── gru4rec_examples.parquet
│   │   ├── tagnn_examples.pkl
│   │   └── tgn/
│   │       ├── events.parquet
│   │       └── examples.parquet
│   └── challenge_test/               # pełny yoochoose-test.dat (bez subsample)
└── subsample_1_32_with_buys/         # TGN multigraf (kliknięcia + zakupy)
    ├── meta.json
    ├── vocab/
    └── …/tgn/events.parquet          # bez gru4rec/tagnn
```

Konwencje indeksów i kontraktów wejściowych modeli są opisane w `meta.json` oraz w docstringu `src/preprocessing/config.py`.

---

## Trening i eksperymenty

Workflow: **`fit`** (train + val co epokę) → **`evaluate`** (osobno na `test_internal` i `challenge_test`).

Trening baseline **GRU4Rec** i **TAGNN** uruchamiamy przez **LightningCLI** (`src/main.py`). Domyślny logger to **WandbLogger** — entity `project-nn`, project `adm-project-tgnn`.

**Dashboard zespołu:** [https://wandb.ai/project-nn/adm-project-tgnn](https://wandb.ai/project-nn/adm-project-tgnn)

```powershell
# baseline (GPU, 10 epok, logi na W&B)
uv run python -m src.main fit `
  -c config/data/gru4rec_yoochoose.yaml `
  -c config/model/gru4rec.yaml `
  -c config/experiments/gru4rec_baseline.yaml

# smoke (CPU, 1 epoka, bez W&B)
uv run python -m src.main fit `
  -c config/data/gru4rec_yoochoose.yaml `
  -c config/model/gru4rec.yaml `
  -c config/experiments/gru4rec_smoke.yaml
```

**TAGNN** (ten sam workflow `fit` → `evaluate`):

```powershell
uv run python -m src.main fit `
  -c config/data/tagnn_yoochoose.yaml `
  -c config/model/tagnn.yaml `
  -c config/experiments/tagnn_baseline.yaml

uv run python -m src.main evaluate `
  -c config/data/tagnn_yoochoose.yaml `
  -c config/model/tagnn.yaml `
  -c config/experiments/tagnn_baseline.yaml `
  --ckpt_path best
```

**TGN** (PyG; trening BCE na strumieniu zdarzeń; **walidacja i evaluate** — sampled Recall@K z `eval_num_negatives` negatywów):

```powershell
uv run python -m src.main fit `
  -c config/data/tgn_yoochoose.yaml `
  -c config/model/tgn.yaml `
  -c config/experiments/tgn_bce_baseline.yaml

# smoke (CPU, 5 batchy treningu, bez W&B)
uv run python -m src.main fit `
  -c config/data/tgn_yoochoose.yaml `
  -c config/model/tgn.yaml `
  -c config/experiments/tgn_smoke.yaml

uv run python -m src.main evaluate `
  -c config/data/tgn_yoochoose.yaml `
  -c config/model/tgn.yaml `
  -c config/experiments/tgn_bce_baseline.yaml `
  --ckpt_path best
```

Ewaluacja po treningu (oba zbiory testowe, osobne prefiksy metryk w W&B):

```powershell
uv run python -m src.main evaluate `
  -c config/data/gru4rec_yoochoose.yaml `
  -c config/model/gru4rec.yaml `
  -c config/experiments/gru4rec_baseline.yaml `
  --ckpt_path best
```

### Automatyczny search hiperparametrów (W&B Sweep)

Korzystamy z **wbudowanego mechanizmu W&B Sweep + `${args}`** — żadne zmiany w kodzie nie są potrzebne. Nazwy parametrów w YAML-u odpowiadają ścieżkom `jsonargparse` (np. `model.init_args.learning_rate`). `wandb agent` rozwija `${args}` na `--model.init_args.learning_rate=0.001 ...` i przekazuje do LightningCLI jako CLI overrides.

```powershell
# 1. Utwórz sweep (raz) — zwróci <sweep_id>
wandb sweep scripts/sweeps/gru4rec.yaml

# 2. Uruchom agenta (ile chcesz procesów równolegle)
wandb agent <sweep_id>

# TAGNN:
wandb sweep scripts/sweeps/tagnn.yaml
wandb agent <sweep_id>
```

Domyślnie `method: bayes` + `early_terminate: hyperband` — nieobiecujące trial-e są zabijane wcześnie. Parametry:

| Model | Szukane hiperparametry |
|-------|----------------------|
| **GRU4Rec** | `learning_rate` (log), `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`, `batch_size` |
| **TAGNN** | `learning_rate` (log), `hidden_dim`, `gnn_steps`, `nonhybrid`, `weight_decay` (log), `batch_size`, `max_seq_len` |

Wyniki wszystkich trial-i widać w [W&B Dashboard](https://wandb.ai/project-nn/adm-project-tgnn) w zakładce **Sweeps**.

### Zapis modeli (`saved_models/`)

```text
saved_models/
├── gru4rec/
│   └── gru4rec-baseline/
│       ├── best.ckpt
│       └── config.yaml
└── tagnn/
    └── tagnn-baseline/
        ├── best.ckpt
        └── config.yaml
└── tgn/
    └── tgn-bce-baseline/
        ├── best.ckpt
        └── config.yaml
```

`--ckpt_path best` wskazuje na `saved_models/gru4rec/<run_name>/best.ckpt` (ten sam experiment YAML co przy `fit`). Możesz też podać pełną ścieżkę do `.ckpt`.

Podczas `fit` checkpoint wybierany jest po `val/recall@20` (GRU4Rec/TAGNN) lub `val/sampled_recall@20` (TGN). Metryki z `evaluate` trafiają do W&B jako `test_internal/sampled_*` i `challenge_test/sampled_*` (sampled Recall@K: target + `eval_num_negatives` losowych negatywów z katalogu). Baseline POP (`recall@20_pop`) bez zmian. Implementacja w `src/evaluation/sampled.py`.

Ustawienia W&B (`entity`, `project`, `login_wandb()`, `verify_wandb_access()`) są w `src/config/wandb_settings.py`.

---

## Struktura repozytorium

```text
adm-project-tgnn/
├── docs/                         # dokumentacja (prezentacja, materiały ADM)
│   ├── first_presentation.md
│   ├── adm_projekt_wm_mo.docx
│   └── projekt_info.pdf
├── notebooks/                    # EDA i walidacja preprocessingu
│   ├── eda_yoochose.ipynb
│   └── validate_preprocessing.ipynb
├── config/                       # YAML: preprocessing, trening GRU4Rec
├── scripts/
│   └── download_raw_data.py      # pobieranie danych z Kaggle → data/raw
├── src/
│   ├── common/                   # ścieżki, stałe (get_project_root, PAD_IDX)
│   ├── artifacts/                # odczyt meta.json i ścieżek do processed/
│   ├── preprocessing/            # pipeline danych (load → export)
│   ├── models/
│   │   ├── gru4rec/              # model + dataset + LightningModule
│   │   ├── tagnn/                # TAGNN (port CRIPAC-DIG/TAGNN)
│   │   └── tgn/                  # TGN (BCE train, sampled val + test)
│   ├── data_modules/             # LightningDataModule (GRU4Rec, TAGNN, TGN)
│   ├── training/                 # NextItemLitModule, saved_models paths
│   ├── main.py                   # LightningCLI: fit / evaluate
│   ├── evaluation/               # metryki i baseline'y
│   └── config/                   # wandb defaults, preprocessing YAML loader
├── tests/                        # testy jednostkowe
├── data/
│   ├── raw/                      # surowe pliki .dat (lokalnie, po download)
│   └── processed/                # wynik preprocessingu (lokalnie)
├── .python-version               # Python 3.12
├── pyproject.toml
├── uv.lock
└── README.md
```

### Narzędzia w kodzie

**`get_project_root()`** — `src/common/paths.py`

Zwraca absolutną ścieżkę do roota repozytorium. Używana w skrypcie pobierania danych i w notebookach:

```python
from src.common.paths import get_project_root

raw_dir = get_project_root() / "data" / "raw"
```

Konwencje pakietów (`src/common/__init__.py`): preprocessing zapisuje artefakty na dysk, reszta kodu czyta je przez `src/artifacts/`.

---

## Środowisko i zależności

Projekt używa **uv**. Główne pakiety (`pyproject.toml`):

| Obszar | Pakiety |
|--------|---------|
| Dane / API | `kaggle`, `python-dotenv` |
| Analiza i ML | `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `torch` (CUDA 12.6 na Linux/Windows) |
| Eksperymenty | `jupyter`, `matplotlib`, `ipykernel`, `lightning`, `wandb` |
| Grafy temporalne | `torch-geometric` |

**PyTorch + CUDA:** na Windows i Linux `uv sync` instaluje `torch` z indeksu PyTorch (`cu126`). macOS dostaje build CPU z PyPI. Po instalacji sprawdź GPU:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Trening z `accelerator: gpu` (domyślnie w `config/default.yaml`) używa GPU, gdy CUDA jest dostępne.

---

## Konfiguracja YAML (`config/`)

Eksperymenty i domyślne parametry trzymamy w plikach YAML w katalogu `config/`:

| Plik | Opis |
|------|------|
| `config/preprocessing.yaml` | subsample, split, ścieżki raw/processed |
| `config/default.yaml` | trainer, callbacks, wandb (LightningCLI) |
| `config/data/gru4rec_yoochoose.yaml` | LightningDataModule — ścieżki i batch |
| `config/model/gru4rec.yaml` | LightningModule — architektura i lr |
| `config/experiments/gru4rec_baseline.yaml` | baseline (W&B name, 10 epok) |
| `config/experiments/gru4rec_smoke.yaml` | smoke (CPU, 1 epoka, bez W&B) |
| `config/data/tagnn_yoochoose.yaml` | LightningDataModule TAGNN |
| `config/model/tagnn.yaml` | LightningModule TAGNN |
| `config/experiments/tagnn_baseline.yaml` | TAGNN baseline (W&B, 10 epok) |
| `config/experiments/tagnn_smoke.yaml` | TAGNN smoke (CPU, 1 epoka) |
| `config/data/tgn_yoochoose.yaml` | LightningDataModule TGN |
| `config/model/tgn.yaml` | LightningModule TGN (BCE training) |
| `config/experiments/tgn_bce_baseline.yaml` | TGN baseline (W&B, 10 epok) |
| `config/experiments/tgn_baseline.yaml` | alias `tgn-bce-baseline` |
| `config/experiments/tgn_smoke.yaml` | TGN smoke (CPU, limit batches) |

Ścieżki w YAML są względne do roota repozytorium (`data/processed`, `data/raw`).

---

## Stan prac

| Etap | Status |
|------|--------|
| Konfiguracja projektu (`uv`, Python 3.12, zależności) | gotowe |
| Pobieranie danych z Kaggle | `scripts/download_raw_data.py` |
| EDA (jakość, next-item, kaskadowość, category, buys, cold start, TGN) | `notebooks/eda_yoochose.ipynb` |
| Walidacja preprocessingu | `notebooks/validate_preprocessing.ipynb` |
| Preprocessing (subsample, split, vocab, eksport GRU/TAGNN/TGN) | `src/preprocessing/` |
| Baseline GRU4Rec (fit + evaluate, W&B, `saved_models/`) | `src/main.py`, `src/training/`, `config/` |
| Baseline TAGNN (fit + evaluate, port SIGIR 2020) | `src/models/tagnn/`, `config/data/tagnn_yoochoose.yaml` |
| TGN (PyG, BCE fit + sampled evaluate) | `src/models/tgn/`, `config/data/tgn_yoochoose.yaml` |
| Metryki rankingowe + POP@20 baseline | `src/evaluation/` |
| Ustawienia W&B | `src/config/wandb_settings.py` |

Wyniki treningów GRU4Rec: [W&B — adm-project-tgnn](https://wandb.ai/project-nn/adm-project-tgnn).

---

## Licencja i dane

Zbiór Yoochoose pochodzi z [RecSys Challenge 2015](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015). Przed publikacją wyników sprawdź warunki użycia na Kaggle i w `dataset-README.txt`.
