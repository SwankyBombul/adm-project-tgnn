# adm-project-tgnn

Projekt z przedmiotu **Advanced Data Mining (ADM)** — rekomendacje sesyjne z wykorzystaniem **Temporal Graph Networks (TGN)** na zbiorze **Yoochoose** (RecSys Challenge 2015).

**Zespół:** Wiktor Małysa (349105), Mikołaj Orzechowski (363917)

Szczegółowy opis problemu, danych i planu eksperymentów: [`docs/first_presentation.md`](docs/first_presentation.md)  
Notatki / prezentacja: [HackMD](https://hackmd.io/56eeHBjMQfmq4Wh2M82m4A)

## Cel projektu

- **Zadanie:** next-item prediction w sesji e-commerce (na podstawie dotychczasowych kliknięć przewidzieć następny produkt).
- **Model główny:** TGN na grafie dynamicznym w czasie ciągłym (sesje ↔ produkty).
- **Baseline:** modele sekwencyjne z grafowej reprezentacji sesji (np. **TAGNN** — patrz `docs/first_presentation.md`).
- **Dane:** [Yoochoose na Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015).

## Wymagania

- Python **3.12** (pin w `.python-version`)
- [uv](https://docs.astral.sh/uv/) — środowisko wirtualne i zależności
- Konto Kaggle + klucz API (do pobrania danych)

## Szybki start

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

## Struktura repozytorium

```text
adm-project-tgnn/
├── docs/                         # dokumentacja (prezentacja, materiały ADM)
│   ├── first_presentation.md
│   ├── adm_projekt_wm_mo.docx
│   └── projekt_info.pdf
├── notebooks/                    # EDA / walidacja preprocessingu (colab — później)
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
│   │   └── gru4rec/              # model + dataset + LightningModule
│   ├── data_modules/             # LightningDataModule (GRU4Rec, …)
│   ├── main.py                   # LightningCLI: fit / validate / test
│   ├── runtime/                  # Colab (Drive, unpack)
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

## Środowisko i zależności

Projekt używa **uv**. Główne pakiety (`pyproject.toml`):

| Obszar | Pakiety |
|--------|---------|
| Dane / API | `kaggle`, `python-dotenv` |
| Analiza i ML | `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `torch` (CUDA 12.6 na Linux/Windows) |
| Eksperymenty | `jupyter`, `matplotlib`, `ipykernel`, `lightning`, `wandb` |

**PyTorch + CUDA:** na Windows i Linux `uv sync` instaluje `torch` z indeksu PyTorch (`cu126`). macOS dostaje build CPU z PyPI. Po instalacji sprawdź GPU:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Trening z `accelerator: gpu` (domyślnie w `config/default.yaml`) używa GPU, gdy CUDA jest dostępne.

## Narzędzia w kodzie

**`get_project_root()`** — `src/common/paths.py`

Zwraca absolutną ścieżkę do roota repozytorium. Używana w skrypcie pobierania danych i w notebooku EDA:

```python
from src.common.paths import get_project_root

raw_dir = get_project_root() / "data" / "raw"
```

Konwencje pakietów (`src/common/__init__.py`): preprocessing zapisuje artefakty na dysk, reszta kodu czyta je przez `src/artifacts/`.

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

**Preprocessing:**

```powershell
uv run python -m src.preprocessing --config config/preprocessing.yaml --force
```

**Trening GRU4Rec (LightningCLI):**

```powershell
# baseline
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

Walidacja:

```powershell
uv run python -m src.main validate `
  -c config/data/gru4rec_yoochoose.yaml `
  -c config/model/gru4rec.yaml `
  --ckpt_path best
```

`best` / `last` szukają najnowszego pliku `.ckpt` w `checkpoints/gru4rec/` (przydatne po osobnym `fit`). Możesz też podać pełną ścieżkę, np. `checkpoints/gru4rec/adm-project-tgnn/by69hj21/checkpoints/best-epoch_004.ckpt`.

**Colab** — `prepare_colab_session(drive_dir, run_name=...)` rozpakowuje dane, potem `python -m src.main fit ...` z `data.init_args.processed_dir` wskazującym na lokalny unpack.

Ścieżki w YAML są względne do roota repozytorium (`data/processed`, `data/raw`).

## Stan prac

| Etap | Status |
|------|--------|
| Konfiguracja projektu (`uv`, Python 3.12, zależności) | gotowe |
| Pobieranie danych z Kaggle | `scripts/download_raw_data.py` |
| EDA (jakość, next-item, kaskadowość, category, buys, cold start, TGN) | `notebooks/eda_yoochose.ipynb` |
| Preprocessing (subsample, split, vocab, eksport GRU/TAGNN/TGN) | `src/preprocessing/` — `uv run python -m src.preprocessing` |
| Baseline GRU4Rec (LightningCLI, W&B, checkpoint `best`) | `src/main.py`, `config/`, `src/models/gru4rec/` |
| Metryki rankingowe + POP@20 baseline | `src/evaluation/` |
| TAGNN → TGN | planowane |
| Notebooki Colab / trening interaktywny | planowane (po przebudowie) |

## Licencja i dane

Zbiór Yoochoose pochodzi z [RecSys Challenge 2015](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015). Przed publikacją wyników sprawdź warunki użycia na Kaggle i w `dataset-README.txt`.
