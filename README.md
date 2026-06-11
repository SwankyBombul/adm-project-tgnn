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

**Wariant clicks-only** (GRU4Rec, TAGNN, TGN bez zakupów):

```powershell
uv run python -m src.preprocessing --force
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
├── notebooks/
│   └── eda_yoochose.ipynb        # EDA + wnioski do preprocessingu
├── scripts/
│   └── download_raw_data.py      # pobieranie danych z Kaggle → data/raw
├── src/
│   ├── preprocessing/            # pipeline danych (load → export)
│   ├── __init__.py
│   └── utlis.py                  # narzędzia wspólne (get_project_root)
├── tests/
│   └── test_preprocessing.py     # testy jednostkowe helperów
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
| Analiza i ML | `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `torch` |
| Eksperymenty | `jupyter`, `matplotlib`, `ipykernel` |

## Narzędzia w kodzie

**`get_project_root()`** — `src/utlis.py`

Zwraca absolutną ścieżkę do roota repozytorium. Używana w skrypcie pobierania danych i w notebooku EDA:

```python
from src.utlis import get_project_root

raw_dir = get_project_root() / "data" / "raw"
```

## Stan prac

| Etap | Status |
|------|--------|
| Konfiguracja projektu (`uv`, Python 3.12, zależności) | gotowe |
| Pobieranie danych z Kaggle | `scripts/download_raw_data.py` |
| EDA (jakość, next-item, kaskadowość, category, buys, cold start, TGN) | `notebooks/eda_yoochose.ipynb` |
| Preprocessing (subsample, split, vocab, eksport GRU/TAGNN/TGN) | `src/preprocessing/` — `uv run python -m src.preprocessing` |
| TGN: porównanie wariantów clicks-only vs clicks+buys | planowane (dwa przebiegi preprocessingu) |
| Implementacja modeli (GRU4Rec → TAGNN → TGN) | planowane |
| Ewaluacja (Recall@20, MRR@20, POP@20 baseline) | planowane |

## Licencja i dane

Zbiór Yoochoose pochodzi z [RecSys Challenge 2015](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015). Przed publikacją wyników sprawdź warunki użycia na Kaggle i w `dataset-README.txt`.
