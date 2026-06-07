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
Zakres: surowe pliki `clicks`, `buys`, `test` — typy danych, struktura rekordów i różnice między plikami, bez preprocessingu i splitów.

## Struktura repozytorium

```text
adm-project-tgnn/
├── docs/                         # dokumentacja (prezentacja, materiały ADM)
│   ├── first_presentation.md
│   ├── adm_projekt_wm_mo.docx
│   └── projekt_info.pdf
├── notebooks/
│   └── eda_yoochose.ipynb        # wstępna analiza surowych danych Yoochoose
├── scripts/
│   └── download_raw_data.py      # pobieranie danych z Kaggle → data/raw
├── src/
│   ├── __init__.py
│   └── utlis.py                  # narzędzia wspólne (get_project_root)
├── data/
│   └── raw/                      # surowe pliki .dat (lokalnie, po download)
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
| Analiza i ML | `numpy`, `pandas`, `scikit-learn`, `torch` |
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
| EDA surowych danych | `notebooks/eda_yoochose.ipynb` |
| Preprocessing (mapowanie ID, podział czasowy, format pod TGN) | planowane |
| Implementacja TGN + baseline (TAGNN) | planowane |
| Ewaluacja (HR@K, MRR@K) | planowane |

## Licencja i dane

Zbiór Yoochoose pochodzi z [RecSys Challenge 2015](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015). Przed publikacją wyników sprawdź warunki użycia na Kaggle i w `dataset-README.txt`.
