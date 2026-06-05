# adm-project-tgnn

Projekt z przedmiotu **Advanced Data Mining (ADM)** — rekomendacje sesyjne z wykorzystaniem **Temporal Graph Networks (TGN)** na zbiorze **Yoochoose** (RecSys Challenge 2015).

**Zespół:** Wiktor Małysa (349105), Mikołaj Orzechowski (363917)

Szczegółowy opis problemu, danych i planu eksperymentów: [`docs/first_presentation.md`](docs/first_presentation.md)  
Notatki / prezentacja: [HackMD](https://hackmd.io/56eeHBjMQfmq4Wh2M82m4A)

## Cel projektu

- **Zadanie:** next-item prediction w sesji e-commerce (na podstawie dotychczasowych kliknięć przewidzieć następny produkt).
- **Model główny:** TGN na grafie dynamicznym w czasie ciągłym (sesje ↔ produkty).
- **Baseline:** m.in. modele sekwencyjne (np. SR-GNN / TAGNN — patrz dokumentacja w `docs/`).
- **Dane:** [Yoochoose na Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015).

## Wymagania

- Python **≥ 3.12**
- [uv](https://docs.astral.sh/uv/) (zarządzanie środowiskiem i zależnościami)
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

Pobranie surowych danych:

```powershell
uv run python scripts/download_raw_data.py
```

Po pobraniu w `data/raw/` powinny być m.in.:

| Plik | Opis |
|------|------|
| `yoochoose-clicks.dat` | kliknięcia (train) |
| `yoochoose-buys.dat` | zakupy (train) |
| `yoochoose-test.dat` | kliknięcia (test) |
| `dataset-README.txt` | opis formatu od organizatorów |

Katalog `data/` jest w `.gitignore` (duże pliki `.dat` nie trafiają do repozytorium).

## Struktura repozytorium

```text
adm-project-tgnn/
├── docs/                    # dokumentacja projektu (prezentacja, notatki)
├── scripts/
│   └── download_raw_data.py # pobieranie Yoochoose z Kaggle → data/raw
├── scr/
│   └── utlis.py             # narzędzia wspólne (np. get_project_root)
├── data/
│   └── raw/                 # surowe pliki .dat (lokalnie, po download)
├── pyproject.toml           # zależności i metadane projektu
├── uv.lock
└── README.md
```