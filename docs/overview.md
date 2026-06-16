# Przegląd projektu

**Przedmiot:** Advanced Data Mining (ADM)  
**Zespół:** Wiktor Małysa (349105), Mikołaj Orzechowski (363917)  
**Zadanie:** rekomendacje sesyjne next-item na zbiorze **Yoochoose** (RecSys Challenge 2015).

Pełna wersja akademicka (literatura, szczegóły TGN-attn): [`first_presentation.md`](first_presentation.md).  
Decyzje o danych: [`data-and-eda.md`](data-and-eda.md). Implementacja pipeline: [`preprocessing.md`](preprocessing.md).

---

## Definicja problemu

Systemy e-commerce rejestrują krótkie **sesje** interakcji (kliknięcia produktów). Zadanie to **dynamiczne rekomendacje sesyjne**: na podstawie dotychczasowych działań przewidzieć **następny produkt**, zanim sesja się zakończy.

**Next-Item Prediction** w tym projekcie:

- Mając historię do momentu \(t_{n-1}\), przewidujemy item kliknięty w \(t_n\).
- Uczymy się **samonadzorowanie** — każde kolejne kliknięcie jest etykietą dla poprzedniego kroku.

W podejściu **Temporal Graph Networks (TGN)** zadanie staje się **link prediction** w grafie dynamicznym ewoluującym w **czasie ciągłym** (CTDG), a nie czystą sekwencją jak w RNN.

---

## Model grafowy: CTDG

| Element | Definicja w projekcie |
|--------|------------------------|
| **Wierzchołki** | Graf dwudzielny: **sesje** ↔ **przedmioty** (items). W Yoochoose brak jawnego user ID — każda sesja to osobny węzeł „użytkownika”. |
| **Zdarzenia** | Kliknięcie = interakcja między sesją a itemem w czasie \(t\). Opcjonalnie zakup jako drugi typ zdarzenia ([`src/preprocessing/events.py`](../src/preprocessing/events.py)). |
| **Cel** | Dla pary \((sesja, t)\) wskazać item z najwyższym prawdopodobieństwem — **future edge / link prediction**. |

Implementacja strumienia zdarzeń i eksportu TGN: [`src/preprocessing/events.py`](../src/preprocessing/events.py), [`src/models/tgn/dataset.py`](../src/models/tgn/dataset.py).

---

## Dane: Yoochoose

**Źródło:** [RecSys Challenge 2015 na Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015). Pobranie lokalne: [`scripts/download_raw_data.py`](../scripts/download_raw_data.py).

| Plik | Zawartość |
|------|-----------|
| `yoochoose-clicks.dat` | Kliknięcia (train) |
| `yoochoose-buys.dat` | Zakupy (train) |
| `yoochoose-test.dat` | Kliknięcia testowe challenge |

Parser: [`src/preprocessing/load.py`](../src/preprocessing/load.py). Format czasu: ISO z milisekundami — istotne dla \(\Delta t\) w TGN.

**Dlaczego Yoochoose pasuje do TGN:** gęste sesje z krótkimi odstępami, opcjonalne typy zdarzeń (clicks/buys), cechy krawędzi (`category`, `price`, `quantity`), cold start w teście challenge. Szczegóły liczbowe: [`notebooks/eda_yoochose.ipynb`](../notebooks/eda_yoochose.ipynb), podsumowanie decyzji: [`data-and-eda.md`](data-and-eda.md).

---

## Oś modeli: GRU4Rec → TAGNN → TGN

Porównanie na **tym samym** podzbiorze po preprocessingu ([`src/preprocessing/pipeline.py`](../src/preprocessing/pipeline.py)):

```text
GRU4Rec (sekwencja)  →  TAGNN (graf sesyjny + target attention)  →  TGN (graf dynamiczny, czas ciągły)
```

| Model | Idea | Czas / reprezentacja |
|-------|------|----------------------|
| **GRU4Rec** | Sesja = sekwencja ID; GRU → następny item | Ukryty w kolejności kroków |
| **TAGNN** | Sesja = graf skierowany (klik → klik); GGNN + target attention | Dyskretna kolejność w sesji |
| **TGN** | Globalny graf dwudzielny sesja–item; pamięć + wiadomości + temporal attention | Ciągły, \(\Delta t\) między zdarzeniami |

W prezentacji początkowej planowano SR-GNN; w repozytorium baseline grafowy to **TAGNN** (port [SIGIR 2020](https://arxiv.org/pdf/2005.02844)) — [`src/models/tagnn/`](../src/models/tagnn/).

---

## Mapa model → kod w repo

| Model | Pakiet | DataModule | Config data | Config model | Experiment baseline |
|-------|--------|------------|-------------|--------------|---------------------|
| GRU4Rec | [`src/models/gru4rec/`](../src/models/gru4rec/) (`model.py`, `dataset.py`, `module.py`) | [`src/data_modules/gru4rec.py`](../src/data_modules/gru4rec.py) | [`config/data/gru4rec_yoochoose.yaml`](../config/data/gru4rec_yoochoose.yaml) | [`config/model/gru4rec.yaml`](../config/model/gru4rec.yaml) | [`config/experiments/gru4rec_baseline.yaml`](../config/experiments/gru4rec_baseline.yaml) |
| TAGNN | [`src/models/tagnn/`](../src/models/tagnn/) (`model.py`, `dataset.py`, `graph_batch.py`, `batch_sampler.py`, `module.py`) | [`src/data_modules/tagnn.py`](../src/data_modules/tagnn.py) | [`config/data/tagnn_yoochoose.yaml`](../config/data/tagnn_yoochoose.yaml) | [`config/model/tagnn.yaml`](../config/model/tagnn.yaml) | [`config/experiments/tagnn_baseline.yaml`](../config/experiments/tagnn_baseline.yaml) |
| TGN | [`src/models/tgn/`](../src/models/tgn/) (`model.py`, `memory.py`, `embedding.py`, `decoder.py`, `dataset.py`, `temporal_batch.py`, `module.py`) | [`src/data_modules/tgn.py`](../src/data_modules/tgn.py) | [`config/data/tgn_yoochoose.yaml`](../config/data/tgn_yoochoose.yaml) | [`config/model/tgn.yaml`](../config/model/tgn.yaml) | [`config/experiments/tgn_bce_baseline.yaml`](../config/experiments/tgn_bce_baseline.yaml) |

**Punkt wejścia treningu:** [`src/main.py`](../src/main.py) → [`src/utils/cli.py`](../src/utils/cli.py) (`AdmLightningCLI`). Wspólna baza modułów Lightning: [`src/training/base_module.py`](../src/training/base_module.py).

**Splity testowe** (nazwy w [`src/artifacts/paths.py`](../src/artifacts/paths.py)):

| Split | Znaczenie |
|-------|-----------|
| `test_internal` | Transdukcyjny — z chronologicznego subsample 70/15/15 |
| `challenge_test` | Indukcyjny — pełny `yoochoose-test.dat`, nowe sesje |

Kontrakt artefaktów na dysku: [`artifacts.md`](artifacts.md).

---

## Architektura TGN (koncepcja)

Pięć modułów TGN (Rossi et al.) — implementacja w [`src/models/tgn/`](../src/models/tgn/):

| Moduł | Wybór w projekcie |
|-------|-------------------|
| **Memory** | Wektor stanu per węzeł; init = 0 — [`memory.py`](../src/models/tgn/memory.py) |
| **Message function** | Identity (konkatenacja stanu, \(\Delta t\), cech krawędzi) |
| **Message aggregator** | Last (najświeższa wiadomość w batchu) |
| **Memory updater** | GRU |
| **Embedding** | Temporal Graph Attention — [`embedding.py`](../src/models/tgn/embedding.py) |

**Dekoder:** MLP na parach embeddingów → [`decoder.py`](../src/models/tgn/decoder.py).  
**Loss treningowa:** BCE z negative sampling — [`src/models/tgn/module.py`](../src/models/tgn/module.py).  
**Information leakage:** pamięć aktualizowana zdarzeniami z poprzednich batchy, nie z bieżącej predykcji — [`temporal_batch.py`](../src/models/tgn/temporal_batch.py).

---

## Stan implementacji

| Faza | Opis | Status | Kluczowe pliki |
|------|------|--------|----------------|
| 1 | Wspólny preprocessing | gotowe | [`src/preprocessing/`](../src/preprocessing/), [`config/preprocessing.yaml`](../config/preprocessing.yaml) |
| 2 | GRU4Rec baseline | gotowe | [`src/models/gru4rec/`](../src/models/gru4rec/), [`config/experiments/gru4rec_baseline.yaml`](../config/experiments/gru4rec_baseline.yaml) |
| 3 | TAGNN | gotowe | [`src/models/tagnn/`](../src/models/tagnn/), [`config/experiments/tagnn_baseline.yaml`](../config/experiments/tagnn_baseline.yaml) |
| 4 | TGN | gotowe | [`src/models/tgn/`](../src/models/tgn/), [`config/experiments/tgn_bce_baseline.yaml`](../config/experiments/tgn_bce_baseline.yaml) |
| 5 | Ewaluacja i raport | w toku | [`src/evaluation/`](../src/evaluation/), W&B |

Smoke testy CPU: `config/experiments/*_smoke.yaml`.

---

## Ewaluacja

**Metryki rankingowe:** Recall@20, MRR@20 — [`src/evaluation/metrics.py`](../src/evaluation/metrics.py).

**TGN i duży katalog:** sampled Recall@K (target + losowe negatywy) — [`src/evaluation/sampled.py`](../src/evaluation/sampled.py), test: [`tests/test_sampled_eval.py`](../tests/test_sampled_eval.py).

**Baseline popularnościowy:** POP@20 — [`src/evaluation/baselines.py`](../src/evaluation/baselines.py), test: [`tests/test_baselines.py`](../tests/test_baselines.py).

| Ustawienie | Opis |
|------------|------|
| **Transdukcyjne** | `test_internal` — sesje/itemy z wcześniejszego okresu w subsample |
| **Indukcyjne** | `challenge_test` — nowe sesje; TGN buduje pamięć od pierwszych kliknięć |

Logi eksperymentów: [W&B — adm-project-tgnn](https://wandb.ai/project-nn/adm-project-tgnn). Ustawienia: [`src/config/wandb_settings.py`](../src/config/wandb_settings.py).

---

## Literatura

- Rossi, E. et al. — *Temporal Graph Networks for Deep Learning on Dynamic Graphs* (TGN)
- Hidasi, B. et al. — *Session-based Recommendations with Recurrent Neural Networks* (GRU4Rec)
- Wu, S. et al. — *TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation* ([arXiv:2005.02844](https://arxiv.org/pdf/2005.02844))
- RecSys Challenge 2015 — dokumentacja Yoochoose

---

## Powiązane dokumenty

- [`data-and-eda.md`](data-and-eda.md) — decyzje preprocessingu z EDA
- [`preprocessing.md`](preprocessing.md) — szczegóły pipeline
- [`artifacts.md`](artifacts.md) — `meta.json` i wejście modeli
- [`README.md`](README.md) — spis całej dokumentacji
