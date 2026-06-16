# Dane i decyzje z EDA

Ten dokument podsumowuje **decyzje projektowe** wynikające z eksploracji danych Yoochoose. Pełna analiza (wykresy, liczby, kaskadowość) jest w [`notebooks/eda_yoochose.ipynb`](../notebooks/eda_yoochose.ipynb) — sekcja *„Wnioski z EDA i rekomendacje preprocessingu”*. Walidację wyjścia pipeline można sprawdzić w [`notebooks/validate_preprocessing.ipynb`](../notebooks/validate_preprocessing.ipynb).

Implementacja decyzji: [`preprocessing.md`](preprocessing.md). Kontrakt na dysku: [`artifacts.md`](artifacts.md).

---

## Tabela decyzja → kod

| Decyzja EDA | Wartość w v1 | Pliki implementacji | Config / testy |
|-------------|--------------|---------------------|----------------|
| Pobranie raw | Kaggle → `data/raw/` | [`scripts/download_raw_data.py`](../scripts/download_raw_data.py), [`src/preprocessing/load.py`](../src/preprocessing/load.py) | — |
| Subsample dev | 1/32, pełne sesje | [`src/preprocessing/subsample.py`](../src/preprocessing/subsample.py) | `subsample_fraction` w [`config/preprocessing.yaml`](../config/preprocessing.yaml); [`tests/test_preprocessing.py`](../tests/test_preprocessing.py) |
| Split | 70/15/15, sesje w całości | [`src/preprocessing/split.py`](../src/preprocessing/split.py) | `split_ratios` w YAML; test split w [`tests/test_preprocessing.py`](../tests/test_preprocessing.py) |
| Dedup | duplikaty `(session_id, item_id, timestamp)` | [`src/preprocessing/clean.py`](../src/preprocessing/clean.py) | `remove_exact_duplicates: true` |
| Min. długość sesji | ≥ 2 kliknięcia | [`src/preprocessing/examples.py`](../src/preprocessing/examples.py) | `min_session_clicks: 2` |
| Powtórzenia A→A | zostają | — (brak merge) | `merge_consecutive_repeats: false` |
| Target val/test | `last_click` | [`src/preprocessing/examples.py`](../src/preprocessing/examples.py) | `eval_mode: last_click` |
| Vocab + UNK | tylko train clicks | [`src/preprocessing/vocab.py`](../src/preprocessing/vocab.py) | [`tests/test_preprocessing.py`](../tests/test_preprocessing.py) |
| Category buckets | 6 bucketów, mode/item | [`src/preprocessing/category.py`](../src/preprocessing/category.py) | test `classify_category` w [`tests/test_preprocessing.py`](../tests/test_preprocessing.py) |
| Timestamp | `t_sec` od `t_min` train | [`src/preprocessing/timestamps.py`](../src/preprocessing/timestamps.py) | — |
| Strumień TGN | events + examples | [`src/preprocessing/events.py`](../src/preprocessing/events.py), [`export.py`](../src/preprocessing/export.py) | — |
| Challenge test | pełny test.dat, streaming | [`src/preprocessing/pipeline.py`](../src/preprocessing/pipeline.py), [`examples.py`](../src/preprocessing/examples.py) | katalog `challenge_test/` |
| Buys (opcjonalnie) | osobny wariant TGN | [`src/preprocessing/events.py`](../src/preprocessing/events.py) (`enrich_buys`) | `--include-buys`, `include_buys` w YAML |
| Orkiestracja | end-to-end | [`src/preprocessing/pipeline.py`](../src/preprocessing/pipeline.py) | CLI: [`src/preprocessing/__main__.py`](../src/preprocessing/__main__.py) |
| Ładowanie config | dataclass | [`src/preprocessing/config.py`](../src/preprocessing/config.py) | [`src/config/yaml_loader.py`](../src/config/yaml_loader.py); [`tests/test_yaml_config.py`](../tests/test_yaml_config.py) |

Domyślne wartości w [`config/preprocessing.yaml`](../config/preprocessing.yaml):

```yaml
preprocessing:
  subsample_fraction: 0.03125  # 1/32
  split_ratios: [0.70, 0.15, 0.15]
  min_session_clicks: 2
  remove_exact_duplicates: true
  merge_consecutive_repeats: false
  include_buys: false
  eval_mode: last_click
```

---

## 1. Jakość i skala danych

| Obserwacja (pełny train) | Wniosek |
|--------------------------|---------|
| Brak NaN w clicks / buys / test | Brak imputacji — [`load.py`](../src/preprocessing/load.py) |
| Sesje zakupowe bez kliknięć: 0 | `buys` spójne z `clicks` |
| Konwersja (sesja z buy): ~5.5% | Zakupy jako osobny sygnał — opcjonalnie w TGN |
| `test.dat` nakłada się czasowo z trainem | To **inne sesje**, nie późniejszy okres — osobny split `challenge_test` |

Statystyki trafiają do `meta.json` przez [`write_meta()`](../src/preprocessing/export.py) w [`src/preprocessing/export.py`](../src/preprocessing/export.py).

---

## 2. Podpróbkowanie i split

| Ustawienie | Liczby z EDA (orientacyjnie) |
|------------|----------------------------|
| Pełny zbiór | ~33M kliknięć, ~9.25M sesji, 52 739 itemów |
| Subsample **1/32** | ~1.14M interakcji, ~289k sesji, 21 490 itemów |
| Split **70/15/15** | granice czasowe na kliknięciach; sesje przypisane w całości |

**Subsample** — najstarsze 1/32 sesji (według czasu pierwszego kliknięcia), sesja wchodzi w całości lub wcale:

```8:23:src/preprocessing/subsample.py
def subsample_sessions(
    clicks: pd.DataFrame,
    buys: pd.DataFrame,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Keep the oldest `fraction` of sessions (by first click time)."""
    if fraction >= 1.0:
        return clicks.copy(), buys.copy(), clicks["session_id"].unique()

    session_start = clicks.groupby("session_id")["timestamp"].min().sort_values()
    n_keep = max(1, int(len(session_start) * fraction))
    kept = session_start.iloc[:n_keep].index

    clicks_sub = clicks[clicks["session_id"].isin(kept)].copy()
    buys_sub = buys[buys["session_id"].isin(kept)].copy()
    return clicks_sub, buys_sub, kept
```

**Split** — percentyle na timestampach kliknięć, przypisanie sesji po czasie pierwszego kliknięcia: [`split.py`](../src/preprocessing/split.py) (`SPLIT_NAMES`: `train`, `val`, `test_internal`).

**Challenge test** — pełny `yoochoose-test.dat` bez subsample; eksport strumieniowy w [`pipeline.py`](../src/preprocessing/pipeline.py).

---

## 3. Next-item i filtrowanie sesji

| Metryka (pełny train) | Wniosek |
|-----------------------|---------|
| Sesje 1-klikowe: ~13.6% | Brak par treningowych — filtr `min_session_clicks=2` |
| Powtórzenie poprzedniego itemu: ~12% kliknięć | Silne self-loopy — w v1 **zostają** |
| Duplikaty `(session_id, item_id, timestamp)`: 69 | Usuwane przez `remove_exact_duplicates` |

Logika targetów train vs val/test w [`examples.py`](../src/preprocessing/examples.py):

```57:63:src/preprocessing/examples.py
        n_clicks = len(click_df)
        if is_train:
            target_positions = range(1, n_clicks)
        elif eval_mode == "last_click":
            target_positions = [n_clicks - 1]
        else:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")
```

- **Train:** sliding window — każda pozycja `1 … n_clicks-1` jako target.
- **Val / test_internal / challenge_test:** tylko **ostatni klik** (`eval_mode=last_click`).

---

## 4. Kaskadowość i popularność

- Top przejścia to głównie **self-loopy** (item → ten sam item).
- Top 1% itemów generuje ~35% kliknięć.
- Tylko ~3.6% targetów trafia w globalne TOP-20 — zadanie trudne poza surową popularnością.

**W treningu / ewaluacji:**

- Obowiązkowy baseline **POP@20** — [`src/evaluation/baselines.py`](../src/evaluation/baselines.py), test: [`tests/test_baselines.py`](../tests/test_baselines.py).
- Negatywy ważone popularnością — kontekst dla TGN i sampled eval: [`src/evaluation/sampled.py`](../src/evaluation/sampled.py).

Uwaga z prezentacji ([`first_presentation.md`](first_presentation.md)): kaskadowość w trainie może zawyżać częstość lokalnych przejść (np. `c1→c2` vs globalnie częstszy `c7`).

**Jak to zaprezentować (3 kroki):**

1. **Yoochoose (prawdziwe dane)** — [`notebooks/cascade_sliding_window_demo.ipynb`](../notebooks/cascade_sliding_window_demo.ipynb): subsample 1/32 jak w dev; przykład `c1→c2` vs `c1→global_top`, wykres Lorenza, tabela `lift`, wyszukiwanie itemów z sekcji 5.
2. **Pełna EDA** — sekcja „Przejścia sesyjne i kaskadowość” w [`eda_yoochose.ipynb`](../notebooks/eda_yoochose.ipynb): tabela `lift` na pełnym trainie.
3. **Wniosek:** sliding window waży sesje proporcjonalnie do `(długość − 1)`; model uczy **lokalnych** przejść, nie globalnej popularności katalogu.

---

## 5. Category (cecha krawędzi TGN)

Pole `category` jest niejednorodne (~67% itemów ma więcej niż jedną wartość). Pipeline mapuje na **6 bucketów** (`BUCKET_NAMES` w [`category.py`](../src/preprocessing/category.py)) i bierze **dominujący bucket per item** z train.

Encoding: `cat_bucket2idx.json` w `vocab/` — szczegóły w [`preprocessing.md`](preprocessing.md) §6.

---

## 6. Zakupy (buys)

- 100% zakupionych itemów było wcześniej klikniętych w sesji.
- Tylko ~35% zakupów = ostatni klik w sesji.

**v1:** GRU4Rec i TAGNN uczą się tylko z **kliknięć** (`clicks_only`). TGN może dostać wariant `with_buys` (`--include-buys`) — osobny katalog wyjściowy, bez plików sekwencyjnych. Zobacz `exports.sequence_models` w `meta.json` ([`export.py`](../src/preprocessing/export.py)).

---

## 7. Timestamp i graf TGN

| Metryka | Wartość (EDA) |
|---------|---------------|
| Mediana Δt w sesji | ~58.5 s |
| Węzły grafu dwudzielnego (pełny train) | ~9.3M |
| Krawędzie | ~33M |

**Preprocessing:**

1. `t_min` z train — [`timestamps.py`](../src/preprocessing/timestamps.py) (`compute_t_min`, `to_t_sec`).
2. Sortowanie stabilne: `session_id`, `timestamp`, `row_order` — [`events.py`](../src/preprocessing/events.py).
3. Eksport `events.parquet` pod `TemporalData` — [`src/models/tgn/dataset.py`](../src/models/tgn/dataset.py).

Dla TAGNN: graf sesji z krawędziami `(i → i+1)` budowany w [`src/models/tagnn/dataset.py`](../src/models/tagnn/dataset.py).

---

## 8. Test i cold start

| Metryka | Wartość |
|---------|---------|
| Itemy w challenge test | 42 155 |
| Cold start (poza train vocab) | 1 548 (~3.67%) |

- Słownik z **train clicks only** + token UNK — [`vocab.py`](../src/preprocessing/vocab.py).
- Indukcja TGN: nowe sesje w `challenge_test`, pamięć od zera — [`src/models/tgn/memory.py`](../src/models/tgn/memory.py).

---

## 9. Rozstrzygnięte decyzje otwarte (EDA §10)

| Temat | Propozycja w EDA | Decyzja w v1 (kod) |
|-------|------------------|---------------------|
| Merge repeatów `A,A→A` | test A/B | **Nie** — `merge_consecutive_repeats: false` |
| Target val/test | last vs sliding | **`last_click`** — `eval_mode` w YAML |
| Subsample finalny | 1/32 dev vs pełny | **1/32** na dev (`0.03125`) |
| Integracja buys | v1 bez vs multigraf | **Opcjonalnie** — flaga `include_buys`, osobny `output_dir` |

---

## Następny krok

Uruchomienie i szczegóły modułów preprocessingu → [`preprocessing.md`](preprocessing.md).
