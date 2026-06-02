# Temporal Graphs for Recommendations — plan podejścia (prezentacja „Approach”)

**Przedmiot:** Advanced Data Mining (ADM)  
**Temat:** Temporal Graph Networks w dynamicznych rekomendacjach sesyjnych  
**Zespół:** Wiktor Małysa (349105), Mikołaj Orzechowski (363917)  
**Cel dokumentu:** struktura od definicji problemu przez dane i metodologię po plan implementacji i ewaluacji (punkt 2 harmonogramu: *presentation of the approach*).

---

## 1. Definicja problemu

### 1.1 Kontekst biznesowy

Systemy e-commerce rejestrują strumień interakcji użytkowników (kliknięcia, oglądane produkty) w ramach krótkich **sesji**. Kluczowe zadanie to **dynamiczne rekomendacje sesyjne**: na podstawie dotychczasowych działań w sesji przewidzieć **następny produkt**, który użytkownik najprawdopodobniej wybierze — zanim sesja się zakończy.

### 1.2 Sformułowanie zadania: Next-Item Prediction

W naszym projekcie **Next-Item Prediction** oznacza:

- Mając historię interakcji w sesji do momentu \(t_{n-1}\), przewidzieć przedmiot kliknięty w \(t_n\).
- Nie wymaga ręcznych etykiet — uczymy się **samonadzorowanie** z logów kliknięć (każde kolejne kliknięcie jest etykietą dla poprzedniego kroku).

W podejściu z **Temporal Graph Networks (TGN)** zadanie to nie jest czysta sekwencja (jak w RNN), lecz **dynamiczne przewidywanie krawędzi (link prediction)** w grafie ewoluującym w czasie ciągłym.

### 1.3 Model grafowy: Continuous-Time Dynamic Graph (CTDG)

| Element | Definicja w projekcie |
|--------|------------------------|
| **Wierzchołki** | Graf dwudzielny: **sesje** (jako „użytkownicy”) ↔ **przedmioty** (items). Brak jawnego user ID w Yoochoose — każda sesja = osobny węzeł po stronie „użytkownika”. |
| **Zdarzenia** | Każde kliknięcie = interakcja \(e_{ij}(t)\) między sesją \(i\) a przedmiotem \(j\) w czasie \(t\). |
| **Cel** | Dla pary \((i, t)\) wskazać przedmiot \(j\) z najwyższym \(p(i,j \mid t)\) — **Future Edge / Link Prediction**. |

Opcjonalnie (rozszerzenie): plik zakupów (`yoochoose-buys.dat`) jako drugi typ zdarzenia o silniejszym sygnale decyzyjnym.

---

## 2. Dane

### 2.1 Zbiór: Yoochoose (RecSys Challenge 2015)

**Źródło:** [RecSys Challenge 2015](http://2015.recsyschallenge.com/) — logi sesji e-commerce.

| Plik | Zawartość | Pola |
|------|-----------|------|
| `yoochoose-clicks.dat` | Kliknięcia | Session ID, Timestamp, Item ID, Category |
| `yoochoose-buys.dat` | Zakupy | Session ID, Timestamp, Item ID, Price, Quantity |
| `yoochoose-test.dat` | Sesje testowe (tylko kliknięcia) | Jak clicks — bez odpowiadających buy w buys |

**Format czasu:** `YYYY-MM-DDThh:mm:ss.SSSZ` — rozdzielczość do **milisekund** (kluczowe dla \(\Delta t\) w TGN).

**Pole Category (cechy krawędzi):**

- `"S"` — promocja / oferta specjalna  
- `0` — brak wartości  
- `1`–`12` — kategoria produktu  
- inne liczby (8–10 cyfr) — kontekst marki (np. BOSCH)

### 2.2 Dlaczego Yoochoose pasuje do TGN

1. **Czas ciągły** — gęste sesje z krótkimi odstępami między kliknięciami; TGN wykorzystuje \(\Delta t\) między zdarzeniami.
2. **Multigraf / typy zdarzeń** — clicks vs buys jako różne typy krawędzi (opcjonalnie).
3. **Cechy krawędzi** — Category, Price, Quantity → wejście do message function i temporal attention.
4. **Indukcja / cold start** — nowe sesje w teście (`yoochoose-test.dat`); TGN inicjalizuje pamięć zerami i buduje reprezentację „w locie”.
5. **Skala** — dziesiątki milionów kliknięć → w Colab planujemy **chronologiczne podpróbkowanie** (np. 1/32–1/64 zbioru) z zachowaniem struktury sesji.

### 2.3 Alternatywy (odrzucone)

Rozważaliśmy m.in. **Steam** i **MovieLens** — odrzucamy na rzecz Yoochoose, bo:

- Steam: niska rozdzielczość czasowa (dni/miesiące), profil długoterminowy zamiast sesji w sekundach.
- TGN i *temporal attention* najlepiej wykorzystują **staleness** i mikro-dynamikę sesji — to dokładnie profil Yoochoose.

### 2.4 Przygotowanie danych (wspólny pipeline)

1. **Podpróbkowanie** — losowa/chronologiczna frakcja (np. 5–10% lub 1/64) pod RAM Colaba.  
2. **Podział chronologiczny** — np. 70% / 15% / 15% czasu (train / val / test); **bez** losowego `train_test_split`.  
3. **Mapowanie ID** — Session ID i Item ID → indeksy \(0 \ldots N-1\).  
4. **Timestamp** — konwersja do skalarów (sekundy/ms od początku zbioru).  
5. **Cechy krawędzi** — One-Hot / embedding dla `Category` (i opcjonalnie buy features).  
6. **Format wyjściowy** — wspólny format sesji + interakcji czasowych dla wszystkich modeli.

---

## 3. Podejście metodologiczne

### 3.1 Oś projektu: ewolucja modeli

Porównanie trzech poziomów abstrakcji na **tym samym** podzbiorze Yoochoose:

```text
GRU4Rec (sekwencja)  ⟷  SR-GNN (graf statyczny/sesyjny)  ⟷  TGN (graf dynamiczny, czas ciągły)
```

| Model | Idea | Czas |
|-------|------|------|
| **GRU4Rec** | Sesja = sekwencja ID produktów; GRU → następny item | Ukryty w kolejności kroków |
| **SR-GNN** | Sesja = mały graf skierowany (kliknięcia → krawędzie); GGNN + attention pooling | Dyskretna kolejność, bez ms |
| **TGN** | Globalny graf dwudzielny sesja–item; pamięć + wiadomości + temporal attention | Ciągły, \(\Delta t\) |

### 3.2 Architektura docelowa: TGN-attn

Pięć modułów TGN (zgodnie z literaturą Rossi et al.):

| Moduł | Wybór w projekcie | Uzasadnienie |
|-------|-------------------|--------------|
| **Memory** | Wektor stanu \(s_i(t)\) per węzeł; init = 0 | Długoterminowa kompresja historii |
| **Message function** | **Identity** (konkatenacja stanu, \(\Delta t\), cech krawędzi) | Proste, skuteczne na start |
| **Message aggregator** | **Last** (najświeższa wiadomość w batchu) | Szybkość w Colab vs mean |
| **Memory updater** | **GRU** | Dobry kompromis jakość / koszt |
| **Embedding** | **1-warstwowa Temporal Graph Attention** | Walka ze *memory staleness*; sąsiedzi dostarczają świeży kontekst gdy węzeł nieaktualizowany |

**Memory staleness:** pamięć sesji odświeża się tylko przy jej kliknięciach; TGN-attn agreguje stany sąsiadów (items, współwystępujące w trendach), więc embedding w \(t\) pozostaje aktualny.

**Optymalizacja:** 1 warstwa attention wystarczy — pamięć sąsiadów niesie informację z dalszej historii.

### 3.3 Dekoder i uczenie (TGN)

- **Dekoder:** MLP na parach embeddingów \(z_i(t), z_j(t)\) → \(p((i,j) \mid t)\).
- **Loss:** Binary Cross-Entropy z **negative sampling** (pozytywna krawędź = prawdziwe kliknięcie; negatywy = losowe itemy).
- **Batch size:** ok. **200** — zbyt duży batch → rozjazd pamięci wewnątrz paczki.

### 3.4 Trening bez wycieku informacji (information leakage)

Kolejność w pętli treningowej:

1. Batch \(N\) interakcji (chronologicznie).
2. Aktualizacja pamięci wiadomościami z **poprzednich** batchy.
3. Embeddingi + predykcja + strata (BCE) dla bieżącego batcha.
4. Zapis bieżących interakcji do **Raw Message Store** na kolejny krok.

Nigdy nie aktualizujemy pamięci zdarzeniem **przed** jego przewidzeniem w tym samym kroku.

### 3.5 Próbkowanie sąsiedztwa

- **Neighbor sampling** zamiast pełnej historii (ograniczenie RAM).
- **Most recent** (np. \(k=10\) najświeższych krawędzi) — w danych sekwencyjnych najnowsze interakcje niosą najwięcej sygnału.

### 3.6 Narzędzia

| Warstwa | Stack |
|---------|--------|
| Język | Python 3 |
| Deep learning | PyTorch |
| Grafy | PyTorch Geometric (`TGN`, `TemporalData`, `GatedGraphConv`, …) |
| Środowisko | Google Colab (GPU) |

---

## 4. Plan implementacji

### Faza 1 — Wspólny preprocessing

- [ ] Pobranie / podpróbkowanie Yoochoose  
- [ ] Chronologiczny split + mapowanie ID + cechy krawędzi  
- [ ] Eksport: sekwencje (GRU4Rec), grafy sesyjne (SR-GNN), strumień interakcji (TGN)

### Faza 2 — GRU4Rec (baseline)

- `nn.Embedding` → `nn.GRU` → warstwa liniowa na cały katalog itemów  
- Wejście: item w \(t\); target: item w \(t+1\)  
- Loss: **Cross-Entropy** (wieloklasowa)

### Faza 3 — SR-GNN

- Sesja \([A,B,C]\) → węzły \(\{A,B,C\}\), krawędzie \((A\to B), (B\to C)\)  
- `GatedGraphConv` + attention pooling reprezentacji sesji  
- Skalarne podobieństwo do embeddingów kandydatów

### Faza 4 — TGN

- `TemporalData`: `(src=session, dst=item, t, edge_attr)` posortowane po czasie  
- Konfiguracja: `TGNMemory`, Identity, Last, GRUCell, TemporalGraphAttention, \(k=10\) neighbors  
- Trening: negative sampling + `BCEWithLogitsLoss` + Raw Message Store

### Faza 5 — Ewaluacja i raport

- Te same metryki i ten sam zbiór testowy dla wszystkich modelów  
- Porównanie czasu epoki i liczby parametrów (trade-off)

---

## 5. Ewaluacja

### 5.1 Metryki (rekomendacje sesyjne)

- **Recall@20** / **Hit Rate@20**  
- **MRR@20** (Mean Reciprocal Rank)

Dla każdej interakcji: ranking wszystkich itemów (lub próbka + pełna ewaluacja na val) — pozycja ground-truth w TOP-20.

### 5.2 Ustawienia TGN (warto wspomnieć na prezentacji)

| Ustawienie | Opis |
|------------|------|
| **Transdukcyjne** | Sesje/itemy widziane w treningu (wcześniejszy okres czasu) |
| **Indukcyjne** | Nowe sesje z `yoochoose-test.dat` — TGN buduje pamięć od pierwszych kliknięć |

### 5.3 Baselines dodatkowe (opcjonalnie, jeśli starczy czasu)

- **TGAT** — temporal attention **bez** modułu pamięci (ablation vs TGN)  
- **JODIE / DyRep** — pamięć bez pełnej agregacji GNN

Priorytet: trójka **GRU4Rec → SR-GNN → TGN**.

---

## 6. Proponowana struktura slajdów (3–5 min)

| # | Slajd | Treść |
|---|--------|--------|
| 1 | Problem i zadanie | Sesyjne Next-Item Prediction jako CTDG / link prediction |
| 2 | Dane | Yoochoose: pola, czas ms, Category, skala, podpróbkowanie |
| 3 | Podejście główne | Schemat TGN-attn: Memory → Message → Last → GRU → Temporal Attention; staleness |
| 4 | Trening | Raw Message Store, chronologiczny split, negative sampling, batch ≈ 200 |
| 5 | Porównanie | GRU4Rec / SR-GNN / TGN + metryki Recall@20, MRR@20 |
| 6 | Plan prac | Pipeline → 3 modele → ewaluacja w Colab / PyG |

---

## 7. Ryzyka i mitigacje

| Ryzyko | Mitigacja |
|--------|-----------|
| RAM / czas w Colab | Podpróbkowanie, neighbor sampling \(k=10\), aggregator Last |
| Wyciek czasowy | Chronologiczny split + Raw Message Store |
| Niejednoznaczna Category | Embedding / bucketing (S, kategoria, marka, missing) |
| Zbyt ambitny scope | Najpierw clicks-only; buys jako rozszerzenie |

---

## 8. Literatura i odniesienia (skrót)

- Rossi, E. et al. — *Temporal Graph Networks for Deep Learning on Dynamic Graphs* (TGN, TGN-attn)  
- Hidasi, B. et al. — *Session-based Recommendations with Recurrent Neural Networks* (GRU4Rec)  
- Wu, S. et al. — *Session-based Recommendation with Graph Neural Networks* (SR-GNN)  
- RecSys Challenge 2015 — dokumentacja Yoochoose  

---

## 9. Następne kroki po prezentacji

1. Zaimplementować wspólny pipeline danych w repozytorium.  
2. Baseline GRU4Rec (szybka walidacja pipeline’u).  
3. SR-GNN na grafach sesyjnych.  
4. TGN z konfiguracją z sekcji 3.2.  
5. Tabela wyników + wykres trade-off jakość vs czas treningu.

---

*Dokument przygotowany na podstawie zgłoszenia tematu ADM oraz analizy podejścia (TGN, Yoochoose, baselines). Aktualizować wraz z postępem implementacji.*
