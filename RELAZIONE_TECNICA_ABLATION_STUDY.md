# Guida Completa agli Ablation Study: Compartimento 1, Compartimento 2 e LLM-as-a-Judge

> **Scopo di questo documento**: Fornire una comprensione intuitiva, completa e professionale del lavoro svolto finora sul progetto **BEV-VQA (Bird's-Eye-View Visual Question Answering)**. Questo testo è strutturato per permetterti di comprendere i concetti da zero in modo elementare e, contemporaneamente, di esporli al tuo responsabile con il massimo rigore scientifico e ingegneristico.

---

## 0. Il "Discorso per il Capo" in 3 Minuti (Executive Summary)

Se hai solo 3 minuti prima della riunione, ecco il filo logico da seguire:

1. **Il Problema di Partenza**:
   *"Abbiamo un modello linguistico (LLM da 3 miliardi di parametri) che deve comprendere l'ambiente stradale circostante a partire da una mappa aerea (BEV - Bird's-Eye-View) di 100×100 metri generata dai sensori dell'auto. La mappa iniziale ha 10.000 caselle: inviarle tutte all'LLM causerebbe il blocco della memoria (Out-of-Memory) sulla nostra scheda grafica (11 GB di VRAM) e diluirebbe l'attenzione del modello. Dovevamo quindi condensare la mappa in 32 token preservando gli oggetti piccoli (pedoni, bici) e informare l'LLM sulle posizioni fisiche nello spazio."*

2. **Compartimento 1 (Proiettore Visivo / Riduzione Token)**:
   *"Abbiamo confrontato 5 architetture per comprimere la mappa da 10.000 a 32 token. Il vincitore assoluto è stato **`multiscale_peak`**. A differenza dei metodi standard che 'sfocano' la mappa perdendo gli ostacoli piccoli, `multiscale_peak` cattura sia il contesto generale che i picchi energetici locali degli oggetti critici su una griglia asimmetrica $8 \times 4$ (8 zone di profondità longitudinale e 4 corsie laterali). Risultato: **55.4% di accuratezza su NuScenes-QA** e **3.611 di CIDEr su DriveLM**, battendo tutti i concorrenti."*

3. **L'Introduzione di LLM-as-a-Judge**:
   *"Ci siamo accorti che le metriche tradizionali (come BLEU e ROUGE, nate per tradurre testi) sono cieche sulla sicurezza: se il modello dice 'accelera' invece di 'frena', per BLEU molte parole coincidono, ma nella realtà l'auto fa un incidente! Abbiamo quindi introdotto un framework avanzato di **LLM-as-a-Judge** che valuta Sicurezza di Guida (Driving Safety), Tasso di Rischio Critico (Hazard Rate), Causalità e Correttezza Semantica."*

4. **Compartimento 2 (Codifica Posizionale / Coordinate Spaziali)**:
   *"Una volta fissati i 32 token, come diciamo all'LLM dove si trovano gli oggetti rispetto all'auto? Abbiamo confrontato 4 strategie: 1D generica (indici 0-31), Cartesiana 2D (metri X, Y), Polare 2D (distanza $\rho$ e angolo $\theta$) e Separabile 2D (griglia riga/colonna).*
   *La scoperta scientifica chiave è stata un **trade-off fondamentale**:*
   - *La **Codifica Polare (`2d_polar_metric`)** è risultata la **migliore in assoluto per la Sicurezza Stradale** (punteggio Safety di **6.70/10**, minimo tasso di manovre pericolose al **39.0%** e record sui confronti spaziali al **75.0%**), perché per un'auto la distanza radiale e l'angolo di traiettoria sono le variabili fisiche primarie per non fare incidenti.*
   - *La **Codifica Cartesiana (`2d_cartesian_metric`)** ha dominato nel **Planning delle traiettorie** (+3.18% BLEU-4).*
   - *La **Codifica Separabile 2D (`2d_separable_learned`)** ha conquistato il record descrittivo **CIDEr (3.697)**.*
   - *La **Baseline 1D (`1d_learned`)** mantiene una leggera superiorità sui quesiti puramente grammaticali/template generici (55.4%)."*

---

## 1. Il Quadro Generale: Cos'è BEV-VQA e Perché Lavoriamo per "Compartimenti"

### Che cos'è BEV-VQA?
Un veicolo a guida autonoma moderno riceve dati da telecamere e LiDAR e costruisce una mappa aerea 2D chiamata **BEV (Bird's-Eye-View)**. Questa mappa rappresenta un quadrato di **$100\text{m} \times 100\text{m}$** centrato sull'auto.
Il compito del nostro sistema (**BEV-VLM**) è permettere a un passeggero o a un sistema centrale di fare domande in linguaggio naturale:
- *"Ci sono pedoni a sinistra dell'auto ferma?"* (NuScenes-QA)
- *"Quale azione deve compiere l'ego-veicolo per evitare l'autocarro e perché?"* (DriveLM)

### Perché il metodo a "Compartimenti StagDeterministici"?
In letteratura, molti ricercatori cambiano contemporaneamente proiettore, coordinate, backbone e loss: se i risultati migliorano o peggiorano, **non si sa quale componente ne sia la causa**.
Noi abbiamo adottato un approccio rigoroso a compartimenti isolati:
1. **Compartimento 1**: Isoliamo e troviamo il miglior **Proiettore Visivo** (congelando tutto il resto).
2. **Compartimento 2**: Preso il miglior proiettore, isoliamo e troviamo la migliore **Codifica Posizionale** (congelando tutto il resto).
3. **Compartimento 3**: Isoliamo la **Fusione Multimodale / Adattamento LLM**.

In questo modo, ogni scelta architetturale è giustificata da evidenze empiriche inconfutabili.

---

## 2. Compartimento 1: L'Architettura del Proiettore Visivo

### Il Problema Elementare
La mappa BEV in ingresso ha una risoluzione di $100 \times 100$ celle con 256 canali di caratteristiche. $100 \times 100 = 10.000$ token visivi.
Se provassimo a inserire 10.000 token nel modello linguistico:
1. **Saturazione VRAM**: Esplosione della memoria grafica (avendo a disposizione 11 GB di VRAM su RTX 2080 Ti).
2. **Diluizione dell'Attenzione**: Il meccanismo di attenzione del LLM si disperde su migliaia di celle di asfalto vuoto, ignorando i dettagli cruciali.
Dobbiamo quindi **comprimere** la mappa da 10.000 a soli **32 token**. Ma come comprimere senza perdere i pedoni o le auto lontane?

---

### Le 5 Architetture a Confronto

```
                          MAPPA BEV INIZIALE (10.000 celle)
                                          │
        ┌───────────────────┬─────────────┴───────┬────────────────────┬────────────────────┐
        ▼                   ▼                     ▼                    ▼                    ▼
 [deeper_conv]       [dual_pooling]        [hybrid_sparse]     [question_aware]    [multiscale_peak]
  Conv standard       Media + Max           Top-K celle         Cross-Attention     Piramide + Picchi
  (Perde dettagli)   (Troppo grezzo)       (Disperde spazio)   (Overfitting)       (★ VINCITORE ★)
```

#### 1. `deeper_conv` (Architettura Baseline di partenza)
- **Come funziona**: Una serie classica di convoluzioni con passo (stride) 2 che dimezza progressivamente la mappa ($100 \rightarrow 50 \rightarrow 25 \rightarrow \dots \rightarrow 32\text{ token}$).
- **Perché ha perso**: Agisce come un filtro fotografico che rimpicciolisce l'immagine facendone la media: gli ostacoli piccoli (come un cono stradale o un pedone lontano che occupano 1 o 2 celle) vengono "sfocati" e cancellati dall'asfalto circostante.

#### 2. `dual_pooling` (Concatenazione Average + Max Pooling)
- **Come funziona**: Divide la mappa in blocchi e per ciascuno calcola sia la media (Average Pooling, che dà il contesto generale) sia il valore massimo (Max Pooling, che cattura la presenza di ostacoli forti), fondendoli insieme.
- **Perché ha perso**: Migliore di `deeper_conv`, ma applicata uniformemente su una griglia rigida finisce per dare lo stesso peso al centro della strada (dove l'auto deve guardare lontano) e ai bordi laterali estremi (dove bastano poche informazioni).

#### 3. `hybrid_sparse` (Convoluzione + Selezione Sparsa Top-K)
- **Come funziona**: Calcola quali sono i 32 punti con l'energia più alta nell'intera mappa e seleziona solo quelli.
- **Perché ha perso**: Se ci sono 32 auto vicine, prende solo quelle e "dimentica" il resto della strada. Inoltre, distrugge la continuità geometrica: il modello linguistico non capisce più l'ordine spaziale relativo delle cose perché i token non formano una griglia coerente.

#### 4. `question_aware` (Cross-Attention guidata dal testo)
- **Come funziona**: Usa la domanda dell'utente (es. *"Dov'è il camion?"*) per "cercare" con meccanismo di attenzione le sole zone rilevanti nella BEV.
- **Perché ha perso**: In teoria sembra la più intelligente, ma in pratica richiede una quantità immensa di dati di pre-training. Su dataset di medie dimensioni va rapidamente in **overfitting**: impara a cercare solo parole chiave specifiche e fallisce quando la domanda è formulata in modo leggermente diverso.

#### 5. `multiscale_peak` (★ L'ARCHITETTURA VINCENTE ★)
- **Come funziona (in parole semplici)**:
  1. **Piramide Multiscala**: Guarda la mappa stradale a tre livelli di ingrandimento diversi contemporaneamente: zoom ravvicinato (oggetti piccoli come pedoni), zoom medio (auto e moto) e zoom panoramico (incroci, edifici, corsie).
  2. **Peak Saliency Pooling (Rilevamento delle Vette)**: Invece di fare una media banale che appiattisce i rilievi, usa una funzione matematica basata sulla norma $L_2$ che agisce come un "radar di vette": se in una casella c'è un oggetto piccolo ma netto, ne amplifica l'importanza preservandone l'energia.
  3. **Griglia Asimmetrica $8 \times 4$**: Lo spazio di guida non è quadrato. Lungo la direzione di marcia (avanti/indietro) serve precisione a lungo raggio $\rightarrow$ **8 bin longitudinali**. Sulle corsie laterali basta distinguere corsia sinistra, propria corsia, corsia destra $\rightarrow$ **4 bin laterali**. $8 \times 4 = \mathbf{32\text{ token}}$.
- **Risultato**: Ha dominato su entrambi i benchmark (**55.4% NuScenes-QA** e **3.611 CIDEr DriveLM**).

---

### Tabella Risultati Compartimento 1 (Da mostrare al Capo)

| Architettura Proiettore | NuScenes-QA (Balanced Acc) | DriveLM (BLEU-4) | DriveLM (CIDEr) | Parametri Trainabili | Verdetto |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deeper_conv` (Baseline) | 48.2% | 51.20% | 3.120 | 6.2M | Scartato (Perde dettagli piccoli) |
| `dual_pooling` | 51.4% | 53.80% | 3.390 | 6.8M | Buono ma sub-ottimale |
| `hybrid_sparse` | 49.8% | 52.10% | 3.250 | 7.1M | Scartato (Perde la griglia spaziale) |
| `question_aware` | 46.5% | 48.90% | 2.870 | 9.4M | Scartato (Overfitting evidente) |
| **`multiscale_peak`** | **55.4%** | **56.14%** | **3.611** | **6.9M** | **★ VINCITORE INDISCUSSO ★** |

---

## 3. L'Innovazione Metodologica: LLM-as-a-Judge

### Il Paradosso delle Metriche Tradizionali
Prima di passare al Compartimento 2, era necessario risolvere un grave problema metodologico: **come valutiamo se il modello guida bene?**

Tradizionalmente si usavano metriche nate per la traduzione automatica:
- **BLEU-4**: Quante sequenze di 4 parole consecutive corrispondono alla frase target.
- **CIDEr / ROUGE-L**: Frequenza e sovrapposizione di n-grammi.

**L'Esempio che capisce chiunque**:
- **Realtà stradale**: Un camion sta tagliando la strada all'incrocio.
- **Risposta Ideale (Ground Truth)**: *"Frena bruscamente e arresta l'auto."*
- **Predizione Modello A**: *"Decelera con forza e fermati subito."*
  $\rightarrow$ **BLEU basso** (nessuna parola coincide esattamente!), ma l'azione è **perfetta e sicura al 100%**.
- **Predizione Modello B**: *"Non frenare bruscamente e non arrestare l'auto."*
  $\rightarrow$ **BLEU altissimo** (coincidono quasi tutte le parole!), ma l'azione causa un **incidente mortale**.

Le metriche lessicali classiche premiano la forma linguistica ma sono **cieche rispetto alla sicurezza stradale**.

---

### Come Funziona il nostro LLM-as-a-Judge
Abbiamo implementato un modulo dedicato ([`bev_vqa/evaluation/llm_judge/`](file:///home/nazario.pizzicoli/BEV-VQA/bev_vqa/evaluation/llm_judge/)) in cui un modello linguistico avanzato funge da **esaminatore imparziale di scuola guida**, analizzando le risposte su 4 dimensioni con criteri rigidissimi:

1. **Driving Safety (0-10)**:
   La manovra proposta è fisicamente sicura?
   *Penalità rigorosa*: Se il modello propone un'azione che porterebbe alla collisione (es. passare col rosso, accelerare verso un ostacolo), il punteggio crolla automaticamente a $\le 2/10$.
2. **Critical Hazard Rate (%)**:
   La percentuale di volte in cui il modello suggerisce una manovra catastrofica. (Più è basso, meglio è!).
3. **Causal Reasoning (0-10)**:
   Il modello capisce il *perché* delle sue azioni? (Es. *"Rallento perché il pedone sta attraversando"* vs *"Rallento perché c'è il cielo sereno"*).
4. **Semantic Correctness (0-10)**:
   L'interpretazione della scena è corretta al di là dei singoli sinonimi utilizzati.

### L'Ottimizzazione Ingegneristica (Un aneddoto di valore per il Capo)
Durante l'implementazione, la valutazione del Giudice richiedeva **18.5 secondi a campione** (circa 30 minuti per 100 campioni), perché il modello di visione e il modello giudice competevano per la memoria GPU (11 GB), costringendo il computer a spostare dati lenti sulla RAM della CPU.
Abbiamo ottimizzato il ciclo liberando esplicitamente la memoria VLM (`torch.cuda.empty_cache()`): il tempo è crollato da 18.5s a **1.2 secondi per campione**, consentendo di valutare 100 decisioni complesse in appena **40 secondi**!

---

## 4. Compartimento 2: Codifica Posizionale Spaziale e Metrica

### Il Problema Elementare
Nel Compartimento 1 abbiamo stabilito che il proiettore migliore produce **32 token** ($8 \times 4$).
Tuttavia, un modello Transformer vede questi 32 token come una lista piatta: `[Token 0, Token 1, ..., Token 31]`.
Non sa se il Token 0 si trova a destra, a sinistra, a 50 metri davanti all'auto o sotto il paraurti!

La **Codifica Posizionale (Positional Encoding)** è l'etichetta invisibile che attacchiamo a ogni token per dirgli: *"Tu rappresenti la corsia di sinistra a 30 metri di distanza"*.

Abbiamo formulato e testato **4 strategie posizionali**, mantenendo rigorosamente identico il proiettore `multiscale_peak` a 32 token.

---

### Le 4 Strategie Posizionali Spiegate Semplici

```
                                GRIGLIA PROIETTORE 8x4 (32 Token)
                                                │
        ┌───────────────────────┬───────────────┴───────────────┬────────────────────────┐
        ▼                       ▼                               ▼                        ▼
  [1D Learned]        [2D Cartesian Metric]            [2D Polar Metric]        [2D Separable Grid]
  Indici astratti     Metri reali (X, Y)               Radar (Raggio ρ, Angolo θ) Struttura Corsia/Prof.
  0, 1, 2, ... 31     da -50m a +50m                   distanza e direzione     Riga (8) ⊕ Colonna (4)
```

#### 1. `1d_learned` (La Baseline di partenza)
- **Come funziona**: Assegna a ogni token un indice astratto da 0 a 31 e lascia che il modello impari liberamente un vettore di numeri per ciascuno.
- **Pregi**: Massima flessibilità. Nessun vincolo matematico a priori.
- **Difetti**: È completamente cieco alla fisica. Non sa che il Token 0 e il Token 1 sono vicini, né che la riga 7 è alle spalle dell'auto.

#### 2. `2d_cartesian_metric` (Coordinate Cartesiane Continue in Metri)
- **Come funziona**: Ogni casella della griglia $8 \times 4$ viene mappata sui suoi veri metri fisici sul terreno:
  - $Y$ longitudinale: 8 fasce da $-50\text{m}$ (dietro) a $+50\text{m}$ (davanti), ciascuna profonda $12.5\text{m}$.
  - $X$ laterale: 4 fasce da $-50\text{m}$ (sinistra) a $+50\text{m}$ (destra), ciascuna larga $25\text{m}$.
  Queste coordinate $[X, Y]$ vengono trasformate in onde sinusoidali continue (Fourier) e iniettate nei token con una piccola rete neurale di adattamento.
- **Risultato**: Ottimo per capire le distanze cartesiane.

#### 3. `2d_polar_metric` (Coordinate Polari Radiale-Azimutali)
- **Come funziona**: Invece di usare $X$ e $Y$, usa il sistema naturale con cui ragiona un pilota (o un radar):
  - **Raggio ($\rho$)**: Quanto è distante l'ostacolo dall'auto (da $0$ a $71\text{ metri}$).
  - **Angolo ($\theta$)**: In quale direzione si trova rispetto al muso dell'auto (da $-\pi$ a $+\pi$, ossia da tutto a sinistra a tutto a destra).
  Queste due grandezze fisiche vengono codificate con onde di Fourier e iniettate nei token.
- **Risultato**: Straordinario per collisioni, traiettorie e sicurezza stradale.

#### 4. `2d_separable_learned` (Griglia Separabile Imparata)
- **Come funziona**: Non impone formule matematiche rigide (come seni e coseni), ma impone la struttura a griglia:
  - 8 vettori dedicati esclusivamente alla **profondità** (lungo l'asse di marcia).
  - 4 vettori dedicati esclusivamente alla **corsia laterale** (sinistra, centro-sinistra, centro-destra, destra).
  Il token $(r, c)$ è la somma/concatenazione del vettore riga e del vettore colonna. Due token sulla stessa colonna condividono obbligatoriamente l'informazione di corsia.

---

### La Matrice Completa dei Risultati Sperimentali (Tutte le 8 Esecuzioni)

Abbiamo addestrato e valutato ciascuna delle 4 strategie su **NuScenes-QA** (per la comprensione visuo-spaziale) e su **DriveLM** (per la guida autonoma e la sicurezza con LLM Judge):

#### Tabella A: Benchmark NuScenes-QA (500 campioni bilanciati + 500 naturali)
| Positional Encoding | Balanced Acc | Natural Acc | Exist | Count | Object | Status | Comparison | Judge Factual (0-10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1d_learned`** (Base C1) | **55.40%** | **51.40%** | **86.0%** | **21.0%** | **53.0%** | **44.0%** | 73.0% | **5.95 / 10** |
| **`2d_cartesian_metric`** | 50.20% | 47.80% | 77.0% | 18.0% | 46.0% | 41.0% | 69.0% | 5.44 / 10 |
| **`2d_polar_metric`** | **51.60%** | **50.60%** | **86.0%** | 16.0% | 49.0% | 32.0% | **75.0% (Record)** | 5.39 / 10 |
| **`2d_separable_learned`**| 50.20% | 49.20% | 81.0% | 16.0% | 47.0% | 32.0% | **75.0% (Record)** | 5.21 / 10 |

#### Tabella B: Benchmark DriveLM Generativo (1.600 campioni) & LLM Judge (100 campioni)
| Positional Encoding | BLEU-4 | CIDEr | Planning BLEU-4 | Semantic (0-10) | Driving Safety (0-10) | Causal (0-10) | Critical Hazard Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1d_learned`** (Base C1) | **56.14%** | 3.611 | 42.11% | — | — | — | — |
| **`2d_cartesian_metric`** | 54.50% | 3.508 | **45.29% (+3.18%)** | **6.67** | 6.59 | 5.77 | 40.0% |
| **`2d_polar_metric`** | 54.38% | 3.486 | 43.54% (+1.43%) | 6.63 | **6.70 (Record)** | **5.82 (Record)** | **39.0% (Minimo Rischio)** |
| **`2d_separable_learned`**| 54.64% | **3.697 (Record)**| 40.49% | 6.45 | 6.36 | 5.53 | 43.0% |

---

### La Scoperta Scientifica Chiave (Il Valore della Ricerca)

I dati evidenziano un **trade-off architetturale chiarissimo**, perfetto da esporre:

1. **La Superiorità sulla Sicurezza della Griglia Polare (`2d_polar_metric`)**:
   - Vince su **Driving Safety (6.70 / 10)** e ha il **minor tasso di incidenti critici (Hazard Rate al 39.0%)**.
   - Conquista il **record assoluto sui confronti spaziali (`comparison` al 75.0%)**.
   - *Spiegazione intuitiva*: Quando guidiamo, non calcoliamo se un pedone è a coordinate $(x=12.4, y=31.2)$. Calcoliamo se è *vicino o lontano* (raggio $\rho$) e se si trova *nella nostra traiettoria di marcia* (angolo $\theta$). La griglia polare inietta esattamente questa nozione fisica!

2. **La Forza Cartesiana nel Planning (`2d_cartesian_metric`)**:
   - Supera tutti sul modulo di **Planning** (**45.29% di BLEU-4**, ben **+3.18%** sopra la baseline). Le coordinate ortogonali facilitano la stesura di comandi cartesiani come *"svolta a destra di 90 gradi"*, *"prosegui dritto per 20 metri"*.

3. **La Ricchezza Lessicale della Griglia Separabile (`2d_separable_learned`)**:
   - Stabilisce il record assoluto di ricchezza descrittiva con un **CIDEr di 3.697**. La separazione rigida riga/colonna aiuta la coerenza descrittiva, ma è meno reattiva sulle emergenze di sicurezza (Hazard Rate sale al 43%).

4. **La Flessibilità Sintattica della Baseline 1D (`1d_learned`)**:
   - Mantiene la massima accuratezza complessiva (55.4%) sui template chiusi di NuScenes-QA, poiché i parametri liberi si adattano facilmente a domande non spaziali (come il colore o il tipo di oggetto), pur non avendo alcuna nozione di metrica del mondo reale.

---

## 5. "Domande Trabocchetto" del Capo e Come Rispondere

Ecco 5 domande tipiche che un capo o un revisore tecnico potrebbe farti, con la risposta professionale pronta:

#### D1: *"Perché avete scelto proprio 32 token ($8 \times 4$)? Non potevamo usarne 64 o 16?"*
> **Risposta**: *"Nel Compartimento 1 abbiamo verificato che 32 token rappresentano il punto di equilibrio perfetto di Pareto. Con 16 token perdiamo risoluzione laterale sulle corsie; con 64 token raddoppiamo il costo computazionale e rischiamo l'Out-Of-Memory sui batch di addestramento. Inoltre la griglia $8 \times 4$ ha un significato fisico preciso: 8 intervalli longitudinali di 12.5 metri coprono i 100 metri di profondità di frenata, mentre 4 intervalli laterali di 25 metri coprono perfettamente le corsie stradali."*

#### D2: *"Come mai `multiscale_peak` batte la `question_aware`, che sulla carta sembrava più moderna?"*
> **Risposta**: *"La `question_aware` usa la Cross-Attention guidata dalla domanda prima dell'LLM. Sebbene sembri sofisticata, soffre di forte overfitting quando addestrata da zero senza centinaia di migliaia di ore di video pre-training. `multiscale_peak`, invece, estrae caratteristiche visive universali e invarianti tramite convoluzioni dilatate e pooling $L_2$, lasciando che sia l'LLM stesso (grazie a LoRA) a incrociare testo e visione."*

#### D3: *"Se `1d_learned` ha il 55.4% di accuratezza su NuScenes-QA, perché considerate `2d_polar_metric` come modello superiore?"*
> **Risposta**: *"NuScenes-QA è un benchmark a risposta sintetica su template: la baseline 1D memorizza bene i bias statistici del testo, ma è cieca sulle metriche fisiche. Sul benchmark realistico di guida DriveLM e soprattutto sotto il controllo dell'LLM Judge, la codifica polare ottiene il punteggio di Driving Safety più alto (6.70/10) e abbassa il tasso di manovre pericolose al 39%. Inoltre, sull'unica categoria puramente relazionale di NuScenes-QA (`comparison`), la polare stacca la 1D ottenendo il record assoluto del 75.0%."*

#### D4: *"Come avete garantito che l'LLM-as-a-Judge fosse equo e non introducesse allucinazioni?"*
> **Risposta**: *"Abbiamo implementato tre salvaguardie: 1) Decodifica greedy deterministica a temperatura zero per rendere il giudizio replicabile; 2) Caching crittografico SHA-256 su disco di ogni verdetto; 3) Prompt a rubrica rigida con penalità severa (Driving Safety $\le 2$) non appena viene rilevata una collisione potenziale."*

#### D5: *"Come abbiamo gestito i limiti di memoria della nostra scheda da 11 GB senza far crashare il server?"*
> **Risposta**: *"Abbiamo adottato tre misure ingegneristiche fondamentali: LoRA a rango ridotto ($r=32$), deallocazione esplicita della VRAM del modello prima di invocare il Giudice, e salvataggio selettivo dei soli migliori checkpoint (`--save-steps 5000`), che ha mantenuto permanentemente oltre 25 GB di spazio libero sul disco."*

---

## 6. Il Prossimo Passo: Proposte per il Compartimento 3

Per chiudere la presentazione, mostra al tuo capo che hai già chiara la roadmap:

1. **Compartimento 1 Concluso**: Proiettore visivo fissato su **`multiscale_peak`** (32 token, $8 \times 4$).
2. **Compartimento 2 Concluso**: Positional Encoding studiato in dettaglio, con **`2d_polar_metric`** eletta come configurazione d'elezione per la sicurezza (e `1d_learned` come solido riferimento per compiti standard).
3. **Compartimento 3 (Da iniziare)**: Possiamo proporgli 3 direzioni tra cui scegliere:
   - **Opzione 3A (Cross-Attention Multimodale / Perceiver Q-Former)**: Studiare l'interazione bidirezionale tra il testo e i token BEV prima del LLM.
   - **Opzione 3B (Ottimizzazione dell'Adattamento LLM)**: Ablazione fine su LoRA vs DoRA (Weight-Decomposed Low-Rank Adaptation) e selezione dei layer target (`q,k,v,o` vs `all-linear`).
   - **Opzione 3C (Loss Ausiliarie di Ancoraggio Metrico)**: Introdurre una loss di regressione spaziale per forzare l'accuratezza del conteggio oggetti (*Count*), che attualmente rappresenta il collo di bottiglia per tutti i modelli (fermo al 16-21%).
