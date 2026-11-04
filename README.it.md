# RAG Demo · MongoDB + Voyage AI

*[English version](README.md)*

Una demo che risponde a domande su 100 documenti finanziari pubblici di 34 aziende americane
(soprattutto bilanci annuali 10-K, più alcune relazioni trimestrali e comunicati sui risultati),
**citando le pagine** da cui prende ogni informazione.

Mostra due modi di farlo:

- **Query**: la RAG classica. Una ricerca, una risposta. Puoi vedere passo per passo come MongoDB
  sceglie i passaggi: ricerca per parole, ricerca per significato, fusione delle due classifiche e
  riordino finale con un modello Voyage AI.
- **Ricerca agentica**: un agente che, se la prima ricerca non basta, cerca di nuovo in modo mirato,
  divide le domande con più parti, chiede quale anno intendi quando i valori cambiano da un anno
  all'altro e dichiara cosa non ha trovato.

Tutto avviene dentro **MongoDB Atlas**: documenti, ricerca testuale, ricerca vettoriale, fusione,
riordino, memoria delle conversazioni e file originali. È questo il messaggio della demo: MongoDB è
la piattaforma, non solo il database.

---

## Indice

1. [Cosa ti serve](#1-cosa-ti-serve)
2. [Installazione](#2-installazione)
3. [Configurazione: il file .env](#3-configurazione-il-file-env)
4. [Primo avvio](#4-primo-avvio)
5. [Preparare database e indici](#5-preparare-database-e-indici)
6. [Scaricare e caricare i 100 documenti](#6-scaricare-e-caricare-i-100-documenti)
7. [Fare le prime domande](#7-fare-le-prime-domande)
8. [Le pagine della demo](#8-le-pagine-della-demo)
9. [Funzioni facoltative](#9-funzioni-facoltative)
10. [Problemi frequenti](#10-problemi-frequenti)
11. [Come è fatta](#11-come-è-fatta)
12. [Dati e licenze](#12-dati-e-licenze)

---

## 1. Cosa ti serve

| Cosa | Perché | Obbligatorio |
|---|---|---|
| **Python 3.11 o 3.12** | fa girare l'applicazione | sì |
| **Un cluster MongoDB Atlas** | contiene documenti, indici e memoria | sì |
| **Un modello linguistico a tua scelta** (per esempio OpenAI, Anthropic o Gemini) | scrive le risposte | sì |
| **Una chiave Voyage AI** | serve solo per le modalità manual, context e multimodal | no |
| **Git** | per scaricare il progetto | consigliato |
| **Tesseract**, **LibreOffice** | OCR delle scansioni, pagine visive di Word e PowerPoint | no |

**Il cluster Atlas.** La demo usa funzioni recenti di MongoDB: `$rankFusion` (MongoDB 8.1 o
successivo), Vector Search con embedding automatici (`autoEmbed`) e lo stage di riordino `$rerank`.
È provata su MongoDB Atlas 9.0. Se nel tuo cluster manca qualcosa, la verifica dei componenti
dell'app (vedi il [passo 4](#4-primo-avvio)) te lo dice subito.

**Costi.** Atlas, Voyage AI e il modello linguistico possono avere costi d'uso. Caricare 100 bilanci
significa calcolare gli embedding di circa 33.000 passaggi.

**Python.** La demo è provata con Python 3.11 e 3.12. Con versioni molto recenti alcune librerie
(per esempio quella che legge i PDF) possono non avere ancora un pacchetto pronto, e l'installazione
fallisce. Con `uv` non devi preoccupartene: il comando del passo 2 scarica da solo Python 3.12.

> I comandi di questa guida sono per macOS e Linux. Su Windows sostituisci `.venv/bin/python` con
> `.venv\Scripts\python`.

---

## 2. Installazione

**1. Scarica il progetto** e entra nella cartella:

```bash
git clone https://github.com/Miccolomi/Rag_VoyageAI_MongoDB.git
cd Rag_VoyageAI_MongoDB
```

**2. Crea l'ambiente Python e installa le librerie.** Scegli uno dei due modi.

Con [uv](https://docs.astral.sh/uv/), più veloce:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Con gli strumenti standard di Python, se hai già Python 3.11 o 3.12 (controlla con
`python3 --version`; al posto di `python3` puoi usare `python3.12`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**3. Controlla che tutto sia a posto** con i test automatici. Non si collegano né ad Atlas né ai
modelli, quindi funzionano anche senza configurazione. Servono due librerie in più, elencate in
`requirements-dev.txt`:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt   # oppure: .venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Devono risultare tutti `passed`.

---

## 3. Configurazione: il file .env

Le impostazioni e le chiavi stanno in un file chiamato `.env`, nella cartella del progetto. Crealo
copiando il modello:

```bash
cp .env.example .env
```

Poi apri `.env` con un editor di testo. Il file è diviso in sezioni numerate e spiegate.
**Devi compilare solo le prime tre**; tutto ciò che inizia con `#` ha già un valore predefinito.

**Sezione 1: MongoDB Atlas.** La stringa di connessione al cluster.

1. In Atlas: *Database* → il tuo cluster → **Connect** → *Drivers* → copia la stringa
   `mongodb+srv://...`.
2. Sostituisci utente e password con quelli di un utente del database (*Database Access*).
3. In *Network Access* aggiungi l'indirizzo IP del tuo computer.

```
MONGODB_URI=mongodb+srv://utente:password@cluster.xxxxx.mongodb.net/
```

**Sezione 2: Voyage AI.** Se per ora userai solo la modalità **auto** (quella consigliata per
iniziare), puoi lasciarla vuota: embedding e riordino li esegue Atlas. Per le altre modalità serve
una chiave Voyage AI:

- creata dal tuo progetto Atlas: inizia con `al-` e funziona con l'indirizzo già impostato;
- creata sul sito di Voyage AI: inizia con `pa-` e richiede anche
  `VOYAGE_API_BASE=https://api.voyageai.com/v1`.

```
VOYAGE_API_KEY=al-...
```

**Sezione 3: il modello che scrive le risposte.** Lo scegli tu: indica il provider e la tua chiave
(per esempio `LLM_PROVIDER=gemini` e `GEMINI_API_KEY=...`). Provider supportati: `openai`, `anthropic`,
`gemini`, `ollama` (gira sul tuo computer, senza chiave).

> **Il file `.env` contiene le tue chiavi: non va mai condiviso né pubblicato.** È già escluso da
> git tramite `.gitignore`. Il file `.env.example`, invece, è il modello senza chiavi e fa parte del
> progetto.

---

## 4. Primo avvio

```bash
.venv/bin/python -m uvicorn server:app --port 8000
```

Apri il browser su **http://localhost:8000**.

In alto a destra c'è un indicatore di stato. Cliccalo: l'app controlla uno per uno la connessione ad
Atlas, i documenti, gli indici, la ricerca ibrida con il riordino, la cifratura e il modello delle
risposte, e ti dice cosa funziona e cosa no. Al primo avvio è normale che manchino documenti e
indici: li prepari nei due passi successivi.

Per fermare l'app premi `Ctrl+C` nel terminale.

---

## 5. Preparare database e indici

Va fatto una volta sola, su un cluster nuovo.

1. Apri la pagina **Admin**.
2. Nel riquadro *Indici di ricerca* premi **Verifica e prepara gli indici**.

L'app crea le collection, gli indici per i file originali (GridFS), l'indice testuale e un indice
vettoriale per ciascuna delle quattro modalità. Non cancella nulla di esistente. Atlas impiega
qualche minuto a rendere pronti gli indici: nella tabella lo stato passa a `READY`.

---

## 6. Scaricare e caricare i 100 documenti

I PDF **non sono inclusi** nel progetto: sono 180 MB e appartengono al dataset pubblico
FinanceBench. Li scarica uno script, sempre gli stessi 100 file, controllando che ognuno sia
identico all'originale.

**1. Scarica i PDF** (circa un minuto):

```bash
.venv/bin/python scripts/prepare_financebench.py
```

Alla fine trovi 100 PDF in `data/financebench_100/pdfs`. Se il download si interrompe, rilancia lo
stesso comando: riprende dai file mancanti.

**2. Caricali nella demo:**

1. Apri la pagina **Upload**.
2. Scegli la modalità **auto**.
3. Premi **una cartella** e seleziona `data/financebench_100/pdfs`. Seleziona solo `pdfs`, non la
   cartella sopra: lì ci sono anche file di servizio.
4. Premi **Avvia il caricamento**.

Per ogni documento l'app estrae il testo (tabelle comprese), lo divide in passaggi, lo salva in
MongoDB e conserva il PDF originale in GridFS. Con 100 bilanci ci vuole un po' di tempo: lascia la
pagina aperta. Puoi già fare domande sui documenti caricati mentre gli altri proseguono.

---

## 7. Fare le prime domande

Apri **Query** oppure **Ricerca agentica**. Sotto la casella della domanda trovi alcuni esempi
pronti; clicca uno e avvia la ricerca. Per esempio:

- *Quali sono state le spese in conto capitale di 3M nel 2018, in milioni di dollari?*
- *Quanto ha speso Adobe per ricerca e sviluppo nell'esercizio 2022?*
- *Quanti stabilimenti produttivi aveva 3M nel 2018 e quanto ha speso Adobe per ricerca e sviluppo
  nell'esercizio 2022?* (domanda in due parti: guarda come la divide l'agente)
- *Come si prepara la pizza napoletana?* (domanda fuori tema: l'app non inventa una risposta)

Puoi scrivere in italiano anche se i documenti sono in inglese. Le domande d'esempio sono pensate
per questi documenti: sulle altre aziende e sugli altri anni le risposte dipendono da cosa c'è nei
100 file.

Il selettore **Dove cercare** sotto la domanda decide in quali passaggi cercare: una modalità di
indicizzazione oppure **tutte** insieme.

---

## 8. Le pagine della demo

| Pagina | Cosa mostra |
|---|---|
| **MongoDB** | perché MongoDB è la piattaforma della demo: modello a documenti, ricerca, AI, sicurezza |
| **Come funziona** | i tre percorsi: domanda, ricerca agentica, caricamento dei documenti |
| **Upload** | il caricamento, con le quattro modalità di indicizzazione |
| **Query** | la RAG classica, passo per passo |
| **Ricerca agentica** | l'agente, con la cronologia di ogni decisione |
| **Admin** | stato, indici, documenti per modalità, azzeramento |
| **Domande di prova** | 150 domande FinanceBench con la risposta di riferimento, da copiare in Query o Ricerca agentica |

### Query

1. MongoDB cerca in parallelo **per parole** (`$search`, BM25) e **per significato**
   (`$vectorSearch`).
2. `$rankFusion` fonde le due classifiche.
3. `$rerank` riordina i candidati con un modello Voyage AI.
4. I passaggi migliori vanno al modello, che scrive **una sola risposta** con le citazioni.

Dopo la risposta, *Dentro la ricerca* mostra cosa ha trovato ogni motore, come sono state fuse le
classifiche, cosa ha cambiato il riordino e la pipeline MQL eseguita. Ogni numero della risposta
viene confrontato con il passaggio citato. Con il cursore **Equilibrio della fusione** puoi
spostare il peso tra parole chiave e significato e vedere subito come cambiano le fonti, senza
chiamare il modello.

Se il passaggio migliore è sotto la **soglia minima di pertinenza**, il modello non viene chiamato.

### Ricerca agentica

L'agente fa una prima ricerca e **valuta elemento per elemento** se le evidenze bastano. Se manca
qualcosa fa ricerche mirate, fino a 5 supplementari (il numero si cambia nelle impostazioni). Si
ferma prima se una ricerca non porta passaggi nuovi o se due ricerche diverse non trovano niente: in
quel caso non inventa una risposta e non mostra fonti che non c'entrano.

- **Domande in più parti**: una ricerca per ogni parte.
- **Valori diversi in anni diversi**, con una domanda senza anno: chiede quale anno intendi.
- **Conversazione**: ricorda le domande precedenti della sessione, così puoi chiedere «e nel 2019?».
- **Numeri calcolati**: se la risposta somma due valori delle fonti, lo dichiara e mostra il
  calcolo.

Ogni passo è salvato in MongoDB (`agentic_runs`) e si può rivedere dal menu *Domande della
sessione*.

### Le quattro modalità di indicizzazione

| Modalità | Chi calcola i vettori | Quando usarla |
|---|---|---|
| **auto** | MongoDB Atlas, con Voyage AI | per iniziare: nessun codice di embedding |
| **manual** | l'applicazione, con l'API di Voyage AI | per scegliere modello e chunking |
| **context** | l'applicazione, con `voyage-context` | ogni passaggio tiene conto del resto del documento |
| **multimodal** | l'applicazione, con `voyage-multimodal` | slide, grafici, scansioni: immagine e testo della pagina |

In **manual** la pagina Upload permette di scegliere la strategia di chunking (per struttura oppure
a finestra fissa), la dimensione e la sovrapposizione dei passaggi.

**Limite della modalità context:** funziona con documenti fino a circa 32.000 token. I bilanci 10-K
sono più lunghi, quindi per questo dataset usa auto o manual.

---

## 9. Funzioni facoltative

**Cifratura lato client (CSFLE).** La prima domanda di ogni sessione dell'agente può essere salvata
cifrata: MongoDB conserva solo il valore cifrato e solo l'applicazione con la chiave lo legge. Per
attivarla, una volta:

```bash
.venv/bin/python -m src.field_encryption setup
```

Il comando crea una chiave master in `~/.config/rag-demo/`, fuori dal progetto, e una chiave dati in
Atlas. Senza questo passo la demo funziona e salva la domanda in chiaro. Il pulsante
**🔒 Come MongoDB conserva i dati**, sotto la risposta dell'agente, mostra lo stesso documento visto
dall'applicazione e visto dal database.

**OCR e documenti Office.** Per le pagine scansionate installa Tesseract. Per le pagine visive di
Word e PowerPoint in modalità multimodal installa LibreOffice. Se non sono nel PATH, indica il
percorso nel `.env` (sezione 6).

**Altri modelli.** Nel `.env` (sezioni 3 e 4) puoi scegliere un altro provider, un altro modello, o
un modello diverso per le decisioni dell'agente. Le istruzioni date al modello sono in
`prompts/system_prompt.txt`.

---

## 10. Problemi frequenti

| Messaggio o sintomo | Cosa fare |
|---|---|
| L'installazione fallisce mentre «costruisce» un pacchetto (`pymupdf`, `cryptography`) | stai usando una versione di Python troppo recente: ricrea l'ambiente con `uv venv --python 3.12 .venv` |
| `KeyError: 'MONGODB_URI'` all'avvio | manca il file `.env` o la riga `MONGODB_URI` ([passo 3](#3-configurazione-il-file-env)) |
| Atlas non raggiungibile | controlla utente e password nella stringa e che il tuo IP sia in *Network Access* |
| «Non ci sono documenti caricati in modalità …» | carica i documenti in quella modalità, oppure scegline un'altra in *Dove cercare* |
| «Indici non interrogabili» | Admin → *Verifica e prepara gli indici*, poi attendi lo stato `READY` |
| «Il modello configurato non è disponibile» | controlla `LLM_PROVIDER` e la chiave corrispondente nel `.env` |
| «Modalità manual: serve VOYAGE_API_KEY» | aggiungi la chiave Voyage nel `.env`, oppure usa la modalità auto |
| Errore in modalità context su un documento lungo | è il limite di circa 32.000 token: usa auto o manual |
| Le pagine non mostrano le ultime modifiche | ricarica il browser con `Cmd+Shift+R` (`Ctrl+Shift+R` su Windows) |

---

## 11. Come è fatta

```
server.py               server web (Starlette): pagine e API
web/                    interfaccia: index.html, style.css, app.js (JavaScript senza framework)
prompts/                istruzioni al modello
scripts/                download dei 100 PDF FinanceBench
src/
  config.py             legge il file .env
  ingestion.py          lettura dei documenti, chunking, salvataggio in MongoDB e GridFS
  indexes.py            definizione e creazione degli indici
  retrieval.py          pipeline di ricerca: $search + $vectorSearch → $rankFusion → $rerank
  web_services.py       logica delle pagine (Query, Upload, Admin, verifica dei componenti)
  agent.py              motore della ricerca agentica
  agentic_search.py     pianificazione, valutazione delle evidenze, ricerche mirate
  citation_check.py     confronto dei numeri della risposta con le fonti
  agentic_memory.py     sessioni e cronologia in MongoDB
  field_encryption.py   cifratura lato client (CSFLE)
tests/                  test automatici (non usano Atlas né i modelli)
```

**Dati in MongoDB** (database `rag`):

| Collection | Contenuto |
|---|---|
| `pdf_chunks` | un documento per passaggio: testo, metadati, pagina, modalità, eventuale vettore |
| `pdf_files.files`, `pdf_files.chunks` | i file originali (GridFS) |
| `agentic_sessions` | le conversazioni con l'agente |
| `agentic_runs` | ogni ricerca agentica, passo per passo |
| `app_settings` | l'ultima modalità di ricerca scelta |
| `encryption.__keyVault` | la chiave dati della cifratura (solo con CSFLE) |

---

## 12. Dati e licenze

I documenti provengono da **FinanceBench** di Patronus AI
([repository](https://github.com/patronus-ai/financebench)): documenti finanziari pubblici di
società quotate negli Stati Uniti. Non sono inclusi in questo progetto; lo script li scarica da una versione precisa
di quel repository (elenco in `scripts/financebench_100.json`). Verifica le condizioni d'uso di
FinanceBench prima di usarli fuori da una demo.

Le 150 domande con risposta di riferimento della pagina **Domande di prova** (`web/domande.json`)
vengono da FinanceBench di Patronus AI, licenza CC BY-NC 4.0: uso non commerciale, citando la fonte.
