# RAG con MongoDB Atlas e Voyage AI

Applicazione dimostrativa end-to-end per caricare documenti, recuperarne i
contenuti con ricerca ibrida e generare risposte fondate sulle fonti archiviate
in MongoDB Atlas.

## Cosa fa

- Acquisisce documenti PDF e Office ed estrae testo e metadati utili.
- Suddivide i documenti in chunk conservando pagina e tipo di contenuto.
- Supporta embedding gestiti da MongoDB Atlas, embedding applicativi e
  contextualized embedding con Voyage AI.
- Esegue ricerca keyword, semantica o ibrida.
- Combina le classifiche con `$rankFusion`.
- Applica i filtri metadata direttamente alle pipeline di retrieval.
- Riordina i candidati con i modelli di reranking Voyage AI.
- Genera risposte grounded usando esclusivamente i passaggi recuperati.
- Mantiene separate le citazioni nel testo e i documenti originali scaricabili.
- Mostra il contributo delle diverse fasi della pipeline nella pagina Query.

## Architettura

La stessa piattaforma dati gestisce documenti, metadati, ricerca testuale,
ricerca vettoriale, filtri e fusione delle classifiche. L'applicazione usa
Voyage AI per gli embedding e il reranking e un provider LLM configurabile per
comporre la risposta finale sui passaggi selezionati.

La pipeline principale e:

1. estrazione e normalizzazione del documento;
2. chunking e arricchimento dei metadati;
3. generazione o gestione automatica degli embedding;
4. retrieval keyword e semantico;
5. fusione e reranking;
6. costruzione del contesto;
7. risposta grounded e accesso alle fonti originali.

## Avvio locale

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Compilare `.env` con la connessione MongoDB Atlas e le chiavi dei provider
utilizzati. Il file `.env`, gli ambienti virtuali e i documenti caricati
localmente sono esclusi dal repository.

## Struttura del progetto

- `app.py`: interfaccia Streamlit e orchestrazione della demo.
- `src/`: ingestion, embedding, retrieval, reranking e configurazione.
- `schemas/`: definizioni degli indici Atlas Search e Vector Search.
- `prompts/`: prompt di sistema per la generazione grounded.
- `tests/`: test della pipeline.
- `scripts/`: strumenti diagnostici per configurazione e modelli.
