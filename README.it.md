# RAG con MongoDB Atlas e Voyage AI

[English](README.md)

**Stato documentato: 29 ottobre 2026.** Piano e decisioni correnti: [DEMO_PLAN.md](DEMO_PLAN.md).

PoC dimostrativo per caricare documenti, confrontare strategie di indicizzazione,
recuperare passaggi pertinenti e generare risposte con riferimenti alle fonti.
Il corpus può contenere PDF, Word e PowerPoint, anche con immagini, grafici e
slide. Sono disponibili inoltre i formati testuali supportati dal parser.

Questa guida descrive il codice presente nel repository e la struttura della
pagina How it Works. I limiti e le attività ancora aperte sono indicati
esplicitamente; non rappresenta un collaudo del cluster.

## Obiettivo della demo

**Obiettivo concordato e piano di lavoro:** [DEMO_PLAN.md](DEMO_PLAN.md).
Perimetro attuale: 100 documenti simili, caricamento di una cartella e 5 domande
con risposte e fonti verificate, di cui almeno 2 a trabocchetto. Valutazione
manuale con una scheda; confronti tra strategie spiegati a voce. Nessun nuovo
benchmark automatico o sviluppo multimodale previsto in questa fase.

Mostrare **perché** servono preparazione dei documenti, embedding, ricerca
lessicale, ricerca semantica, fusione e reranking. Ogni scelta nell'interfaccia
deve spiegare cosa farà, quale risultato produce e dove verificarlo.

Le risposte sono accompagnate da citazioni e accesso agli originali. Il prompt
istruisce il modello a usare i passaggi recuperati: la presenza delle citazioni
non garantisce da sola la correttezza della risposta.

## File originali, GridFS e object storage

Gli originali sono i file completi caricati, conservati nel loro formato binario
per poterli scaricare e aprire con l'applicazione appropriata.
[GridFS](https://www.mongodb.com/docs/manual/core/gridfs/) è un meccanismo di
MongoDB per archiviare e recuperare binari di qualsiasi formato: PDF, Word,
Excel, PowerPoint, immagini e altri. Divide il file in blocchi binari e ne
conserva i metadati; il driver ricompone il file quando viene richiesto.
I blocchi GridFS hanno una funzione di archiviazione distinta dai chunk estratti
per la ricerca. Archiviare un formato non implica che i parser della demo siano
in grado di estrarne il contenuto.

GridFS è la scelta implementata nella demo per mantenere anche i binari in
MongoDB. In alternativa, i file possono risiedere in un **object storage**, un
servizio dedicato all'archiviazione dei file, lasciando in MongoDB contenuti
ricercabili, metadati e riferimenti ai file. Questa alternativa richiederebbe
un'integrazione dedicata nell'applicazione.

## Due flussi distinti

### Inserimento: quando si carica un documento

1. Estrarre testo, tabelle e metadati; applicare OCR alle pagine che lo richiedono.
2. Suddividere il contenuto in chunk, conservando riferimenti a pagina e fonte.
3. Nel percorso multimodale, renderizzare anche pagine e slide come immagini.
4. Generare gli embedding via Voyage nell'app oppure delegarli ad Atlas (`auto`).
5. Salvare chunk e metadati nella collection applicativa e l'originale in GridFS.
   Atlas aggiorna gli indici in modo asincrono.

### Consultazione: ogni volta che si fa una domanda

1. Applicare i filtri selezionati e cercare con Atlas Search (testo) e Atlas
   Vector Search (significato). La pagina Query include tutte le
   modalità presenti: ciascun ramo vettoriale usa il proprio modello e indice.
2. Combinare la classifica testuale e le classifiche vettoriali con `$rankFusion`.
3. Riordinare i candidati con Voyage tramite `$rerank` dopo `$rankFusion`, nella
   stessa pipeline Atlas (percorso predefinito `RERANK_BACKEND=native`).
4. Controllare il miglior punteggio rerank rispetto alla soglia configurata.
5. Inviare i passaggi selezionati al LLM e mostrare risposta, citazioni e originali.

La pagina Query esegue anche ricerche diagnostiche separate per spiegare il
contributo dei canali. L'intera interazione non equivale a una sola chiamata al
database. Il reranking nativo usa l’abilitazione Atlas e non la chiave Voyage
dell’app. Con `RERANK_BACKEND=voyage_api` viene usata la precedente chiamata API;
solo in questo percorso, senza chiave, si usano i primi risultati della fusione.
Le modalità con embedding generati dall'app
richiedono comunque la chiave.

## Le quattro modalità di embedding

I nomi dei modelli sotto sono i valori predefiniti del codice; la configurazione
può modificarli. Le dimensioni devono corrispondere al modello e all'indice.

| Modalità | Cosa rappresenta | Chi genera gli embedding | Dove sono i vettori | Cosa confrontare nella demo |
|---|---|---|---|---|
| `auto` | Testo dei chunk, con `voyage-4-large` | Atlas tramite autoEmbed | Nell'indice Atlas | Gestione degli embedding delegata ad Atlas |
| `manual` | Testo dei chunk, con `voyage-4-large` | L'app tramite Voyage | Campo `embedding` | Controllo applicativo su generazione e memorizzazione |
| `context` | Chunk interpretati insieme agli altri chunk del documento, con `voyage-context-4` | L'app tramite Voyage | Campo `embedding_context` | Recupero di passaggi che dipendono dal contesto circostante |
| `multimodal` | Testo e immagini di pagine/slide, con `voyage-multimodal-3.5` | L'app tramite Voyage | Campo `embedding_multimodal` | Recupero di contenuti riconoscibili anche dalla componente visiva |

In **Upload → Inventario documenti e embedding**, il pulsante **Mostra o aggiorna
inventario** legge i metadati già salvati nei chunk e li raggruppa per documento,
modalità, modello e strategia. Valori mancanti sono mostrati come non disponibili.
Per `auto` mostra separatamente anche l'ultima definizione disponibile dell'indice
Atlas: il modello registrato al caricamento può differire da quello configurato
successivamente nell'indice. La tabella è uno snapshot aggiornabile e non prova
che ogni chunk sia già stato indicizzato.

Verifica documentazione ufficiale del 29 ottobre 2026: la lista dei modelli per
[Atlas Automated Embedding](https://www.mongodb.com/docs/vector-search/crud-embeddings/automated-embedding/models/)
non include `voyage-context-4`; questo modello si usa tramite
[API di embedding contestualizzati](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
e un indice vettoriale su vettori forniti dall'app. Può coesistere nella demo
con il percorso `auto`, in campi e indici separati; i vettori del documento e
della query devono usare il modello appropriato. Non garantisce da solo una
risposta corretta o citazioni valide: va confrontato sui documenti reali.

`context` conserva i chunk locali e i riferimenti di pagina: non aggiunge un
riassunto al testo. L'implementazione raggruppa i chunk del documento in un solo
input e rifiuta documenti oltre il limite stimato di 32K token del percorso
pre-chunked. Non implementa una suddivisione automatica in più gruppi.

In Upload ci sono due scelte:

- **Indicizzazione del testo, delle tabelle e del testo estratto con OCR**:
  usa la modalità testuale selezionata in Admin (`auto`, `manual` o `context`).
- **Indicizzazione del testo e dei contenuti visivi (immagini, grafici, diagrammi
  e slide)**: usa `multimodal` e aggiunge le immagini delle pagine supportate.

Lo stesso file può essere acquisito in modalità diverse. Le politiche `skip` e
`replace` si applicano alla modalità attiva. Cambiare modalità non converte i
chunk già presenti: occorre caricare il documento anche nel percorso scelto.
La pagina Query include sempre tutte le modalità presenti. Ogni ramo
vettoriale resta nel proprio spazio di embedding; il ramo testuale usa lo
stesso perimetro complessivo. Gli eventuali filtri sui documenti si applicano
a tutti i percorsi; non è richiesta una scelta della modalità di ricerca.

## Parser, Voyage e LibreOffice: compiti diversi

- **PyMuPDF** legge e renderizza i PDF; **pdfplumber** estrae le tabelle.
- **Tesseract** fornisce OCR quando necessario, se installato e configurato.
- **MarkItDown** estrae testo e struttura dai documenti Office e dagli altri
  formati supportati.
- **LibreOffice (`soffice`)** converte Word/PowerPoint in PDF per mantenere una
  rappresentazione di pagine e slide. È necessario per il percorso visivo
  Office. Nel percorso testuale viene tentata la conversione, con ripiego sul
  testo MarkItDown se non riesce. I PDF non richiedono LibreOffice.
- **Voyage multimodal** trasforma testo e immagini preparati dall'app in vettori:
  non sostituisce il parser né la conversione dei file Office.

### Limite attuale del percorso multimodale

Gli embedding usano le immagini, ma **reranking e generazione ricevono oggi
soltanto testo e riferimenti**, come mostrano `src/rerank.py` e
`src/generation.py`. Una pagina trovata grazie al grafico può quindi perdere
posizioni nel reranking se il suo testo è scarso; il LLM non vede direttamente
quel grafico. La demo può mostrare il recupero visivo, ma non va presentata come
una lettura completa delle immagini fino alla risposta finale.

## Avvio e preparazione dell'ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Solo al primo setup, se .env non esiste:
cp .env.example .env
streamlit run app.py
```

Prima dell'avvio, compilare `.env` con:

- `MONGODB_URI`, database, collection e bucket GridFS;
- `VOYAGE_API_KEY` e relativo `VOYAGE_API_BASE`;
- `LLM_PROVIDER` e credenziali o indirizzo del provider scelto;
- `OFFICE_CONVERTER` se `soffice` non è nel PATH;
- `TESSERACT_CMD` se Tesseract non è nel PATH, con dati lingua necessari all'OCR.

Il cluster deve supportare le funzionalità selezionate, incluse `$rankFusion`
e autoEmbed se si usa `auto`. Verificare accesso di rete, permessi per creare
collection/indici e requisiti Atlas della funzionalità. Il setup applicativo
non configura il cluster, i servizi esterni o i programmi locali.

1. Aprire **Admin → Prepara database e indici**.
2. Leggere il rendiconto: connessione, collection, indice ordinario dei documenti, indici GridFS e richieste di
   creazione/aggiornamento degli indici Atlas per tutte e quattro le modalità, con avanzamento per indice.
3. Aprire **Dettaglio collection e indici presenti su Atlas** e aggiornare lo
   stato. Prima della ricerca, attendere gli indici della modalità e quello
   testuale pronti. Una richiesta accettata non significa indicizzazione conclusa.
4. Nell'espander **Verifica definizioni e mostra il codice MongoDB di creazione / aggiornamento**,
   ispezionare i comandi generati dalle stesse definizioni usate dal setup.
5. Caricare un documento in **Upload**, leggere il riepilogo e passare a **Query**.

Con i nomi predefiniti vengono preparati:

| Risorsa | Funzione |
|---|---|
| `pdf_chunks` | Chunk, metadati e vettori generati dall'app |
| `document_id_1_embedding_mode_1` | Indice ordinario su `pdf_chunks` per il controllo duplicati e le ricerche per identificativo documento |
| `pdf_files.files`, `pdf_files.chunks` | Originali archiviati tramite GridFS e relativi indici ordinari |
| `pdf_chunks_text_idx` | Atlas Search sul testo, condiviso tra modalità |
| `pdf_chunks_vector_idx` | Indice vettoriale `auto` |
| `pdf_chunks_manual_idx` | Indice vettoriale `manual` |
| `pdf_chunks_context_idx` | Indice vettoriale `context` |
| `pdf_chunks_multimodal_idx` | Indice vettoriale `multimodal` |

Il setup crea le risorse mancanti e confronta le definizioni degli indici già
presenti. Se lo schema differisce, invia `updateSearchIndex`; se coincide,
non esegue scritture. L'indice testuale condiviso viene gestito una sola volta.
Per l'indice ordinario dei documenti riutilizza un indice equivalente se già presente.
Anche **Admin → Verifica / Crea indici** verifica questo indice ordinario, oltre
agli indici Atlas della modalità selezionata. L'opzione **Drop & recreate**
agisce sugli indici Atlas gestiti dal pulsante, non sull'indice ordinario.
Il rendiconto conserva nome, differenze campo per campo, definizione precedente,
definizione richiesta, comando, errori e stato Atlas. Un errore su un indice
non interrompe la verifica degli altri. Un tipo di indice incompatibile viene
segnalato; la ricreazione con drop resta una scelta esplicita.

Le operazioni Atlas sono asincrone: durante un aggiornamento `queryable=true`
può riferirsi ancora alla versione precedente. Attendere `READY` per considerare
completata la nuova definizione. Cambiare modello o dimensioni dei vettori
calcolati dall'app può richiedere una nuova ingestion: aggiornare l'indice non
ricalcola quei vettori.

Il setup corregge inoltre i riferimenti sintetici alla pagina 1 dei vecchi chunk
`office_structure` e del fallback con `creator=MarkItDown`, rimuovendo soltanto
i metadati di pagina non attendibili. Non modifica testo o embedding.
Per i nuovi upload gli estratti a livello documento non hanno numeri di pagina;
i passaggi ottenuti dal PDF renderizzato mantengono pagina/slide. Le citazioni
senza pagina rimandano al solo documento.

Lo script `python -m src.init_db` prepara soltanto collection e indici GridFS;
il pulsante Admin include anche la correzione dei riferimenti e gli indici Atlas.
`.env`, ambienti virtuali e documenti caricati localmente sono esclusi dal repository.

## Caricare i 100 PDF FinanceBench

Il pacchetto locale si prepara con `python scripts/prepare_financebench.py`:
scarica 100 PDF dal repository pubblico, controlla l’integrità rispetto alla
revisione GitHub salvata nel manifest e riprende i download mancanti. Non chiama
Atlas o Voyage. PDF in `data/financebench_100/pdfs`; manifest e annotazioni di
riferimento rimangono nella cartella superiore e non vanno indicizzati.

Nella pagina **Upload**:

1. Scegli **Indicizzazione del testo, delle tabelle e del testo estratto con OCR**.
2. Mantieni **GridFS** e seleziona **Una cartella**.
3. Scegli `data/financebench_100/pdfs` e controlla il conteggio di 100 documenti.
4. Lascia vuoti Titolo, Autore e Anno; gli override si applicano a tutti i file.
   Facoltativamente usa il tag comune `financebench`.
5. Scegli **Salta upload duplicati**, poi **Avvia ingestion**.

Serve Streamlit 1.65 o successivo, come indicato nelle dipendenze. L’app elabora
un documento alla volta e mostra progresso, file corrente e risultati per file.
Gli errori non interrompono gli altri documenti. Il riepilogo distingue inseriti,
duplicati saltati ed errori ed è scaricabile come JSON. Un singolo documento
mantiene il riepilogo dettagliato precedente. I file temporanei vengono rimossi.
Lascia aperta la pagina durante l’operazione. L’elaborazione non è un job in
background; il riepilogo in sessione non è un registro persistente dopo il
riavvio dell’app. Atlas rende ricercabili i nuovi chunk in modo asincrono.

## Filtri, punteggi e funzionalità Atlas

La UI offre filtri su file, autore, lingua, categoria, stato, anno, tag e tipo
file. Il ramo `$vectorSearch` usa campi dichiarati `filter`; il ramo `$search`
usa filtri nativi sui campi compatibili e `$match` per quelli residui.
Non è un filtro sulle coordinate del vettore. La soglia rerank controlla il
**miglior candidato** e può bloccare la generazione: non elimina singolarmente
tutti i passaggi sotto soglia e non misura la probabilità di una risposta corretta.

Atlas dispone anche dell'operatore `vectorSearch` dentro `$search`, con
pre-filtri più ricchi, inclusi testo analizzato, frasi e condizioni geografiche.
Questo progetto usa invece lo stage `$vectorSearch`. La distinzione è descritta
nella [guida ufficiale alla ricerca ibrida](https://www.mongodb.com/docs/vector-search/hybrid-search/hybrid-search-overview/).

La demo usa per impostazione predefinita lo stage nativo
[`$rerank`](https://www.mongodb.com/docs/vector-search/query/aggregation-stages/rerank/)
dopo `$rankFusion` (non nei suoi rami). Conserva `score` e `scoreDetails` della
fusione e salva il nuovo punteggio in `_rerank_score`. Il campo `_hybrid_rank`
permette di mostrare il prima/dopo sugli stessi candidati. I Top-K vengono
riordinati e restituiti per la spiegazione; solo i primi Top-N vanno al LLM.
L’app mostra la pipeline completa e la durata combinata ricerca/reranking.
`RERANK_BACKEND=voyage_api` mantiene disponibile il percorso precedente;
gli errori del percorso nativo non causano fallback silenziosi.

Il 19 ottobre 2026 il cluster è risultato versione 9.0.4 e ha accettato
`rerank-3` nativo. L’abilitazione nelle impostazioni di progetto resta necessaria.
Le pagine ufficiali consultate non sono allineate sul requisito minimo:
la guida delle impostazioni riporta 8.3+, la pagina dello stage riporta 9.0+ e
requisiti dedicati per rerank-3. Usare la disponibilità verificata sul cluster;
non dedurla dalla sola disponibilità di Vector Search.

### Istruzioni al reranker: rilevanza adattata allo scopo

I modelli Voyage supportano istruzioni aggiunte alla query per guidare il
reranking: per esempio, alla domanda «Come ripristino la connessione?» si può
aggiungere «Privilegia le procedure operative con sequenze di azioni».
La capacità permette di adattare l'ordinamento dei candidati allo scopo
dell'utente, utilizzando il corpus già indicizzato. Le istruzioni influenzano
la pertinenza; le esclusioni obbligatorie richiedono filtri espliciti.
Vedi la [documentazione Voyage](https://www.mongodb.com/docs/voyageai/models/rerankers/?client=python).

How it Works mette questa capacità in evidenza con un esempio visibile.
La pagina Query offre **Istruzioni al reranker (opzionale)** sotto la domanda.
Il campo parte vuoto; vuoto o contenente solo spazi mantiene la domanda originale
nella richiesta. Se compilato, `src/rerank.py` aggiunge le istruzioni al campo
`query.text` di `$rerank` (oppure `query` dell'API Voyage nel percorso API).
Retrieval e generazione continuano a usare la domanda
originale: le istruzioni influenzano soltanto l'ordinamento dei candidati.
I dettagli del reranking mostrano la domanda e le istruzioni inviate.
Se il reranking non viene eseguito, le istruzioni non vengono applicate.
Il confronto automatico sugli stessi candidati con/senza istruzioni resta da
implementare. Il beneficio sul corpus della demo va misurato;
non è documentata qui un'esclusività rispetto ad altri prodotti.

## Come raccontare e confrontare la demo

Scegliere documenti e risposte attese prima della presentazione. Per ogni
confronto mantenere corpus, domanda, filtri e parametri costanti, cambiando
una strategia per volta. Non confrontare direttamente score BM25, similarità,
fusione e reranking come se avessero la stessa scala.

| Esempio da preparare sul corpus reale | Cosa osservare |
|---|---|
| Un codice o una sigla presenti nel documento | Contributo del canale lessicale |
| La stessa informazione richiesta con parole diverse | Contributo del canale semantico |
| Un passaggio che rimanda a una regola precedente | Differenza tra embedding standard e `context` |
| Una pagina/slide riconoscibile da grafico o diagramma | Posizione del risultato multimodale prima e dopo il reranking testuale |
| La stessa domanda con e senza filtro di categoria | Fonti ammesse dal filtro |
| Una domanda estranea al corpus | Risultati recuperati e comportamento della soglia |

Misurare posizione della fonte attesa, pertinenza dei primi risultati, tempi per
fase e correttezza delle citazioni. Questi sono criteri per la demo; non sono
risultati di un benchmark già eseguito.

## Pagina Ricerca agentica: presentazione a clienti e partner

L'audit del 29 ottobre su pipeline, indici, memoria e un'esecuzione 3M è in
[DEMO_PLAN.md](DEMO_PLAN.md#audit-mongodb-e-ricerca-agentica--29-ottobre-2026).

La pagina parte dalla domanda: offre gli stessi cinque preset di Query e una
domanda libera, che resta visibile durante l'esecuzione. I filtri e i parametri
sono in una sezione espandibile. La sessione è selezionabile e il pulsante
**Nuova** permette di iniziare una conversazione separata.

La modalità predefinita **Con feedback** esegue una ricerca e si ferma per
mostrare risposta e fonti: è l'utente a decidere se accettare o approfondire.
La modalità **Automatica** resta disponibile e può eseguire ricerche aggiuntive
fino al limite scelto. L'accettazione è un giudizio dell'utente e non certifica
la correttezza della risposta o la copertura delle fonti.

Con **No, approfondisci** si registra subito il feedback e si apre un modulo:
motivo, nuova domanda completa obbligatoria e diversa dalla precedente,
informazioni mancanti facoltative, filtri per documento, categoria, autore e
anno, numero di passaggi candidati e fonti, indicazioni al reranker. È utile
specificare nella domanda soggetto, periodo, dato e fonte desiderata. La nuova
ricerca usa esattamente i parametri salvati, esegue un tentativo e chiede di
nuovo il feedback. Le indicazioni dell'utente orientano la ricerca e la risposta,
ma non sono considerate fatti verificati. Se si approfondisce una ricerca
precedente, il pianificatore riceve il contesto di quella ricerca specifica.

I feedback e le richieste di approfondimento sono salvati sia sulla ricerca in
`agentic_runs` sia nella memoria recente della sessione in `agentic_sessions`
(ultimi 30 feedback); la ricerca successiva conserva il collegamento alla
precedente. Si possono rivedere la domanda e la risposta precedenti e i nuovi
parametri. Se una richiesta è stata salvata ma non avviata, può essere ripresa
dalla ricerca precedente; una richiesta già avviata non crea un duplicato. Se
un'esecuzione è stata interrotta dopo l'avvio, non riparte automaticamente.

Durante l'esecuzione il percorso mostra le fasi **Capisce, Cerca, Valuta,
Approfondisce, Risponde** e gli eventi effettivamente registrati: query,
candidati, prime fonti, valutazione, eventuale query aggiuntiva e motivo di
arresto. Una fase non eseguita non viene presentata come completata. Se la
valutazione automatica fallisce, la pagina lo segnala; l'assenza di una seconda
ricerca non dimostra che il corpus sia completo. Le decisioni del modello e
gli esiti possono variare tra esecuzioni.

Al termine si rivedono domanda, risposta e fonti: il riferimento numerato indica
il passaggio inviato al generatore, con file, pagina ed estratto; i passaggi
effettivamente citati sono in evidenza, gli altri sono espandibili. È possibile
scaricare l'originale quando disponibile in GridFS. Gli estratti sono limitati
in lunghezza: il documento originale resta il riferimento per la verifica.
La validità formale delle citazioni non certifica che le affermazioni siano
corrette o sostenute dalla fonte.

**Percorso demo approfondimento:** scegli la domanda su ricavi e utile operativo
AWS, mostra la risposta e le fonti, poi chiedi un approfondimento con una nuova
domanda e un filtro o un'indicazione più precisa. Mostra il collegamento tra i
due risultati e la memoria. In modalità automatica la seconda ricerca dipende
dalla valutazione e non è garantita; il valutatore può essere indisponibile.

Il riquadro **Ciclo di ricerca** conta le ricerche completate e indica il
massimo impostato per quella esecuzione: una iniziale più il numero di ricerche
aggiuntive scelto nelle impostazioni automatiche; in modalità con feedback il
massimo è uno. Il massimo è un limite, non un numero di
tentativi obbligatori. Il valutatore esamina al massimo i primi otto passaggi
riordinati, fino a 1.800 caratteri ciascuno. Un giudizio di copertura insufficiente
significa che il modello non ha riconosciuto tutti gli elementi necessari in quei
passaggi, non che l'informazione sia assente dal corpus. La pagina riporta la
motivazione, gli eventuali elementi mancanti e i riferimenti segnalati dal modello;
se la motivazione è vaga o assente non viene integrata con spiegazioni inventate.

In modalità automatica, prima di un nuovo tentativo l'app confronta la proposta con le ricerche già
eseguite: se è identica oppure ripete gli stessi termini sostanziali nello
stesso ordine, normalizzando maiuscole, punteggiatura ordinaria e parole di collegamento, il secondo
giro non parte e la traccia mostra la proposta scartata. Il confronto mantiene
numeri, negazioni e ordine dei termini; è una verifica testuale prudente, non
una misura del significato e non riconosce tutte le parafrasi. Una proposta
diversa, ad esempio una ricerca mirata all'intestazione di una tabella, può
procedere se restano tentativi. Il valutatore è istruito a proporre un obiettivo
specifico, ma può non riuscirci; in quel caso non si inventa una query.

Ogni ricerca agentica conserva la domanda originale completa, includendo eventuali
vincoli sulla fonte; l'interpretazione del pianificatore e la query di
approfondimento sono aggiunte come contesto. La traccia mostra anche database,
collection, indice e campo di ciascun ramo, oltre ai filtri. Prima dell'esecuzione
si confrontano le definizioni degli indici attivi con quelle configurate; questo
controllo non certifica che tutti i documenti siano già presenti negli indici.

Dopo più tentativi si riordinano i candidati unici rispetto alla domanda completa.
La ricerca non esegue più le chiamate aggiuntive al modello per identificare la
fonte richiesta né promuove passaggi in base a una corrispondenza incerta. La UI
avvisa che un'eventuale fonte o sezione richiesta va verificata nel documento:
il titolo presente in un passaggio non dimostra da solo che sia la sezione giusta.
Le tracce precedenti mantengono la valutazione storica, senza presentarla come
una verifica certa. Il valutatore di copertura rimane separato e può sbagliare.

Per un giudizio positivo di sufficienza, il valutatore deve restituire almeno un
breve estratto letterale presente nel testo realmente esaminato; un riferimento
o un estratto inventato non viene accettato. Gli estratti esaminati sono
consultabili nella traccia. Anche un estratto autentico non garantisce che il
giudizio sia corretto: il modello configurato può dichiarare insufficienti
passaggi che contengono la risposta, perciò la verifica sul PDF resta necessaria.

Se il controllo automatico non riesce, l'agente interrompe il ciclo di
approfondimento e tenta comunque una risposta sui passaggi già trovati; il
controllo delle citazioni può bloccarla. La causa è distinta quando possibile
(timeout, connessione, errore del servizio, risposta non valida o errore della
richiesta). Le nuove esecuzioni registrano il motivo effettivo di ogni decisione:
passaggi ritenuti sufficienti, valutazione indisponibile, limite raggiunto,
query aggiuntiva non proposta, query ripetuta oppure prosecuzione con nuova query.
Le esecuzioni precedenti non contengono necessariamente il limite, la causa o la
motivazione dettagliata: non è possibile ricostruirli a posteriori.

**Percorso demo memoria:** avvia una nuova sessione, scegli la domanda sulle
spese 3M nel 2018 e poi usa il suggerimento **E nel 2017?**. Il percorso mostra
il contesto precedente fornito al pianificatore e la query autonoma prodotta;
verifica nel PDF azienda, anno e valore. Il contesto mostrato è quello reso
disponibile al modello, non una prova di quali parti abbia effettivamente usato.

MongoDB conserva conversazione e riepiloghi in `agentic_sessions`, e domanda,
stato, eventi e, per le nuove esecuzioni completate, risposta e riferimenti con
estratti in `agentic_runs`. **Rivedi una ricerca precedente** permette di
selezionare fino alle 20 esecuzioni più recenti della sessione. Per quelle più
vecchie il recupero della risposta dipende dai messaggi ancora in memoria e gli
estratti dal chunk originale ancora presente; se non sono disponibili, la pagina
lo segnala. Un'esecuzione salvata come in corso dopo un riavvio non riparte da
sola. I dettagli tecnici sono espandibili. La demo non ha autenticazione per
utente e le sessioni presenti nel database sono visibili nel selettore.

## Pagina Query

Un solo selettore **Domanda** contiene le cinque formulazioni concordate in
[DEMO_QUERIES.md](DEMO_QUERIES.md) e **Domanda libera**. Quest’ultima mostra
**Scrivi la tua domanda**, inizialmente vuoto: eliminata la richiesta precompilata
sulla Doganella, anche dalle sessioni migrate. Una bozza scritta dall’utente resta
conservata quando si passa temporaneamente a una domanda preimpostata.
Sono stati rimossi il selettore Scenari demo e il suo riquadro descrittivo.
La selezione invia solo la domanda, senza risposta attesa o suggerimenti sul
trabocchetto. Restano istruzioni facoltative al reranker, filtri e parametri.
Query 2 è stata eliminata; esiste una sola pagina Query.

L’ultima ricerca completata è conservata nella sessione: domanda eseguita,
parametri, risposta, prompt e classifiche restano associati alla stessa
esecuzione anche se si modificano i campi del modulo. Solo **Esegui** avvia una
nuova ricerca e generazione. Gli ispettori del testo BM25 e semantico usano
`st.fragment`: cambiare passaggio aggiorna solo il riquadro, senza rieseguire
la pagina o chiamare Atlas/Voyage/LLM. Anche il download delle fonti evita il
rerun. La persistenza vale nella sessione corrente, non dopo un riavvio del
server o la perdita della sessione browser.

### Dove cerca la domanda

La pagina cerca sempre in tutte le modalità presenti: `auto`,
`manual`, `context` e `multimodal`, indipendentemente dalla modalità di
inserimento in Upload/Admin. Non c'è un selettore della modalità di ricerca.
La tabella mostra database, collection, conteggi dei chunk prima dei filtri,
modalità incluse, indici, modelli e stato. L'inventario è in cache per 30 secondi
e viene riletto al clic di esecuzione. I chunk senza modalità riconosciuta sono
segnalati ed esclusi; un indice richiesto non interrogabile blocca la ricerca
con un messaggio per Admin, senza eliminare quel percorso in silenzio.

Il backend costruisce un ramo `$search` su `pdf_chunks_text_idx` e un ramo
`$vectorSearch` per ogni modalità popolata inclusa. Per esempio, con documenti
`auto` e `multimodal` usa `pdf_chunks_vector_idx` e
`pdf_chunks_multimodal_idx`: Atlas genera la query autoEmbed; l'app genera
separatamente il vettore della domanda con il modello multimodale. I nomi sono
configurabili. Non si riutilizza un vettore tra modelli diversi.

`$rankFusion` combina le posizioni con peso 1 per ramo; lo stesso `_id` trovato
in più rami compare una volta. Copie del contenuto salvate con `_id` diversi
restano documenti distinti. Top-K vale per ogni ramo e per la lista fusa finale;
il reranker seleziona poi Top-N. L'aggiunta di percorsi aumenta il lavoro di
ricerca e può richiedere altre chiamate di embedding della domanda. Reranker
e LLM continuano a ricevere testo estratto, non le immagini.

La pipeline mostrata è quella inviata ad Atlas, con database, collection,
indici, filtri e vettori effettivi. Le classifiche diagnostiche sono ricerche
separate successive, sullo stesso perimetro; non sostituiscono le posizioni
registrate in `scoreDetails` dalla query principale. Il confronto sui testi
mostra BM25 e una classifica per modalità vettoriale testuale, senza una seconda
fusione dei soli vettori. Il dettaglio multimodale non è mostrato in questo
confronto, ma il relativo ramo resta attivo nella ricerca e nella fusione.

### Risposta e spiegazione della ricerca

Ordine della pagina: risposta accettata, download degli originali citati,
Trust Panel, Dettagli della generazione, spiegazione della pipeline.
I dettagli della generazione elencano file, pagine, riferimenti e stato di
citazione; non mostrano il testo dei chunk né il JSON dei tentativi.
Provider, modello e token dell’ultimo tentativo sono nel Trust Panel.

«Come è stata costruita questa risposta?» comprende:

1. **Analisi Query:** domanda e pipeline MongoDB effettiva in Mostra query.
2. **Ricerca testuale e semantica:** definizioni dei punteggi e liste separate.
   BM25 evidenzia le parole della domanda nel testo. Vector mostra testo neutro
   in **Leggi il passaggio recuperato per significato**, posizione dopo reranking
   e citato sì/no. Nessuna evidenziazione lessicale o spiegazione inventata del
   giudizio semantico. Una posizione rerank non disponibile non è uno zero.
3. **Fusione:** formula ufficiale MongoDB `somma(peso / (60 + posizione))`,
   costante 60, peso 1 e un esempio dichiarato illustrativo. Conteggi delle
   origini riferiti ai soli candidati conservati, nascondendo le righe con zero
   passaggi; rank `NA`/`N/A` significa
   assenza. Tabella con ranghi e variazioni per motore, senza indice interno
   del chunk. “auto” è presentato come “Testo · embedding generati da Atlas”.
   Rimosso l’ispettore dei passaggi fusi.
4. **Reranking:** confronto prima/dopo di tutti i candidati iniziali, risalite,
   discese, inclusione nel LLM e citazioni. Nel backend nativo sono disponibili
   tutti i ranghi Top-K; nel backend API solo quelli restituiti. Rimosso
   l’ispettore del testo e il Rank Journey separato. Il servizio non fornisce
   motivazioni individuali per gli spostamenti.
5. **Generazione LLM:** tabella delle sole fonti ricevute: riferimento, file,
   pagina, citato sì/no. **Mostra il prompt inviato al LLM** espone solo le
   istruzioni SYSTEM registrate durante la richiesta, senza il messaggio USER.
   I prompt identici tra tentativi sono mostrati una sola volta. Non è
   un’anteprima ricostruita; per vecchie esecuzioni senza registrazione appare
   un messaggio esplicito. Rimosso l’ispettore separato del testo delle fonti.

### Prompt, citazioni e comportamento in caso di errore

`prompts/system_prompt.txt` contiene le regole generali: usare le evidenze,
leggere correttamente tabelle/anni/unità, distinguere dato e fonte richiesta,
rispondere solo per le parti supportate e citare i passaggi interni. Non contiene
risposte FinanceBench, valori Adobe/3M o istruzioni dedicate al calcolo della
variazione. Le istruzioni su aumento/diminuzione e sottrazione aggiunte dopo
l’errore Adobe sono state rimosse su richiesta dell’utente. Non adattare il
prompt alle domande della demo per farle passare.

Il prompt viene riletto a ogni generazione, rispettando gli override
`LLM_SYSTEM_PROMPT` / `LLM_SYSTEM_PROMPT_FILE`. Il primo messaggio usa la domanda
originale; il retry aggiunge un controllo generale delle evidenze e dei riferimenti.
`generate_reviewed_answer` esegue al massimo due tentativi sugli stessi passaggi.
Il controllo riconosce riferimenti interni validi e rifiuta link non interni,
URL esterni e riferimenti a numeri fuori dal contesto. `[#1](URL esterno)` non
viene convertito in una citazione valida.

Se anche l’ultimo risultato non supera il controllo, la risposta viene bloccata.
La pagina distingue collegamenti non consentiti, numeri di passaggio fuori
intervallo, citazioni assenti e risposta vuota. Per le nuove ricerche agentiche,
la traccia MongoDB salva il controllo e il testo grezzo dei tentativi, limitato
a 12.000 caratteri ciascuno, consultabili nel dettaglio diagnostico; non salva
le richieste complete o credenziali. La pagina Query li mantiene nella sessione
Streamlit. Le esecuzioni agentiche precedenti possono non avere tali dettagli.
Una dichiarazione di insufficienza senza riferimenti proibiti può essere
mostrata, con spiegazione che non dimostra l’assenza del dato nel database.
Il controllo di forma **non certifica fatti, calcoli o supporto delle affermazioni**.

Nel percorso locale Ollama `/api/chat` non vengono passati strumenti web e non
esiste una ricerca Internet del generatore. Un URL nel testo generato non prova
una navigazione: nel caso Adobe era un collegamento prodotto dal modello.
La ricerca del corpus resta su Atlas; embedding/rerank utilizzano i rispettivi
servizi configurati. Non descrivere l’intero sistema come offline.

### Configurazione osservata e verifiche

La demo usa ancora **Ollama `mistral-nemo:latest`**, Top-K 50, Top-N 8,
`num_ctx=8192`, temperatura 0.1 e top_p 0.9. Qwen3.6, contesto 16384 e Top-N 12
sono stati proposti, **non applicati**. `RERANK_BACKEND=native` usa `rerank-3`.
Le variabili in `.env` possono sovrascrivere i default di configurazione.

La diagnosi 3M ha trovato il dato nel contesto LLM (p.39) e il rendiconto p.46
fuori dal Top-8 (9° nell’esecuzione globale ispezionata). Il confronto locale
su otto passaggi pubblici salvati, con seed 17/42/73, ha dato 2/3 risposte corrette
con il vecchio prompt e 3/3 con quello semplificato e citazioni esplicite.
Sono prove storiche circoscritte, non un collaudo globale o una garanzia.
Il 20 ottobre la domanda Adobe 5 è fallita nonostante i valori corretti nei
primi tre passaggi: +476 e URL esterno invece di −66 milioni. Le correzioni
successive non sono state ancora verificate con una nuova esecuzione LLM.
Per esiti e limiti consultare [DEMO_PLAN.md](DEMO_PLAN.md) e la scheda delle domande.

Dopo modifiche ai moduli Python o a `.env`, riavviare Streamlit: il solo rerun
può mantenere moduli importati in memoria. Questo ha causato l’AttributeError
su `RERANK_BACKEND`, risolto con il riavvio. Il file del prompt viene invece
riletto a ogni richiesta. Il riavvio azzera lo stato transitorio della sessione.

## Pagina How it Works

La pagina è organizzata per accompagnare la presentazione:

1. **Introduzione:** significato di RAG e ruolo di Atlas per dati, full-text e
   ricerca vettoriale.
2. **Inserimento:** diagramma dedicato, significato di chunk ed embedding,
   tabella delle quattro modalità e approfondimento sui parser.
3. **Domanda:** diagramma con i due canali di ricerca, fusione, reranking e
   risposta; spiegazione del limite attuale del percorso multimodale.
4. **Esempio guidato:** domande illustrative e indicazioni su cosa osservare
   nelle classifiche, da adattare ai documenti realmente caricati.
5. **Approfondimenti:** HNSW e filtri, pipeline di esempio, strategie testuali,
   reranking e significato della soglia.

Sono stati rimossi l'elenco ridondante “Cosa dimostra questa demo”, le card
“KPI osservabili” con valori statici e il confronto generico con la “RAG
tradizionale”. La spiegazione del reranking nella pagina non riporta vincoli
su tier o versione del cluster. Le funzionalità native non adottate restano
distinte dal percorso implementato.

### Attività tecniche ancora aperte

Il passaggio delle immagini a reranking/generazione rimane un possibile
sviluppo: la riorganizzazione della pagina non modifica il percorso multimodale.

Il filtro `author` usa ora il mapping testuale `token`, coerente con
`equals`/`in`, ed è presente anche nei filtri dell'indice multimodale.
Il setup aggiorna le installazioni con le vecchie definizioni.

### Piano di verifica concordato

Il riferimento è [DEMO_PLAN.md](DEMO_PLAN.md): 100 PDF, caricamento cartella e
5 domande, almeno 2 a trabocchetto. Per ogni domanda preparare prima risposta
attesa, documento, pagina ed evidenza; confrontare manualmente risposta e
citazione con il PDF. Il pacchetto da 100 PDF è disponibile localmente e il
caricamento cartella è implementato. [DEMO_QUERIES.md](DEMO_QUERIES.md) contiene
le 5 domande con risposte e pagine verificate nei PDF; gli esiti distinguono le risposte segnalate corrette dall’utente dalle verifiche
dirette e registrano il fallimento della domanda 5. Non caricare questa scheda nel database.
Il confronto tra strategie resta a voce; nessun runner o dashboard da costruire.
La pagina Query continua a cercare in tutte le modalità popolate.

## Struttura del progetto

- `app.py`: interfaccia Streamlit e orchestrazione.
- `src/retrieval_details.py`: identità dei passaggi e lettura delle posizioni nella fusione.
- `src/search_scope.py`: inventario e perimetro di ricerca condiviso, con indici visibili.
- `src/answer_quality.py`: generazione della risposta, pulizia dei tag di ruolo e controllo dei riferimenti.
- `src/ingestion.py`, `src/embeddings.py`: parsing, chunking e embedding.
- `src/retrieval.py`, `src/rerank.py`, `src/generation.py`: ricerca e risposta.
- `src/init_db.py`, `src/indexes.py`: preparazione database e indici.
- `schemas/`: definizioni degli indici Atlas Search e Vector Search.
- `prompts/`: istruzioni al LLM per risposte con fonti.
- `tests/`: test della pipeline; `scripts/`: strumenti diagnostici.

Mantenere questo README e la sintesi inglese allineati quando cambiano setup,
modalità, dipendenze o limiti. La pagina How it Works deve riflettere il
comportamento implementato; le proposte devono restare riconoscibili come tali.
