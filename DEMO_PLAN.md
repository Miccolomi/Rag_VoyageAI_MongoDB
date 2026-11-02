# Obiettivo e piano della demo RAG

**Ultimo aggiornamento: 29 ottobre 2026.** Questo documento descrive lo stato
corrente; la cronologia delle prove è separata più sotto. Guida tecnica e UI:
[README.it.md](README.it.md), versione inglese [README.md](README.md).

## Obiettivo confermato e perimetro attuale

«Lasciare tutti a bocca aperta» con una demo concreta e comprensibile su un
archivio consistente. Ultima decisione dell’utente: mantenere il lavoro semplice.
Questo piano sostituisce il precedente programma di benchmark esteso.

## Attività concordate

1. Preparare 100 PDF simili da FinanceBench: aziende e anni diversi, con voci
   contabili ricorrenti che richiedano di distinguere la fonte corretta.
2. Aggiungere il caricamento di una cartella, riutilizzando l’ingestion esistente,
   con avanzamento e riepilogo di successi, errori e duplicati.
3. Preparare esattamente 5 domande con risposte attese verificate nei PDF;
   almeno 2 devono essere a trabocchetto.
4. Preparare una semplice scheda di verifica manuale per le 5 domande.

Fonte del corpus: https://github.com/patronus-ai/financebench
Corpus preparato: 100 PDF, con manifest di provenienza e integrità.
Non caricare nel database le risposte attese o la scheda di verifica.

## Cinque casi preparati

Le domande definitive e le risposte verificate sono in
[DEMO_QUERIES.md](DEMO_QUERIES.md). I cinque casi sono:

1. Estrazione di un valore esplicito da una tabella.
2. Domanda in italiano su una voce contabile in inglese.
3. Domanda che richiede due dati espliciti, con verifica di entrambi.
4. Trabocchetto sull’anno o sull’azienda: fonti simili contengono numeri diversi;
   il sistema deve scegliere il valore per l’entità e il periodo richiesti.
5. Trabocchetto con premessa falsa: la domanda suggerisce un aumento, ma la
   tabella mostra una diminuzione. Il sistema deve correggere la premessa.

Il quinto caso usa una contraddizione dimostrabile nel PDF: il mancato
recupero di un passaggio non dimostra che l’informazione sia assente dal database.

## Come decidere se la risposta è corretta

Preparare PRIMA dell’esecuzione una scheda Markdown con, per ogni domanda:

- Domanda esatta da incollare.
- Risposta attesa: fatti, numeri, unità di misura, azienda e periodo.
- File sorgente e pagina del PDF, distinguendola dalla numerazione stampata.
- Passaggio o tabella da controllare, e breve spiegazione dell’eventuale trabocchetto.
- Risposta ottenuta ed esito manuale: corretta, parziale o errata.
- Controllo separato della citazione: sostiene davvero quanto affermato?

Usare le annotazioni FinanceBench come punto di partenza e controllarle nel PDF
originale. La numerazione delle evidenze nel dataset parte da zero: normalizzarla.
Una formulazione equivalente è valida; numeri, unità, periodo e soggetto devono
coincidere. Nei casi composti verificare tutte le parti richieste.
Una risposta esatta con citazione sbagliata va segnalata come tale, senza
presentarla come prova completamente riuscita. Una risposta senza evidenza
sufficiente è un caso non riuscito anche se non inventa un numero.

Durante la demo: domanda, risposta, apertura della fonte citata e confronto
con la scheda. Non usare il punteggio del reranker come probabilità di correttezza.
Cinque casi dimostrano quei comportamenti su 100 documenti; non misurano
l’accuratezza generale del sistema.

## Escluso dal lavoro attuale per decisione dell’utente

- Confronti automatici tra testuale, vettoriale, ibrida e reranking: per ora se
  ne parla a voce usando le informazioni già disponibili nella pagina Query.
- Confronti tra modalità di embedding.
- Nuovi sviluppi multimodali o set aggiuntivi Word/PowerPoint.
- Runner di benchmark, dashboard di valutazione e report automatici.

Queste attività non sono passi successivi già approvati.

## Vincoli da mantenere

- Query cerca in tutte le modalità popolate; non reintrodurre «Dove cercare» o Query 2.
- Conservare i documenti esistenti e rendere riconoscibili quelli del nuovo set.
- Ogni operazione deve spiegare cosa fa e cosa ha prodotto.
- Le istruzioni al reranker rimangono opzionali.
- Non modificare ora la gestione del contesto LLM.

## Stato attuale dell’implementazione

### Dati, ingestion e setup

- Preparati 100 PDF FinanceBench (179,9 MB) in `data/financebench_100/pdfs`.
  `scripts/prepare_financebench.py` verifica integrità e consente ripresa dei download.
  Manifest e annotazioni sono fuori dalla cartella PDF da caricare.
- Upload supporta uno/più file e cartella; avanzamento sequenziale, errori per
  file, duplicati, riepilogo scaricabile e informazioni su pagine/chunk/contenuti.
  I percorsi di inserimento distinguono testo e testo/contenuti visivi.
- L’utente ha caricato i documenti. Nell’inventario ispezionato erano presenti
  53.729 chunk auto e 4 multimodali; questi numeri non certificano l’ingestion
  completa di tutti i 100 file. Nessun collaudo end-to-end dell’upload da parte
  dell’assistente in questa fase.
- Admin prepara database, collection, GridFS e indici con piano, azioni e stato.
  Mostra il codice MongoDB di creazione/aggiornamento. Il testo sul setup è sotto
  il pulsante. L’allineamento degli schemi include author e percorso multimodale.
- How it Works spiega quattro modalità, parser, conversione Office tramite
  LibreOffice, GridFS come opzione per binari e alternativa object storage,
  Atlas Search testuale, Vector Search e fusione. Le immagini sono utilizzate
  negli embedding multimodali; reranker e generatore ricevono ancora testo.

### Ricerca, modelli e parametri

- Una sola pagina Query; Query 2 eliminata. Ricerca globale su tutte le modalità
  popolate, senza selettore “Dove cercare”. Filtri metadata applicati a tutti i rami.
- Un ramo testuale condiviso e uno vettoriale per modalità. Nessun confronto
  diretto tra vettori di modelli diversi. Fusione RRF con peso 1 per ramo.
- Default `RERANK_BACKEND=native`, `$rerank` dopo `$rankFusion`, modello `rerank-3`.
  Alternativa API esplicita; nessun fallback silenzioso se il percorso nativo fallisce.
  Istruzioni al reranker opzionali, vuote per le prove concordate.
- Il cluster è stato verificato 9.0.4 con rerank-3 disponibile il 19 ottobre.
  Non è una verifica periodica dello stato del servizio.
- Modello della demo ancora **mistral-nemo:latest** su Ollama, Top-K 50, Top-N 8,
  contesto 8192, temperatura 0.1, top_p 0.9. Qwen3.6, contesto 16384 e Top-N 12
  sono proposte non applicate. Conservare questi fatti nelle spiegazioni.

### Pagina Query attuale

- Corretto il ciclo di visualizzazione: risultati salvati in Session State e
  rendering separato dal pulsante **Esegui**. Domanda, parametri, risposta,
  prompt e classifiche appartengono alla stessa esecuzione salvata.
  Gli ispettori BM25/Vector sono frammenti Streamlit: selezionare un passaggio
  aggiorna solo il riquadro. Download con `on_click="ignore"`.
  Nessuna nuova chiamata a retrieval, reranker o LLM durante la consultazione.
  Correzione implementata, collaudo interattivo non effettuato in questo turno.
- Selettore unico **Domanda**: cinque domande identiche a `DEMO_QUERIES.md` più
  **Domanda libera**. Campo libero inizialmente vuoto, anche dopo migrazione da
  sessioni con il vecchio default Doganella. Le bozze dell’utente sono preservate
  passando tra libera e preset. Rimossi Scenari demo e relativa card.
- Preset = solo testo della domanda. Nessuna risposta attesa o indicazione sul
  trabocchetto viene inviata a ricerca, reranker o LLM.
- Risposta accettata, download fonti citate, Trust Panel (inclusi provider,
  modello e token), dettagli con soli file/pagine/riferimenti e citato sì/no.
- **1. Analisi query:** pipeline MongoDB eseguita visibile.
- **2. Ricerca:** BM25 con parole evidenziate; Vector con testo neutro, posizione
  dopo reranking e citato sì/no. Sono ricerche diagnostiche successive separate.
  Niente tabella multimodale in questo confronto; il ramo resta nella ricerca.
- **3. Fusione:** formula ufficiale MongoDB e link, spiegazione del 60 e dei pesi,
  esempio esplicitamente illustrativo. Conteggi delle origini partizionano i
  candidati conservati; le righe con zero passaggi sono nascoste.
  NA/N/A significa assente, non “condiviso”. Posizioni e
  variazioni per motore, nomi parlanti delle modalità, niente colonna chunk né
  ispettore del testo.
- **4. Reranking:** prima/dopo sull’intera lista iniziale, inclusi esclusi dal
  LLM. Nel backend API ranghi non restituiti restano sconosciuti. Nessuna
  motivazione inventata per gli spostamenti. Rimossi ispettore e Rank Journey.
- **5. LLM:** tabella delle sole fonti ricevute, con riferimento/file/pagina/
  citato sì-no. Il riquadro del prompt mostra solo SYSTEM dalla richiesta reale,
  senza USER e senza ripetere prompt identici tra tentativi.
  Niente matrice ridondante o ispettore separato.

### Pagina Ricerca agentica per la demo

- Ottimizzazione: il controllo della sezione non fa più chiamate aggiuntive al
  modello e non promuove automaticamente risultati in base a un titolo.
  L'ordine del reranker è preservato e l'eventuale fonte richiesta deve essere
  controllata nel documento. Il valutatore di copertura resta attivo.
- Risposte bloccate: visualizzati motivi specifici e dettaglio diagnostico dei
  tentativi. Le nuove tracce agentiche salvano motivo, esiti, modello e testo
  grezzo limitato a 12.000 caratteri per tentativo; non le richieste complete.
  Le ricerche storiche possono non avere questi dettagli. La pagina Query tiene
  il dettaglio nella sessione Streamlit.
- Collaudo su `3M_2018_10K.pdf`, stessa domanda e filtro della prova precedente:
  i primi tre passaggi restano nell'ordine del reranker e la fonte richiesta è
  dichiarata non verificata. La nuova esecuzione ha prodotto una risposta con
  citazione formalmente valida e ha salvato il dettaglio in MongoDB. La risposta
  citava p.49: verificare sul PDF se soddisfa la richiesta del rendiconto.
  Intervallo tra eventi fusione e controllo fonte: circa 9,2 s nella prova
  precedente e 0,6 s nella nuova; include le scritture di checkpoint. Tempo fino
  all'evento risposta: circa 73,7 s prima e 66,1 s dopo. Sono due sole prove e
  tempi di modello e rete variano; non rappresentano una garanzia di prestazioni.
  I casi di blocco e il recupero dei dettagli sono coperti da test simulati.
- Modalità predefinita **Con feedback**: una ricerca, risultato e fonti, quindi
  scelta dell'utente tra accettare e approfondire. L'accettazione è un giudizio
  personale, distinto dalla valutazione automatica. **Automatica** conserva il
  ciclo con il limite configurabile e la protezione dalle query ripetute.
- Il feedback negativo è salvato subito. Per approfondire occorre una nuova
  domanda completa e diversa, con motivo e informazioni aggiuntive facoltative;
  si possono modificare filtri documento/categoria/autore/anno, candidati,
  fonti e istruzioni al reranker. La nuova richiesta fa una sola ricerca e
  attende nuovamente il feedback. Le indicazioni non sono fatti verificati.
- Feedback, domanda, parametri e collegamenti sono conservati nella ricerca e
  nella memoria della sessione (ultimi 30 feedback). La ricerca collegata usa
  la domanda e risposta della ricerca selezionata come contesto anche se più
  vecchia di altre; mostra il confronto e il percorso. Una richiesta salvata ma
  non avviata è riprendibile, senza creare duplicati dopo l'avvio; una ricerca
  interrotta durante l'esecuzione non viene riavviata automaticamente.
- Test automatici su validazione del modulo, persistenza del feedback, parametri,
  collegamento e contesto della ricerca selezionata. L'uso effettivo del modello
  e la qualità delle risposte dipendono dai servizi e dal corpus disponibili.
- Collaudo reale 3M 2018: una sola ricerca, domanda mantenuta e modulo di
  feedback con il filtro documento conservato. Al primo approfondimento una
  data MongoDB nel riepilogo causava un errore di serializzazione; corretto e
  coperto da test. Nella prova ripetuta la nuova query 2017 include le
  indicazioni aggiuntive e la traccia mostra la domanda e risposta precedente.
  L'approfondimento si è concluso mostrando confronto, fonti e nuovo feedback,
  ma la risposta è stata bloccata dal controllo formale; il motivo preciso non
  era stato salvato. La prova non dimostra una risposta corretta al 2017.
  Nel controllo di fonte precedente il modello ha
  indicato «Business Segment Information» come possibile rendiconto finanziario;
  il titolo esiste nel passaggio, ma non è il rendiconto richiesto. È necessario
  controllare personalmente la fonte prima di usarla nella demo.
- Per evitare giri inutili, il valutatore è invitato a proporre una ricerca
  mirata all'elemento mancante. Il ciclo rifiuta una query identica o con la
  stessa sequenza di termini sostanziali, ignorando punteggiatura e parole di
  collegamento; mostra nella traccia la proposta scartata. Numeri, negazioni e
  ordine sono mantenuti. È un confronto testuale conservativo: non riconosce
  tutte le parafrasi e non garantisce che una query diversa trovi nuove fonti.
  La decisione ricalcolata sulla traccia AWS salvata scarta la proposta ripetuta;
  i test coprono anche cambi di anno, entità e negazione. Non è una nuova
  esecuzione end-to-end del modello sul corpus.
- Il ciclo mostra ricerche completate e massimo configurato, e distingue i
  motivi di prosecuzione o arresto: esito favorevole, valutazione non riuscita,
  limite raggiunto, query non proposta o duplicata. Le esecuzioni precedenti non
  hanno necessariamente il limite o la causa salvati.
- Il controllo confronta domanda e query con al massimo 8 passaggi riordinati,
  fino a 1.800 caratteri per passaggio. La UI spiega che la copertura è una
  valutazione del modello sui passaggi ricevuti, non una prova di assenza nel
  corpus; mostra commento ed eventuali elementi mancanti e riferimenti indicati.
  Errori del valutatore sono classificati senza salvare messaggi grezzi potenzialmente
  sensibili; in assenza di valutazione il ciclo si ferma e il generatore tenta
  comunque una risposta sui passaggi già trovati, soggetta ai controlli esistenti.
- Verifica UI del 29 ottobre su una ricerca 3M 2017 precedente: mostrati due
  tentativi, commento generico del valutatore al primo e valutazione fallita al
  secondo. Per quella esecuzione il limite e la causa tecnica non erano stati
  salvati e la pagina li segnala come non ricostruibili. Nuovo ciclo e diagnosi
  verificati con test simulati; non ancora con una nuova interrogazione completa.
- Domanda libera e cinque preset condivisi con Query; il testo resta visibile
  all'invio e la bozza libera si conserva passando ai preset.
- Percorso visibile mentre l'agente lavora: interpretazione, ricerca,
  valutazione, eventuale approfondimento, risposta. Sono mostrati query,
  fonti in evidenza, motivazione e motivo di arresto quando registrati; una
  valutazione indisponibile o una ricerca non eseguita non vengono mascherate.
- Domanda e risposta sono seguite dalle fonti numerate effettivamente fornite
  al modello; le citate sono in evidenza, le altre espandibili. Sono mostrati
  estratto e pagina e, se presente in GridFS, il download del file originale.
- La sessione conserva la conversazione; le nuove esecuzioni salvano risposta e
  snapshot dei riferimenti in `agentic_runs`, per rivedere insieme domanda,
  risposta, fonti e percorso anche dopo un aggiornamento. Le ricerche precedenti
  usano la memoria e il chunk originario, quando disponibili. I dettagli tecnici
  e la conversazione sono espandibili. L'esecuzione interrotta non riprende da sola.
- Due percorsi guidati: una domanda AWS da osservare per eventuale ricerca
  aggiuntiva e domanda 3M 2018 seguita da «E nel 2017?» per la memoria. La UI
  mostra il contesto fornito al pianificatore senza affermare che sia stato
  certamente utilizzato. L'approfondimento dipende dalla valutazione reale.
- Aggiunte verifiche automatiche su salvataggio, ripresa dei dati precedenti e
  visualizzazione dei casi di valutazione indisponibile e ricerca aggiuntiva.
  Il 29 ottobre: 11 test automatici superati; verificati nell'interfaccia la
  ripresa di una ricerca 3M precedente, fonti, percorso con seconda query e
  inserimento del suggerimento «E nel 2017?». Non è stata eseguita una nuova
  ricerca completa con la pagina riprogettata: collaudare i percorsi e le fonti
  sul corpus della demo prima della presentazione.

### Prompt e controllo della risposta

- Prompt generale in `prompts/system_prompt.txt`, riletto per ogni chiamata,
  con override inline/file rispettati. Regole su evidenze, tabelle, anni/unità,
  fonte richiesta, risposte parziali e citazioni interne.
- Nessuna risposta o dato di riferimento Adobe/3M nel prompt. Rimosse dal prompt
  e dal retry le istruzioni su aumento/diminuzione e sottrazione aggiunte dopo
  il fallimento Adobe. Non adattare le istruzioni per far passare i cinque casi.
- Al massimo due tentativi sugli stessi passaggi. Non si assegnano citazioni
  automaticamente e non si sostituisce la risposta con estratti copiati.
- Risposta finale con link esterni/non interni, riferimenti fuori intervallo o
  senza citazioni valide: **bloccata**, non mostrata con un semplice avviso.
  I nuovi tentativi agentici sono salvati nella traccia per diagnosi. Il formato
  `[#1](URL esterno)` non è accettato come citazione interna.
- Una dichiarazione di insufficienza senza riferimenti proibiti può essere
  mostrata. Non prova che il dato manchi nel corpus. Il controllo delle citazioni
  non certifica accuratezza fattuale, supporto delle frasi o correttezza aritmetica.
- Ollama riceve una richiesta locale `/api/chat` senza strumenti web. L’URL
  esterno generato nel caso Adobe non è il risultato di una navigazione del LLM.
  Atlas e Voyage sono servizi di rete: non descrivere tutta la RAG come offline.

## Evidenze e cronologia delle verifiche

### Inventario embedding e compatibilità context — 29 ottobre 2026

- Lettura in `rag.pdf_chunks`: 53.729 chunk `auto` con modello registrato
  `voyage-4-large` in 104 documenti distinti; 4 chunk `multimodal` con
  `voyage-multimodal-3.5` per un documento. Nessun chunk `context` rilevato.
  L'indice `pdf_chunks_vector_idx` risultava READY e la sua definizione
  autoEmbed su `content` indicava `voyage-4-large`. Il report non certifica
  singolarmente l'indicizzazione dei chunk.
- Aggiunto in Upload un inventario su richiesta raggruppato per documento,
  modalità, modello e strategia registrati, senza inventare metadati mancanti.
  L'ultima definizione disponibile dell'indice auto è mostrata separatamente.
- La [lista ufficiale Atlas Automated Embedding](https://www.mongodb.com/docs/vector-search/crud-embeddings/automated-embedding/models/)
  non include `voyage-context-4`. L'[API contestualizzata](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
  supporta invece questo modello; nella demo il percorso `context` usa vettori
  propri e un indice separato. Chunk pre-divisi sono limitati a 32K token nella
  nostra implementazione, che non usa l'auto-chunking dell'API. La coesistenza
  non prova che il recupero o le risposte migliorino: serve un confronto sul corpus.
- Verificati inventario reale e indice in sola lettura e test automatici, inclusi
  casi di metadati mancanti e indice non consultabile; nessuna reingestion eseguita.

### Audit MongoDB e ricerca agentica — 29 ottobre 2026

Verifica in sola lettura sul database configurato `rag`, collection
`pdf_chunks`: 53.729 chunk `auto`, 4 `multimodal`, nessun chunk `manual` o
`context` nell'inventario. Gli indici `pdf_chunks_text_idx`,
`pdf_chunks_vector_idx`, `pdf_chunks_manual_idx`, `pdf_chunks_context_idx` e
`pdf_chunks_multimodal_idx` risultavano READY, interrogabili e conformi alle
definizioni configurate. READY non prova da solo che ogni chunk sia indicizzato
o che la ricerca restituisca la fonte corretta.

La ricerca agentica usa la stessa pipeline della pagina Query. Il ramo `$search`
interroga il campo `content`; i rami vettoriali attivi usano `content` con
autoEmbed e `embedding_multimodal` per il multimodale. I filtri sui metadati
sono applicati ai rami e dopo la fusione. `$rankFusion` combina i ranghi e
`$rerank` nativo riordina i candidati. Le modalità non popolate non vengono
incluse. Prova in sola lettura su `3M_2017_10K.pdf`: ricerca testuale,
vettoriale auto e ibrida hanno restituito risultati; prova separata del ramo
multimodale con vettore sintetico ha restituito 4 chunk. La prova multimodale
verifica l'interrogabilità, non la qualità semantica. Nessuna prova end-to-end
di `$rerank` o della completezza degli indici eseguita durante questo audit.

`agentic_sessions` contiene conversazione, riepiloghi e stato della sessione;
`agentic_runs` contiene domanda, stato, eventi e risultato. Gli indici B-tree
`updated_at_-1` e `session_id_1_created_at_-1` erano presenti e i piani explain
mostravano IXSCAN. GridFS aveva gli indici standard attesi. Su `pdf_chunks` era
presente come indice B-tree solo `_id_`: il controllo duplicati per
`document_id` e `embedding_mode` ha fatto COLLSCAN su 53.733 documenti per un
identificativo assente. Valutare un B-tree composto su tali campi per ingestion
e accessi per document_id; non sostituisce gli indici Atlas Search e non è
dimostrato che risolva la qualità delle risposte. Nessun indice è stato creato
o aggiornato durante l'audit.

Nell'esecuzione salvata 3M 2017, la query pianificata era «3M spese in conto
capitale 2017»: aveva omesso il vincolo esplicito sul rendiconto finanziario.
La prima ricerca aveva comunque recuperato al numero 8 `3M_2019_10K.pdf` p.46,
con la sezione Cash Flows from Investing Activities, la voce PP&E e 1.373 nei
primi 1.800 caratteri inviati al valutatore. Il valutatore ha dichiarato le
fonti insufficienti senza spiegare un elemento mancante preciso. La seconda
valutazione è fallita senza causa tecnica registrata nella vecchia versione.
La risposta ha citato p.39 del 3M 2017 invece del rendiconto richiesto; il
passaggio 3M 2018 p.46 è comparso al 7° posto nella seconda ricerca, ma non
nelle otto fonti finali selezionate dalla fusione tra i tentativi. Queste sono
evidenze sulla singola esecuzione, non una diagnosi generale di accuratezza.

### Interventi successivi all'audit — 29 ottobre 2026

- Creata su `rag.pdf_chunks` la B-tree `document_id_1_embedding_mode_1` e
  aggiunta sia al setup Admin sia al pulsante **Verifica / Crea indici**.
  Per il controllo su un identificativo assente,
  `explain` usa IXSCAN con 0 documenti esaminati, contro i 53.733 della
  precedente COLLSCAN. Non sono stati modificati documenti o indici Search.
- La domanda originale completa viene conservata nella query di ciascun
  tentativo e nel riordino finale; interpretazione e approfondimento sono
  aggiunti senza sostituirla. La traccia indica database, indici, campi e
  filtri. Il controllo dello scope rileva anche definizioni degli indici attivi
  diverse dalla configurazione, senza dichiarare copertura completa degli indici.
- In questa verifica storica, dopo più tentativi i candidati unici venivano
  riordinati e il controllo della fonte esaminava gruppi di passaggi. Il controllo
  e la promozione automatica sono stati successivamente rimossi: vedere lo stato
  corrente nella sezione «Pagina Ricerca agentica per la demo».
- La valutazione positiva richiede almeno un estratto letterale riscontrabile nei
  passaggi ricevuti; le evidenze esaminate sono consultabili nella traccia. Un
  estratto presente non certifica il ragionamento e la valutazione negativa può
  essere erronea. I modelli e le decisioni restano gli stessi configurati.
- Prova reale limitata al PDF pubblico `3M_2018_10K.pdf`: la query completa ha
  recuperato p.46 al 2° posto nel primo riordino; il riordino finale di 22
  candidati l'ha mantenuta nelle prime 8. Un collaudo UI con una prima versione
  del controllo ha invece scambiato p.39 per il rendiconto e la risposta ha
  citato p.39; la successiva versione richiede un titolo autonomo e nella prova
  dedicata ha individuato p.46 e il titolo `Cash Flows from Investing Activities`.
  Nel successivo collaudo UI della versione che verifica il titolo, con filtro
  `3M_2018_10K.pdf` e nessuna ricerca aggiuntiva, la risposta ha riportato 1.373
  milioni citando p.46; l'estratto della fonte mostra intestazione, anni e riga
  PP&E. Il valutatore ha comunque dichiarato insufficienti i
  passaggi che contenevano quel dato. In un'altra prova il valutatore
  locale ha comunque dichiarato insufficienti i primi due passaggi, pur essendo
  presenti la pagina 46 e il valore richiesto: non dichiarare il problema del
  valutatore risolto o il risultato generale garantito; questa è una sola prova
  filtrata, non una verifica dell'intero corpus o delle cinque domande.
- Le 26 verifiche automatiche includono conservazione dei vincoli, controllo
  degli estratti, selezione della fonte, riordino finale, indici e gestione dei
  casi di errore; non sostituiscono il collaudo del modello con dati reali.

### Corpus e fonti di riferimento

Il 19 ottobre sono state verificate le risposte attese leggendo le tabelle e le
pagine dei PDF originali, anche visivamente. Sono riferimenti per la valutazione,
non risultati della RAG. La scheda non va caricata nel database.

### 3M e variabilità di Mistral

- Nella risposta negativa ispezionata, il LLM aveva già il dato 1.577 a pagina 39.
  La pagina 46 del rendiconto era recuperata ma esclusa dal Top-8: 9ª nella
  successiva esecuzione globale ispezionata; 10ª in una prova pubblica separata.
- La prova nativa limitata ai PDF pubblici ha risposto correttamente con lo
  stesso modello. Non dimostra superiorità del backend o successo globale.
- Confronto locale su otto passaggi pubblici salvati, seed abbinati 17/42/73:
  vecchio prompt 2/3 corrette; prima bozza 3/3 sul dato ma URL inventati in 2/3
  (scartata); prompt con citazioni esplicite 3/3 corrette con riferimento interno.
- Controlli aggiuntivi: 2017 → 1.373; 2025 → assenza del dato. La risposta 2025
  era prolissa. Il modello non esplicitava sempre il limite della fonte p.39
  rispetto al rendiconto richiesto. Non generalizzare da queste prove.
- Il controllo automatico aveva bloccato una prova globale dell’assistente per
  invio di candidati non pubblici al nuovo reranking nativo. È stata eseguita
  l’alternativa sui soli PDF pubblici; non aggirare quel vincolo. L’ispezione
  successiva delle risposte già eseguite dall’utente è un’attività distinta.

### Adobe, domanda 5 — fallimento osservato il 20 ottobre

- La pagina mostrava +476 milioni e un URL esterno; il prompt conteneva già
  i valori corretti nei passaggi #1 p.55, #2 p.54 e #3 p.57: 2022 4.756 e 2021
  4.822 milioni. Risposta attesa: diminuzione di 66 milioni.
- Il modello ha sbagliato nonostante evidenze presenti. L’app mostrava l’output
  anche dopo l’avviso sulle citazioni; questo comportamento è stato corretto.
- Le istruzioni aritmetiche aggiunte in seguito sono state rimosse per decisione
  dell’utente. La domanda resta fallita; nessuna nuova esecuzione LLM ha ancora
  verificato le correzioni successive. Non trasformare questo esito in successo.
- L’utente ha segnalato corrette le risposte precedenti alla quinta. La domanda 1
  ha anche fallimenti storici osservati. Gli esiti segnalati e quelli controllati
  direttamente vanno distinti nella scheda.

## Limiti aperti e continuità del lavoro

- Da verificare nuovamente la domanda 5 con configurazione corrente, annotando
  esito e fonti, senza adattare il prompt alla risposta attesa.
- Risolto il blocco di pubblicazione dei riferimenti non validi nel codice;
  non ancora verificato con una nuova generazione dopo la modifica. Nessuna
  promessa che una citazione valida impedisca errori fattuali.
- Qualità di estrazione delle tabelle, selezione del Top-N e affidabilità LLM
  restano limiti osservati. Non modificare contesto/modello di nascosto.
- Ultime revisioni UI e campo libero: implementate, senza nuovo collaudo completo
  della pagina o delle cinque query. Non dichiararle un benchmark superato.
- L’AttributeError `RERANK_BACKEND` era dovuto al modulo importato ancora in
  memoria; risolto riavviando Streamlit. Riavviare dopo modifiche Python/.env,
  considerando la perdita dei risultati transitori. Il prompt è riletto per chiamata.
- Non avviare nuovi benchmark, dashboard, sviluppi multimodali o migrazione LLM
  senza una nuova decisione dell’utente. Nessuna cancellazione/reingestion o
  ricreazione degli indici è necessaria per le modifiche di presentazione.
