# Cinque query per la demo FinanceBench

Risposte attese verificate il 19 ottobre 2026 tramite lettura delle tabelle e
controllo visivo delle pagine dei PDF locali. Durante la preparazione iniziale queste query non erano state eseguite contro
la RAG. La tabella finale è stata aggiornata il 20 ottobre con gli esiti successivi,
distinguendo segnalazioni dell’utente e diagnosi dirette. Questa scheda NON va caricata nel database.

## Preparazione

- Nel riepilogo Upload verificare che tutti i 100 file risultino inseriti o già
  presenti; eventuali errori vanno risolti prima di presentare un corpus completo.
- Aprire Query, scegliere una delle cinque domande nel campo Domanda e rimuovere
  i filtri metadata attivi. Domanda libera consente di scrivere una richiesta propria.
- Lasciare vuote le istruzioni al reranker e mantenere invariati gli altri
  parametri per tutte le domande. Annotare modello e impostazioni se modificati.
- Selezionare una domanda per volta e premere Esegui. Per la domanda libera incollare solo la domanda,
  senza risposta attesa, suggerimenti sul trabocchetto o riferimenti di questa scheda.

Le pagine sotto sono quelle del PDF contando da 1; nelle tre pagine di riferimento
coincidono con i numeri stampati. Una diversa fonte o pagina è valida se supporta
effettivamente lo stesso dato per azienda, anno, voce e unità richiesti.
Il numero [#N] indica un passaggio della risposta: non deve coincidere con il
numero di pagina e può cambiare tra esecuzioni.

## 1. Valore esplicito in tabella

> Quali sono state le spese in conto capitale di 3M nel 2018, in milioni di dollari? Usa il rendiconto finanziario.

- **Atteso:** 1.577 milioni di dollari.
- **Fonte:** [3M_2018_10K.pdf](data/financebench_100/pdfs/3M_2018_10K.pdf), pagina 60 del PDF.
- **Evidenza:** tabella Cash Flows from Investing Activities, riga Purchases of
  property, plant and equipment (PP&E), colonna 2018: (1,577).
- Le parentesi indicano un’uscita di cassa; è corretta sia la spesa positiva
  di 1.577 milioni sia un flusso di -1.577 milioni spiegato come uscita.

## 2. Domanda in italiano, voce contabile in inglese

> Quanto ha speso Adobe per ricerca e sviluppo nell’esercizio 2022? Indica l’importo in milioni di dollari.

- **Atteso:** 2.987 milioni di dollari.
- **Fonte:** [ADOBE_2022_10K.pdf](data/financebench_100/pdfs/ADOBE_2022_10K.pdf), pagina 54.
- **Evidenza:** Consolidated Statements of Income, riga Research and development,
  colonna December 2, 2022: 2,987. Non confondere con Sales and marketing (4,968).
- Questo caso verifica il recupero tra lingue e terminologie; da solo non prova
  che il risultato dipenda esclusivamente dalla ricerca vettoriale.

## 3. Due dati dello stesso segmento

> Quali sono stati i ricavi e l’utile operativo del segmento AWS di Amazon nel 2022? Riporta entrambi in milioni di dollari.

- **Atteso:** ricavi 80.096 milioni; utile operativo 22.841 milioni di dollari.
- **Fonte:** [AMAZON_2022_10K.pdf](data/financebench_100/pdfs/AMAZON_2022_10K.pdf), pagina 66.
- **Evidenza:** Note 10 — Segment Information, sezione AWS, colonna 2022,
  righe Net sales (80,096) e Operating income (22,841).
- Devono essere presenti entrambi i valori. Non usare i ricavi consolidati
  Amazon (513,983) o l’utile operativo consolidato (12,248).

## 4. Trabocchetto: cambia l’anno

> Quali sono state le spese in conto capitale di 3M nel 2017, in milioni di dollari? Usa il rendiconto finanziario.

- **Atteso:** 1.373 milioni di dollari.
- **Fonte:** [3M_2018_10K.pdf](data/financebench_100/pdfs/3M_2018_10K.pdf), pagina 60 del PDF.
- **Evidenza:** stessa riga della domanda 1, ma colonna 2017: (1,373).
- **Trabocchetto:** azienda e metrica identiche alla domanda 1; cambiare anno
  deve cambiare il risultato. Rispondere 1.577 significa usare il dato 2018.
- È legittimo citare il bilancio 2018 per un dato 2017: contiene la colonna
  comparativa. Conta l’anno del dato nella tabella, non solo quello nel filename.

## 5. Trabocchetto: premessa falsa

> Di quanto è aumentato l’utile netto di Adobe tra il 2021 e il 2022? Indica la variazione in milioni di dollari.

- **Atteso:** non è aumentato; è diminuito di 66 milioni di dollari, da 4.822
  milioni nel 2021 a 4.756 milioni nel 2022.
- **Fonte:** [ADOBE_2022_10K.pdf](data/financebench_100/pdfs/ADOBE_2022_10K.pdf), pagina 54.
- **Evidenza:** riga Net income, colonne 2021 (4,822) e 2022 (4,756).
  Calcolo: 4.756 - 4.822 = -66 milioni.
- **Trabocchetto:** correggere esplicitamente la premessa di aumento. Non usare
  Operating income, che passa invece da 5,802 a 6,098.

## Verifica durante la demo

Aprire il PDF citato dalla risposta e controllare riga, colonna e unità. Se la
citazione rimanda a una pagina diversa, verificare quella pagina prima di
assegnare un errore: gli stessi dati possono comparire in più punti o bilanci.

| Query | Risposta corretta / parziale / errata | Citazione supporta la risposta: sì / no / assente | Note |
|---|---|---|---|
| 1 | Alternante: successi e fallimenti osservati | Citazione p.39 nelle prove riuscite; rendiconto p.46 escluso dal Top-8 | Confronto locale su contesto fisso distinto dalle esecuzioni globali. Vedi DEMO_PLAN.md. |
| 2 | Corretta secondo la segnalazione dell’utente | Non verificata direttamente nell’ultima esecuzione dall’assistente | Non equivale a un collaudo ripetuto. |
| 3 | Corretta secondo la segnalazione dell’utente | Non verificata direttamente nell’ultima esecuzione dall’assistente | Non equivale a un collaudo ripetuto. |
| 4 | Corretta secondo la segnalazione dell’utente | Non verificata direttamente nell’ultima esecuzione dall’assistente | Non equivale a un collaudo ripetuto. |
| 5 | Errata: +476 invece di −66 milioni (20 ottobre) | Assente: URL esterno, nessuna citazione interna valida | Valori corretti presenti nei passaggi #1–#3. Correzioni successive non ancora verificate con una nuova risposta. |

La formulazione può cambiare: contano fatti, numeri, anno, soggetto e unità.
Una risposta corretta con fonte errata non è una prova pienamente riuscita.
Se il modello dichiara di non avere abbastanza informazioni, la prova non è
riuscita: controllare prima se il passaggio atteso è stato recuperato e poi se
è arrivato al generatore. Non modificare i valori attesi per adattarli all’output.
Cinque prove descrivono questi cinque casi, non l’accuratezza generale del sistema.
