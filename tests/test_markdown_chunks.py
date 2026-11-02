"""Chunk che rispettano la struttura del Markdown prodotto da pymupdf4llm."""
from src import ingestion


def test_tabella_lunga_ripete_intestazione_e_titolo():
    rows = "\n".join(f"|Voce numero {i} con una descrizione abbastanza lunga|{i},000|{i},100|" for i in range(60))
    text = "### Cash Flows from Investing Activities\n\n|(Millions)|2018|2017|\n|---|---|---|\n" + rows
    chunks = ingestion._chunk_markdown(text, 120, 10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.startswith("### Cash Flows from Investing Activities")
        assert "|(Millions)|2018|2017|" in c and "|---|---|---|" in c
    # nessuna riga persa o spezzata
    body = [l for c in chunks for l in c.splitlines() if l.startswith("|Voce")]
    assert len(body) == 60


def test_paragrafi_e_titoli():
    text = "## Properties\n\nThe Company operates 75 manufacturing facilities in 29 states.\n\n## Legal\n\nNothing."
    chunks = ingestion._chunk_markdown(text, 512, 64)
    assert chunks == [text]


def test_pulizia_markdown():
    assert ingestion._clean_markdown("### **_Cash Flows:_**\n|**$**<br>**(1,577)**|") == "### Cash Flows:\n|$ (1,577)|"


def test_anni_dopo_il_separatore_ripetuti():
    rows = "\n".join(f"|Voce numero {i} con una descrizione abbastanza lunga|{i},000|{i},100|" for i in range(60))
    text = "|Years Ended December 31|||\n|---|---|---|\n|(Millions)|2018|2017|\n" + rows
    chunks = ingestion._chunk_markdown(text, 120, 10)
    assert len(chunks) > 1 and all("|(Millions)|2018|2017|" in c for c in chunks)
