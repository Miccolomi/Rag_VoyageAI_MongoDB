/* RAG Demo · MongoDB Atlas — interfaccia web senza framework. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = (ms) => (!ms ? "" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1).replace(".", ",")} s`);
const shortFile = (f) => String(f || "Documento").replace(/\.(pdf|docx?|pptx?|xlsx?)$/i, "");
const pageLabel = (p) => (p === null || p === undefined || p === "" ? "pagina n/d" : `p. ${p}`);
const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };

// La sessione vive finché la pagina è aperta: a ogni apertura si riparte da una sessione nuova.
const state = { info: null, sessionId: null, runId: null, busy: false };
const fmtDate = (iso) => { if (!iso) return ""; const d = new Date(/Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
  return d.toLocaleString("it-IT", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" }); };
const lowerFirst = (t) => t ? t[0].toLowerCase() + t.slice(1) : "";

// ─── API ────────────────────────────────────────────────────────────────
async function getJSON(url) {
  const r = await fetch(url);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}
async function sendJSON(url, body, method = "POST") {
  const r = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}
/** Flusso NDJSON: un evento per riga, consegnato appena arriva. */
async function stream(url, body, onEvent) {
  const isForm = body instanceof FormData;
  const r = await fetch(url, { method: "POST", body: isForm ? body : JSON.stringify(body),
    headers: isForm ? {} : { "Content-Type": "application/json" } });
  if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error || `HTTP ${r.status}`); }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while ((i = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, i).trim();
      buffer = buffer.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

// ─── Testo della risposta: markdown minimo + citazioni ─────────────────
function renderAnswer(text) {
  const lines = String(text || "").split("\n");
  let html = "", list = false;
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\[#(\d+)\]/g, '<span class="cite">$1</span>');
  for (const raw of lines) {
    const line = raw.trim();
    const bullet = line.match(/^[*\-•]\s+(.*)$/);
    if (bullet) { if (!list) { html += "<ul>"; list = true; } html += `<li>${inline(bullet[1])}</li>`; continue; }
    if (list) { html += "</ul>"; list = false; }
    if (line) html += `<p>${inline(line)}</p>`;
  }
  if (list) html += "</ul>";
  return html;
}

// ─── Intestazione ──────────────────────────────────────────────────────
async function loadInfo() {
  try {
    state.info = await getJSON("/api/info");
    const i = state.info;
    $("#meta").textContent = `DB ${i.database} · ${i.collection} · modalità attiva ${i.active_mode} · ${i.embed_model} · rerank ${i.rerank} · LLM ${i.llm}`;
    document.dispatchEvent(new Event("rag:info"));
  } catch (e) {
    $("#meta").textContent = "Server non raggiungibile: " + e.message;
  }
}

// ─── Verifica dei componenti (pill in alto) ───────────────────────────
let health = null;
async function loadHealth(force = false) {
  const pill = $("#status-pill");
  pill.className = "pill"; pill.textContent = "● Verifica in corso…";
  renderHealthPanel(true);
  try {
    health = await getJSON("/api/health" + (force ? "?force=1" : ""));
    const failed = health.checks.filter((c) => !c.ok);
    pill.className = "pill " + (health.ok ? "ok" : "bad");
    pill.textContent = health.ok ? "● Tutto operativo" : `● ${failed.length === 1 ? failed[0].name + " non disponibile" : failed.length + " controlli non superati"}`;
  } catch (e) {
    health = { ok: false, checks: [{ name: "Server della demo", what: "Risposta del server", ok: false, detail: e.message }] };
    pill.className = "pill bad"; pill.textContent = "● Server non raggiungibile";
  }
  renderHealthPanel(false);
}
function renderHealthPanel(loading) {
  const panel = $("#health");
  if (!panel || panel.hidden) return;
  const rows = (health?.checks || []).map((c) => `<div class="hc ${c.ok ? "ok" : "ko"}"><span class="hc-i">${c.ok ? "✓" : "✗"}</span>
      <div><b>${esc(c.name)}</b> <span class="muted small">${esc(c.what)}</span><div class="small">${esc(c.detail || "")}</div></div>
      <span class="muted small">${c.ms != null ? fmtMs(c.ms) || "< 1 ms" : ""}</span></div>`).join("");
  const when = health?.checked_at ? new Date(health.checked_at).toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
  panel.innerHTML = `<div class="row"><b>Cosa ho verificato</b><span class="spacer"></span><button class="small" id="hc-again" ${loading ? "disabled" : ""}>${loading ? "Verifica in corso…" : "↻ Ricontrolla"}</button></div>
    <p class="muted small" style="margin:4px 0 8px">Ogni controllo chiama davvero il componente: una query in Atlas, una risposta del modello.${when ? ` Ultima verifica alle ${when}; vale 5 minuti.` : ""}</p>${rows || (loading ? "<p class='muted'>…</p>" : "")}`;
  $("#hc-again", panel).onclick = () => loadHealth(true);
}
document.addEventListener("click", (e) => {
  const panel = $("#health");
  if (!panel) return;
  if (e.target.closest("#status-pill")) { panel.hidden = !panel.hidden; renderHealthPanel(false); return; }
  if (!panel.hidden && !e.target.closest("#health")) panel.hidden = true;
});

// ─── Navigazione ───────────────────────────────────────────────────────
const pages = {};
function route() {
  const name = (location.hash || "#mongodb").slice(1);
  const page = pages[name] ? name : "mongodb";
  $$("#tabs a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  const main = $("#page");
  main.innerHTML = "";
  pages[page](main);
}
window.addEventListener("hashchange", route);

// ─── Pagina: Come funziona ─────────────────────────────────────────────
pages.come = (main) => {
  main.append($("#tpl-come").content.cloneNode(true));
  const frame = $(".arch-frame", main);
  if (frame) frame.addEventListener("load", () => {
    try {
      const doc = frame.contentDocument;
      // Nella demo niente rimandi esterni: si nasconde il link promozionale del visualizzatore.
      doc.head.insertAdjacentHTML("beforeend", "<style>.diagram-guide-credit{display:none!important}</style>");
      // Il visualizzatore occupa almeno l'altezza della finestra: si misura il contenuto reale.
      const fit = () => {
        const bottom = Math.max(...[...doc.body.children].map((el) => el.getBoundingClientRect().bottom + doc.defaultView.scrollY));
        if (bottom > 200) frame.style.height = Math.ceil(bottom + 24) + "px";
      };
      fit(); setTimeout(fit, 600);
    } catch { /* stessa origine: non dovrebbe succedere */ }
  });
};

// ─── Pagina: MongoDB dettagli ──────────────────────────────────────────
pages.mongodb = (main) => {
  main.append($("#tpl-mongo").content.cloneNode(true));
  const fill = () => {
    const i = state.info; if (!i) return;
    const chunks = Object.values(i.counts || {}).reduce((a, b) => a + b, 0);
    const set = (k, v) => { const el = $(`[data-stat="${k}"]`, main); if (el) el.textContent = v; };
    set("chunks", chunks.toLocaleString("it-IT"));
    set("indexes", String((i.indexes || []).length + 1));
    for (const el of $$("[data-model]", main)) { const m = i.models?.[el.dataset.model]; if (m) el.textContent = m; }
    const code = $(".mx-code", main);
    if (code && i.models?.rerank) code.textContent = code.textContent.replace('model: "rerank"', `model: "${i.models.rerank}"`);
  };
  fill();
  document.addEventListener("rag:info", fill, { once: true });
  // Collega le parti del documento alle capacità che alimentano.
  const doc = $(".mx-doc", main);
  const mark = (f) => $$("[data-f]", doc).forEach((el) => el.classList.toggle("on", !!f && el.dataset.f === f));
  $$(".mx-dc, .mx-json [data-f]", doc).forEach((el) => {
    el.addEventListener("mouseenter", () => mark(el.dataset.f));
    el.addEventListener("mouseleave", () => mark(null));
  });
  // Le capacità entrano in scena quando arrivano sullo schermo.
  const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.15 });
  $$(".mx-p, .mx-arch-col, .mx-ps", main).forEach((el) => io.observe(el));
  loadLiveProtection(main);
};

async function loadLiveProtection(main) {
  const box = $("#mx-csfle-live", main);
  if (!box) return;
  try {
    const v = await getJSON("/api/protection/latest");
    box.innerHTML = v.empty
      ? `<p class="muted">Ancora nessuna sessione cifrata: fai una domanda nella Ricerca agentica e torna qui per vedere il confronto dal vivo.</p>`
      : `<div class="cs-live-tag">● dal vivo · ultima sessione della Ricerca agentica</div>` + protectionCompare(v);
  } catch { box.innerHTML = `<p class="muted">Confronto dal vivo non disponibile: Atlas non risponde.</p>`; }
}

// ─── Componenti condivisi ──────────────────────────────────────────────
function questionBox({ placeholder, button, onSubmit, settingsHtml = "" }) {
  const el = h(`<div class="card ask">
      <textarea id="q" placeholder="${esc(placeholder)}"></textarea>
      <div class="chips" id="demo"></div>
      <div class="ask-actions">
        <button class="primary" id="go">${esc(button)}</button>
        <span class="muted small" id="elapsed"></span>
        <span class="spacer"></span>
      </div>
      ${settingsHtml ? `<details><summary>Impostazioni della ricerca</summary><div class="settings">${settingsHtml}</div></details>` : ""}
    </div>`);
  const q = $("#q", el);
  const fillChips = () => {
    const box = $("#demo", el);
    if (box.childElementCount) return;
    for (const text of state.info?.demo_questions || []) {
      const c = h(`<span class="chip">${esc(text.length > 70 ? text.slice(0, 68) + "…" : text)}</span>`);
      c.title = text;
      c.onclick = () => { q.value = text; q.focus(); };
      box.append(c);
    }
  };
  fillChips();
  document.addEventListener("rag:info", fillChips);
  const submit = () => { const v = q.value.trim(); if (v && !state.busy) onSubmit(v); };
  $("#go", el).onclick = submit;
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); });
  return el;
}

function elapsedTimer(el) {
  const start = performance.now();
  const id = setInterval(() => { el.textContent = `${((performance.now() - start) / 1000).toFixed(1).replace(".", ",")} s`; }, 100);
  return () => clearInterval(id);
}

function errorBox(msg) { return h(`<div class="error">${esc(msg)}</div>`); }

function verdictBadge(kind, label) { return `<span class="badge ${esc(kind)}">${kind === "ok" ? "✓" : kind === "risk" ? "⚠" : "○"} ${esc(label)}</span>`; }

function citationRows(rows, sources) {
  if (!rows?.length) return `<p class="muted small">La risposta non contiene importi o percentuali da verificare.</p>`;
  const labels = { found: "✓ trovato", rounded: "≈ arrotondato", not_found: "⚠ non trovato", uncited: "senza citazione" };
  return rows.map((r) => {
    const src = r.ref ? sources[r.ref - 1] : null;
    const where = src ? `#${r.ref} · ${esc(shortFile(src.filename))} · ${esc(pageLabel(src.page ?? src.page_number))}` : "Nessun passaggio citato";
    const ex = r.excerpt ? `“${esc(r.excerpt.before)}<mark>${esc(r.excerpt.match)}</mark>${esc(r.excerpt.after)}”` :
      r.status === "not_found" ? "Il numero non compare nel passaggio citato." : "";
    return `<div class="check-row"><b>${esc(r.claim)}</b><div><div class="muted">${where}</div><div>${ex}</div></div>
            <span class="tag ${esc(r.status)}">${labels[r.status] || r.status}</span></div>`;
  }).join("");
}

function sourceLine(n, src, matches = []) {
  const file = src.gridfs_file_id ? `<a href="/api/files/${esc(src.gridfs_file_id)}" target="_blank">PDF</a>` : "";
  const m = matches.map((x) => `<mark>${esc(x)}</mark>`).join(" ");
  return `<div class="source"><span class="num">${n}</span><span>${esc(shortFile(src.filename))} · ${esc(pageLabel(src.page ?? src.page_number))}</span> ${m} <span class="spacer"></span>${file}</div>`;
}

// ─── Grafico a corsie (posizioni prima → dopo) ─────────────────────────
const PALETTE = ["#00684a", "#6e5bd6", "#d0632e", "#1f7ab8", "#b83280", "#5f7d0f", "#8a6d00", "#2b8a8a"];
/**
 * Una colonna: sorgente a sinistra, classifica finale a destra (reranking).
 * Due colonne: motore testuale a sinistra, semantico a destra, classifica fusa al centro (fusione).
 */
function lanes(container, { columns, target, items, cut = null }) {
  const anyBeyond = items.some((it) => columns.some((c) => (it.ranks[c.key] || 0) > 10));
  const anyMissing = items.some((it) => columns.some((c) => !it.ranks[c.key]));
  const maxRank = Math.max(items.length, ...items.flatMap((it) => columns.map((c) => Math.min(it.ranks[c.key] || 0, 10))));
  const rows = maxRank + (anyBeyond ? 1 : 0) + (anyMissing ? 1 : 0);
  const W = 960, rowH = 30, top = 46, H = top + rows * rowH + 10;
  const mirrored = columns.length === 2;
  const y = (rank) => {
    const r = !rank ? maxRank + (anyBeyond ? 2 : 1) : rank > 10 ? maxRank + 1 : rank;
    return top + (r - 1) * rowH + rowH / 2;
  };
  const colX = mirrored ? [40, W - 40] : columns.map((_, i) => 40 + i * 90);
  const box = mirrored ? { l: 300, r: 660 } : { l: colX[colX.length - 1] + 280, r: W };
  const files = [...new Set(items.map((i) => i.file))];
  const color = (f) => PALETTE[files.indexOf(f) % PALETTE.length];
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let r = 0; r < rows; r++) if (r % 2 === 0) svg += `<rect x="0" y="${top + r * rowH}" width="${W}" height="${rowH}" fill="currentColor" opacity="0.035"/>`;
  const head = (x, anchor, c) => `<text x="${x}" y="18" text-anchor="${anchor}" font-size="13" font-weight="600" fill="currentColor">${esc(c.title)}</text>` +
    `<text x="${x}" y="34" text-anchor="${anchor}" font-size="11" fill="currentColor" opacity=".6">${esc(c.sub || "")}</text>`;
  columns.forEach((c, i) => svg += head(colX[i], "middle", c));
  svg += head(mirrored ? (box.l + box.r) / 2 : box.l + 12, mirrored ? "middle" : "start", target);
  columns.forEach((_, i) => {
    for (let r = 1; r <= maxRank; r++) svg += `<text x="${colX[i]}" y="${y(r) + 4}" text-anchor="middle" font-size="11" fill="currentColor" opacity=".45">#${r}</text>`;
    if (anyBeyond) svg += `<text x="${colX[i]}" y="${y(11) + 4}" text-anchor="middle" font-size="10.5" fill="currentColor" opacity=".5">oltre #10</text>`;
    if (anyMissing) svg += `<text x="${colX[i]}" y="${y(null) + 4}" text-anchor="middle" font-size="10.5" fill="currentColor" opacity=".5">assente</text>`;
  });
  if (cut) svg += `<line x1="${box.l}" x2="${W}" y1="${top + cut * rowH}" y2="${top + cut * rowH}" stroke="currentColor" stroke-dasharray="5 4" opacity=".5"/><text x="${W - 4}" y="${top + cut * rowH - 4}" text-anchor="end" font-size="10.5" fill="currentColor" opacity=".7">sopra la linea: inviati al modello</text>`;
  items.forEach((it, k) => {
    const c = color(it.file), yt = y(it.final);
    columns.forEach((col, i) => {
      const r = it.ranks[col.key];
      const y0 = y(r);
      const right = mirrored && i === 1;
      if (!mirrored && i < columns.length - 1) { svg += `<circle cx="${colX[i] + 22}" cy="${y0}" r="4" fill="${c}" opacity="${r ? 1 : 0.35}"/>`; return; }
      const x0 = right ? colX[i] - 26 : colX[i] + 26, x1 = right ? box.r + 4 : box.l - 4;
      const mid = (x0 + x1) / 2;
      svg += `<path d="M${x0},${y0} C${mid},${y0} ${mid},${yt} ${x1},${yt}" fill="none" stroke="${c}" stroke-width="${it.cited ? 2.6 : 1.6}" opacity="${r ? 0.8 : 0.25}" stroke-dasharray="${r ? "" : "4 4"}" style="animation:rise .5s ${0.05 * k}s both"/>`;
      svg += `<circle cx="${x0}" cy="${y0}" r="4" fill="${c}" opacity="${r ? 1 : 0.35}"/>`;
      svg += `<circle cx="${x1}" cy="${yt}" r="4.5" fill="${c}"/>`;
    });
    const delta = it.delta ? ` <tspan fill="${it.delta > 0 ? "#0f8a5f" : "#c2640c"}">${it.delta > 0 ? "▲" : "▼"}${Math.abs(it.delta)}</tspan>` : "";
    const tx = mirrored ? (box.l + box.r) / 2 : box.l + 12;
    const sub = mirrored ? "" : `<tspan opacity=".55"> ${esc(it.sub)}</tspan>`;
    svg += `<text x="${tx}" y="${yt + 4}" text-anchor="${mirrored ? "middle" : "start"}" font-size="12" fill="currentColor"><tspan font-weight="700">#${it.final}</tspan> ${esc(it.label)}${it.cited ? " ✓" : ""}${delta}${sub}</text>`;
  });
  svg += "</svg>";
  container.innerHTML = `<div class="lanes">${svg}</div>`;
}

// ─── Cifratura lato client: lo stesso campo visto dall'app e da Atlas ──
function protectionCompare(v) {
  const sv = v.stored_view || {};
  const iso = (d) => (d ? `ISODate("${esc(String(d).slice(0, 19))}Z")` : "null");
  const shortId = esc(String(v.id || "").slice(0, 8)) + "…";
  // Lo stesso documento, solo i campi utili al confronto: come lo mostra Compass.
  const doc = (fieldLine) => `<span class="j-p">{</span>
  <span class="j-k">_id</span>: <span class="j-s">"${shortId}"</span>,
  <span class="j-k">title</span>: <span class="j-s">"${esc(v.title || "")}"</span>,
  <span class="j-k">created_at</span>: <span class="j-d">${iso(v.created_at)}</span>,
  <span class="j-k">memory</span>: { <span class="j-k">turn_count</span>: <span class="j-n">${esc(v.turns ?? 0)}</span>, <span class="j-c">…</span> },
${fieldLine}<span class="j-p">}</span>`;
  const stored = v.encrypted
    ? `<span class="j-hl"><span class="j-k">${esc(v.field)}</span>: <span class="j-b">Binary.createFromBase64("${esc(sv.base64)}", <b>6</b>)</span></span>`
    : `<span class="j-hl warn"><span class="j-k">${esc(v.field)}</span>: <span class="j-s">"${esc(sv.value ?? "")}"</span></span>`;
  const readable = v.app_view == null
    ? `<span class="j-hl warn"><span class="j-k">${esc(v.field)}</span>: <span class="j-c">// chiave non disponibile su questo computer</span></span>`
    : `<span class="j-hl"><span class="j-k">${esc(v.field)}</span>: <span class="j-s">"${esc(v.app_view)}"</span></span>`;
  const atlasNote = v.encrypted
    ? `Il campo è un dato binario di ${esc(sv.bytes)} byte (sottotipo 6 = cifrato lato client). Così lo vedono Compass, un backup, un amministratore del cluster.`
    : v.encryption_error
      ? `Cifratura non riuscita sul server che ha salvato questa sessione, quindi il campo è in chiaro anche in Atlas. Motivo: ${esc(v.encryption_error)}`
      : `Sessione salvata senza cifratura: prima che fosse attiva, o da un server dove non funzionava. Il campo è in chiaro anche in Atlas. Prova con una nuova sessione.`;
  return `<div class="cs-flow">
      <div><b>1 · Scrivi la domanda</b><span>testo in chiaro nell’applicazione</span></div>
      <div class="cs-step-key"><b>2 · 🔑 Il driver la cifra</b><span>AES-256 con la chiave dati, prima di spedirla</span></div>
      <div class="cs-step-db"><b>3 · Atlas salva il cifrato</b><span>disco, repliche, backup: solo byte illeggibili</span></div>
      <div class="cs-step-key"><b>4 · 🔑 L’app la decifra</b><span>quando rilegge la sessione, solo dove c’è la chiave</span></div>
    </div>
    <div class="cs-compare">
      <div class="cs-col cs-atlas">
        <div class="cs-head">🔒 Il documento in Atlas</div>
        <pre class="cs-json">${doc(stored)}</pre>
        <p class="cs-note">${atlasNote}</p>
      </div>
      <div class="cs-arrow" aria-hidden="true">🔑</div>
      <div class="cs-col cs-app">
        <div class="cs-head">Lo stesso documento letto dall’applicazione</div>
        <pre class="cs-json light">${doc(readable)}</pre>
        <p class="cs-note">Il driver decifra il campo con la chiave dati. Gli altri campi sono identici: non erano cifrati.</p>
      </div>
    </div>
    <p class="small muted cs-foot">Sessione del ${esc(fmtDate(v.created_at))}. Algoritmo casuale: la stessa domanda dà ogni volta un cifrato diverso.
      La chiave dati <code>${esc(v.key_alt_name)}</code> sta nel key vault <code>${esc(v.key_vault)}</code>, a sua volta cifrata con la chiave master,
      che Atlas non ha. Titolo, date e contatori restano in chiaro: servono per elenchi e filtri e non sono sensibili.</p>`;
}

// ─── Pagina: Ricerca agentica ──────────────────────────────────────────
pages.agente = async (main) => {
  main.append(h(`<div><h1>Ricerca agentica</h1>
    <p class="muted">L’agente cerca, controlla che le prove coprano ogni dato richiesto, cerca di nuovo ciò che manca e ti chiede quando i documenti danno valori diversi. Ogni passo è salvato in MongoDB.</p></div>`));
  const bar = h(`<div class="row" style="margin-bottom:12px">
      <label style="margin:0">Sessione</label>
      <select id="session" style="max-width:320px"></select>
      <button id="new-session">＋ Nuova sessione</button>
      <button id="show-protection" title="Lo stesso documento visto dall’applicazione e da Atlas">🔒 Come lo vede Atlas</button>
      <span class="spacer"></span>
      <select id="runs" style="max-width:300px"><option value="">Ricerche di questa sessione</option></select>
    </div>`);
  main.append(bar);
  const box = questionBox({
    placeholder: "Fai una domanda sui documenti… (⌘↵ per avviare)",
    button: "Avvia la ricerca agentica",
    onSubmit: (q) => runAgent(main, { question: q }),
    settingsHtml: `
      <div><label>Candidati per ricerca</label><input type="number" id="s-topk" value="20" min="5" max="100"></div>
      <div><label>Fonti per la risposta</label><input type="number" id="s-topn" value="8" min="1" max="20"></div>
      <div><label>Ricerche aggiuntive</label><input type="number" id="s-retries" value="1" min="0" max="2"></div>
      <div style="grid-column:1/-1"><label>Indicazioni per ordinare le fonti (facoltative)</label><input type="text" id="s-instr" placeholder="Es. privilegia il rendiconto finanziario"></div>`,
  });
  main.append(box);
  main.append(h(`<div id="protection" class="card cs-card" hidden></div>`));
  main.append(h(`<div id="live"></div>`));
  main.append(h(`<div id="result"></div>`));
  await loadSessions(main);
  $("#new-session", bar).onclick = async () => {
    const s = await sendJSON("/api/sessions", {});
    state.sessionId = s.id;
    $("#live", main).innerHTML = ""; $("#result", main).innerHTML = ""; $("#protection", main).hidden = true;
    await loadSessions(main);
  };
  $("#session", bar).onchange = async (e) => {
    state.sessionId = e.target.value;
    $("#live", main).innerHTML = ""; $("#result", main).innerHTML = ""; $("#protection", main).hidden = true;
    await loadRuns(main, true);
  };
  $("#runs", bar).onchange = (e) => { if (e.target.value) showRun(main, e.target.value); };
  $("#show-protection", bar).onclick = () => toggleProtection(main);
};

async function toggleProtection(main, refresh = false) {
  const panel = $("#protection", main);
  if (!refresh && !panel.hidden) { panel.hidden = true; return; }
  panel.hidden = false;
  panel.innerHTML = `<p class="muted">Lettura del documento da Atlas…</p>`;
  try {
    const v = await getJSON(`/api/sessions/${state.sessionId}/protection`);
    const head = `<div class="row"><h3 style="margin:0">La prima domanda è cifrata prima di arrivare in Atlas</h3><span class="spacer"></span>
      <button class="small" id="close-protection">Chiudi</button></div>`;
    panel.innerHTML = head + (v.has_question ? protectionCompare(v)
      : `<p class="muted">Questa sessione non ha ancora domande. Fai una domanda: appena l’agente risponde, la prima domanda viene salvata cifrata e qui vedi il confronto.</p>`);
    $("#close-protection", panel).onclick = () => { panel.hidden = true; };
  } catch (e) { panel.innerHTML = ""; panel.append(errorBox(e.message)); }
}

async function loadSessions(main) {
  const resume = !!state.sessionId;
  if (!resume) {
    const s = await sendJSON("/api/sessions", {});
    state.sessionId = s.id;
  }
  await refreshSessionList(main);
  // Appena aperta la pagina: sessione nuova e pagina pulita. Tornando sulla scheda: riprende da dove eri.
  await loadRuns(main, resume);
}

async function refreshSessionList(main) {
  const list = await getJSON("/api/sessions");
  $("#session", main).innerHTML = list.map((s) => {
    const label = `${s.encrypted ? "🔒 " : ""}${fmtDate(s.created_at)} · ${s.turns ? `${s.turns} ${s.turns === 1 ? "domanda" : "domande"} · ${(s.first_question || "").slice(0, 50)}` : "nuova"}`;
    return `<option value="${esc(s.id)}" ${s.id === state.sessionId ? "selected" : ""}>${esc(label)}</option>`;
  }).join("");
}

async function loadRuns(main, showLatest = false) {
  const runs = await getJSON(`/api/sessions/${state.sessionId}/runs`);
  const sel = $("#runs", main);
  sel.innerHTML = `<option value="">Ricerche di questa sessione (${runs.length})</option>` +
    runs.map((r) => `<option value="${esc(r.id)}">${esc((r.created_at || "").slice(11, 16))} · ${esc((r.question || "").slice(0, 70))}</option>`).join("");
  if (showLatest && runs.length && !state.busy) await showRun(main, runs[0].id);
}

const STEP_LABELS = {
  start: "Interpreta la domanda", plan: "Cerca nei documenti con Atlas", retrieval: "Controlla le prove",
  agent_rank_fusion: "Controlla le prove", grading: "Decide il prossimo passo", rewrite: "Ricerca mirata in Atlas",
  generating: "Scrive la risposta citando le fonti",
};

function timelineRow(icon, title, detail = "", ms = null) {
  return h(`<div class="tl-row"><span class="tl-icon">${icon}</span><div><div class="tl-title">${esc(title)}</div>${detail ? `<div class="tl-detail">${esc(detail)}</div>` : ""}</div><span class="tl-time">${ms ? fmtMs(ms) : ""}</span></div>`);
}

function eventRow(ev) {
  const top3 = (refs) => (refs || []).slice(0, 3).map((r) => `${shortFile(r.filename)} ${pageLabel(r.page)}`).join(" · ");
  switch (ev.type) {
    case "plan":
      if (ev.planner_skipped) return timelineRow("🧭", "Domanda usata così com’è", ev.parent_run_id ? "Risposta alla tua scelta: nessun riferimento da risolvere" : "Prima domanda della sessione: nessun riferimento da risolvere");
      if (ev.memory?.turns) return timelineRow("🍃", `Ha letto la conversazione da MongoDB: ${ev.memory.turns} ${ev.memory.turns === 1 ? "domanda precedente" : "domande precedenti"}`,
        `Ha trasformato la domanda in «${ev.planned_query}»`, ev.duration_ms);
      return timelineRow("🧭", "Ha interpretato la domanda", `«${ev.planned_query}»`, ev.duration_ms);
    case "retrieval":
      return timelineRow("🔍", ev.iteration > 1 ? `Ricerca mirata: «${ev.search_query}»` : `Ricerca ibrida in Atlas · ${ev.candidate_count} candidati`,
        "In cima: " + top3(ev.ranked_sources), ev.duration_ms);
    case "agent_rank_fusion":
      return timelineRow("🔗", `Unisce le ricerche: ${ev.unique_candidates} passaggi distinti`, "Riordinati rispetto alla domanda completa", ev.duration_ms);
    case "grading": {
      const els = ev.elements || [];
      const found = els.filter((e) => e.found);
      const missing = els.filter((e) => !e.found).map((e) => e.element);
      if (ev.evaluated === false) return timelineRow("⚠️", "Valutazione non riuscita", ev.failure_detail || ev.failure_reason || "", ev.duration_ms);
      return timelineRow(ev.sufficient ? "✅" : "⚠️",
        ev.sufficient ? `Prove sufficienti: ${found.length} dati su ${els.length || found.length}` : `Manca: ${missing.join("; ") || "qualcosa"}`,
        found.map((e) => `${e.element}${e.value ? " = " + e.value : ""}`).join(" · "), ev.duration_ms);
    }
    case "clarification":
      return timelineRow("❓", "Valori diversi in anni diversi: ti chiede quale anno", (ev.options || []).map((o) => o.year).join(", "));
    case "decision":
      if (ev.reason === "off_topic") return timelineRow("🚫", "Fuori tema: si ferma senza chiamare il modello", `miglior passaggio ${Number(ev.top_rerank).toFixed(3)} < soglia ${ev.threshold}`);
      if (ev.reason === "no_results") return timelineRow("🚫", "Nessun passaggio trovato");
      return null;
    case "answer":
      if (ev.status === "needs_input") return null;
      return timelineRow(ev.status === "blocked" ? "🚫" : "💬", ev.status === "blocked" ? "Risposta non pubblicata" : `Risposta scritta con ${ev.sources?.length || 0} fonti`, "", ev.duration_ms);
    default: return null;
  }
}

async function runAgent(main, { question, clarification = null }) {
  state.busy = true;
  const go = $("#go", main); go.disabled = true;
  const live = $("#live", main), result = $("#result", main);
  result.innerHTML = "";
  live.innerHTML = "";
  const card = h(`<div class="card live"><div class="row"><b>L’agente al lavoro</b><span class="spacer"></span><span class="muted small" id="live-clock"></span></div>
      <div class="question-echo">${esc(question)}</div><div class="timeline" id="tl"></div></div>`);
  live.append(card);
  card.scrollIntoView({ behavior: "smooth", block: "start" });
  const tl = $("#tl", card);
  const stop = elapsedTimer($("#live-clock", card));
  let pending = timelineRow("⏳", STEP_LABELS.start); pending.classList.add("running"); tl.append(pending);
  const setPending = (label) => {
    pending.remove();
    if (!label) return;
    pending = timelineRow("⏳", label); pending.classList.add("running"); tl.append(pending);
  };
  const s = (id) => Number($(id, main)?.value) || undefined;
  let runId = null, failed = null;
  try {
    await stream("/api/agent", {
      session_id: state.sessionId, question, clarification,
      top_k: s("#s-topk"), top_n: s("#s-topn"), retries: Number($("#s-retries", main)?.value ?? 1),
      rerank_instructions: $("#s-instr", main)?.value || "",
    }, (ev) => {
      if (ev.type === "session") { state.sessionId = ev.session_id; return; }
      if (ev.type === "start") { runId = ev.run_id; return; }
      if (ev.type === "error") { failed = ev.message; return; }
      if (ev.type === "done") { setPending(null); return; }
      const row = eventRow(ev);
      if (row) { pending.remove(); tl.append(row); tl.append(pending); }
      if (ev.type === "decision" && ev.action === "stop") setPending(null);
      if (STEP_LABELS[ev.type] && !(ev.type === "grading" && ev.sufficient)) setPending(STEP_LABELS[ev.type]);
      if (ev.type === "grading" && ev.sufficient) setPending("Prepara la risposta");
      if (ev.type === "clarification") setPending(null);
    });
  } catch (e) { failed = e.message; }
  stop();
  state.busy = false; go.disabled = false;
  if (failed) live.append(errorBox("La ricerca si è interrotta: " + failed));
  if (runId) { await showRun(main, runId, { fromLive: true }); await loadRuns(main); refreshSessionList(main);
    if (!$("#protection", main).hidden) toggleProtection(main, true); }
}

async function showRun(main, runId, { fromLive = false } = {}) {
  const view = await getJSON(`/api/runs/${runId}`);
  state.runId = runId;
  const result = $("#result", main);
  result.innerHTML = "";
  if (!fromLive) $("#live", main).innerHTML = "";
  const run = view.run;
  const card = h(`<div class="card answer-card"></div>`);
  result.append(card);
  if (view.status === "needs_input" && view.clarification) {
    card.classList.add("clarify");
    card.innerHTML = `<div class="question-echo">${esc(run.question)}</div><h3>L’agente ha una domanda per te</h3>
      <p>Ho trovato valori diversi in anni diversi per <b>${esc((view.clarification.elements || []).join("; "))}</b>. Quale anno ti interessa?</p>`;
    for (const opt of view.clarification.options || []) {
      const vals = Object.entries(opt.values || {}).map(([k, v]) => `${k}: ${v}`).join(" · ");
      const b = h(`<button class="option"><b>${esc(opt.year)}</b><span style="flex:1">${esc(vals)}</span><span class="muted small">${esc(shortFile(opt.filename))}</span></button>`);
      b.onclick = () => runAgent(main, { question: `${run.question} (anno ${opt.year})`, clarification: {
        parent_run_id: run._id, context: `L’utente ha scelto l’anno ${opt.year}. Usa solo i dati di quell’anno${opt.filename ? ", dal documento " + opt.filename : ""}.`,
        filters: opt.filename ? { filename: opt.filename } : null } });
      card.append(b);
    }
    const cmp = h(`<button class="option" style="justify-content:center;border-color:var(--agent);color:var(--agent)">⇄ Confronta gli anni</button>`);
    cmp.onclick = () => runAgent(main, { question: `${run.question} Confronta i valori per ciascun anno disponibile.`, clarification: {
      parent_run_id: run._id, context: "L’utente vuole il confronto tra gli anni: indica sempre l’anno di ogni valore.", skip_ambiguity: true } });
    card.append(cmp);
  } else if (view.status === "complete" && view.answer) {
    const v = view.verdict;
    const matches = {};
    for (const r of v.rows || []) if (r.excerpt && r.ref) (matches[r.ref] ||= []).includes(r.excerpt.match) || matches[r.ref].push(r.excerpt.match);
    const cited = view.cited || [];
    const others = view.sources.map((_, i) => i + 1).filter((n) => !cited.includes(n));
    card.innerHTML = `<div class="question-echo">${esc(run.question)}</div>
      <div class="answer-head"><h3 style="margin:0">Risposta</h3>${verdictBadge(v.kind, v.label)}</div>
      <div class="answer">${renderAnswer(view.answer)}</div>
      ${v.notes?.length ? `<p class="notes">${esc(v.notes.join(" · "))}</p>` : ""}
      ${cited.length ? `<div class="sources"><div class="muted small">Fonti citate</div>${cited.map((n) => sourceLine(n, view.sources[n - 1], matches[n] || [])).join("")}</div>` : ""}
      <details><summary>Come è stata verificata${others.length ? ` · altre ${others.length} fonti esaminate` : ""}</summary>
        ${citationRows(v.rows, view.sources)}
        <p class="muted small">Controllo meccanico: dice se i numeri vengono dai documenti, non se il modello li ha interpretati bene.</p>
        ${others.length ? `<div class="muted small">Fonti date al modello e non citate</div>${others.map((n) => sourceLine(n, view.sources[n - 1])).join("")}` : ""}
      </details>`;
  } else if (view.status === "blocked") {
    card.innerHTML = `<div class="question-echo">${esc(run.question)}</div><div class="answer-head"><h3 style="margin:0">Nessuna risposta</h3>${verdictBadge("absent", "Fuori dai documenti")}</div><p>${esc(view.answer || "")}</p>`;
  } else if (view.status === "failed") {
    if (fromLive) { card.remove(); result.append(techDetails(run)); return; }
    const err = (run.events || []).filter((e) => e.type === "error").pop();
    card.innerHTML = `<div class="question-echo">${esc(run.question)}</div>${errorBox("La ricerca si è interrotta prima della risposta" + (err?.comment ? ": " + err.comment : ".")).outerHTML}`;
    if (!fromLive) {
      const tl = h(`<div class="timeline"></div>`);
      for (const ev of run.events || []) { const r = eventRow(ev); if (r) tl.append(r); }
      if (tl.childElementCount) card.append(tl);
    }
  } else {
    card.innerHTML = `<div class="question-echo">${esc(run.question)}</div><p class="muted">Ricerca in corso o interrotta.</p>`;
  }
  if (view.value) result.append(valueCard(view.value));
  result.append(techDetails(run));
}

function valueCard(v) {
  const el = h(`<div class="card value"></div>`);
  const render = () => {
    const chips = (found, animate, base) => (v.elements || []).map((name, i) => {
      const on = (found || []).includes(name);
      return `<span class="dchip ${on ? "on" : "off"} ${animate ? "fade" : ""}" style="${animate ? `animation-delay:${base + 0.15 * i}s` : ""}">${on ? "✓" : "✗"} ${esc(name.length > 48 ? name.slice(0, 46) + "…" : name)}</span>`;
    }).join("");
    const steps = (v.steps || []).map((s, i) => `<div class="vstep" style="animation-delay:${0.3 + 0.75 * i}s"><span>${s.icon}</span><span>${esc(s.text)}</span><span class="t">${esc(s.ms || "")}</span></div>`).join("");
    const end = 0.3 + 0.75 * (v.steps || []).length;
    const m = v.memory;
    const mem = m ? `<div class="memory fade" style="animation-delay:.1s">
        <div class="memory-head">🍃 Memoria della conversazione in MongoDB <code>${esc(m.collection)}</code>${m.read_ms != null ? ` · letta in ${m.read_ms < 1 ? "meno di 1 ms" : fmtMs(m.read_ms)}` : ""}</div>
        <div class="memory-flow">
          <div class="mbox"><div class="k">Domande precedenti salvate nella sessione</div>${(m.previous_questions || []).map((q) => `<div class="mq">${esc(q)}</div>`).join("") || "<div class='mq'>—</div>"}</div>
          <div class="marrow">+</div>
          <div class="mbox ask"><div class="k">Hai chiesto</div><div class="mq big">«${esc(m.question)}»</div></div>
          <div class="marrow">→</div>
          <div class="mbox out"><div class="k">L’agente ha cercato</div><div class="mq big">«${esc(m.planned)}»</div></div>
        </div>
        <div class="muted small">Senza la sessione salvata, «${esc(m.question)}» non direbbe all’agente di quale azienda e di quale dato stai parlando. Ogni domanda e risposta resta nel documento della sessione: riaprendola domani, il contesto c’è ancora.</div>
      </div>` : "";
    el.innerHTML = `<div class="value-head"><span class="value-title">✨ Cosa ha fatto l’agente in più: ${esc(lowerFirst(v.title))}</span><button class="small" id="replay">↻ Riproduci</button></div>${mem}
      <div class="value-cols">
        <div class="value-col"><div class="label">🔎 Ricerca standard · una ricerca, risponde subito</div>${v.standard_chips === false ? "" : chips(v.standard_found, false, 0)}<div class="verdict">${esc(v.standard)}</div></div>
        <div class="value-col agent"><div class="label">🧭 Agente · controlla prima di rispondere</div>${steps}${chips(v.agent_found, true, end)}<div class="verdict fade" style="animation-delay:${end + 0.4}s">${esc(v.agent)}</div></div>
      </div><div class="muted small" style="margin-top:8px">${esc(v.cost)}</div>`;
    $("#replay", el).onclick = render;
  };
  render();
  return el;
}

function techDetails(run) {
  const el = h(`<details class="card tech"><summary>Dettagli tecnici: passi, valutazioni, documento in MongoDB</summary>
      <div class="subtabs"><button data-t="steps" class="active">Passi e tempi</button><button data-t="grades">Valutazioni</button><button data-t="mongo">Documento in MongoDB</button></div>
      <div id="tbody"></div></details>`);
  const events = run.events || [];
  const panes = {
    steps: () => {
      const tl = h(`<div class="timeline"></div>`);
      for (const ev of events) { const r = eventRow(ev); if (r) tl.append(r); }
      const total = events.reduce((a, e) => a + (e.duration_ms || 0), 0);
      const parts = [["Atlas", "#00684a", events.filter((e) => ["retrieval", "agent_rank_fusion"].includes(e.type))],
        ["Decisioni dell’agente", "#a79cf0", events.filter((e) => e.type === "grading" || (e.type === "plan" && !e.planner_skipped))],
        ["Risposta", "#6e5bd6", events.filter((e) => e.type === "answer")]].map(([n, c, es]) => [n, c, es.reduce((a, e) => a + (e.duration_ms || 0), 0)]);
      const bar = total ? `<div class="timebar">${parts.map(([, c, ms]) => ms ? `<span style="width:${(ms / total) * 100}%;background:${c}"></span>` : "").join("")}</div>
        <div class="legend">${parts.filter((p) => p[2]).map(([n, c, ms]) => `<span><i style="background:${c}"></i>${esc(n)} ${fmtMs(ms)}</span>`).join("")}</div>` : "";
      const wrap = h(`<div></div>`); wrap.append(tl); wrap.insertAdjacentHTML("beforeend", bar);
      const plan = events.find((e) => e.type === "plan");
      if (plan?.database_search) wrap.insertAdjacentHTML("beforeend", `<p class="muted small">Collection ${esc(plan.database_search.database)}.${esc(plan.database_search.collection)} · indice testuale ${esc(plan.database_search.text_index)} · ${(plan.database_search.vector_indexes || []).map((v) => `vettoriale ${esc(v.index)} (${esc(v.mode)})`).join(" · ")}</p>`);
      for (const r of events.filter((e) => e.type === "retrieval")) wrap.insertAdjacentHTML("beforeend", `<div class="muted small">Ricerca ${r.iteration}: query completa</div><pre>${esc(r.search_query)}</pre>`);
      return wrap;
    },
    grades: () => {
      const wrap = h(`<div></div>`);
      for (const g of events.filter((e) => e.type === "grading")) {
        wrap.insertAdjacentHTML("beforeend", `<h4>Valutazione ${g.iteration}</h4><p class="small">${esc(g.comment || "")}</p>` +
          (g.elements || []).map((e) => `<div class="small">${e.found ? "✓" : e.claimed_found ? "≈" : "✗"} ${esc(e.element)}${e.value ? ` = <b>${esc(e.value)}</b>` : ""}${e.ref ? ` · passaggio #${e.ref}` : ""}</div>`).join("") +
          (g.evaluated_source_refs || []).map((r, i) => `<div class="passage"><b>#${i + 1}</b> ${esc(shortFile(r.filename))} · ${esc(pageLabel(r.page))}<pre>${esc(r.excerpt || "")}</pre></div>`).join(""));
      }
      return wrap.childElementCount ? wrap : h(`<p class="muted">Nessuna valutazione registrata.</p>`);
    },
    mongo: () => h(`<div><p class="muted small">Il documento della collection <code>agentic_runs</code>: stato e decisioni osservabili dell’agente, non il ragionamento interno del modello.</p><pre>${esc(JSON.stringify(run, null, 2))}</pre></div>`),
  };
  const body = $("#tbody", el);
  const show = (k) => { $$(".subtabs button", el).forEach((b) => b.classList.toggle("active", b.dataset.t === k)); body.innerHTML = ""; body.append(panes[k]()); };
  $$(".subtabs button", el).forEach((b) => b.onclick = () => show(b.dataset.t));
  el.addEventListener("toggle", () => { if (el.open && !body.childElementCount) show("steps"); });
  return el;
}

// ─── Pagina: Query ─────────────────────────────────────────────────────
pages.query = (main) => {
  main.append(h(`<div><h1>Query</h1><p class="muted">La RAG classica: una ricerca ibrida in Atlas, il reranking e una sola risposta del modello. Segui ogni passo della pipeline.</p></div>`));
  const box = questionBox({
    placeholder: "Fai una domanda sui documenti… (⌘↵ per avviare)", button: "Esegui",
    onSubmit: (q) => runQuery(main, q),
    settingsHtml: `
      <div><label>Candidati per motore (Top-K)</label><input type="number" id="q-topk" value="50" min="5" max="100"></div>
      <div><label>Passaggi al modello (Top-N)</label><input type="number" id="q-topn" value="8" min="1" max="20"></div>
      <div><label>Soglia fuori tema</label><input type="number" id="q-thr" value="0.15" step="0.05" min="0" max="1"></div>
      <div><label><input type="checkbox" id="q-fuzzy"> Fuzzy sul ramo testuale</label></div>
      <div style="grid-column:1/-1"><label>Istruzioni al reranker (facoltative)</label><input type="text" id="q-instr" placeholder="Es. privilegia le tabelle con i totali"></div>`,
  });
  main.append(box);
  main.append(h(`<div class="pipeline" id="pipe">
      <div class="pstep" data-s="embedding"><div class="k">1 · Domanda</div><div class="v">Embedding</div><div class="ms"></div></div>
      <div class="pstep" data-s="atlas"><div class="k">2 · MongoDB Atlas</div><div class="v">$search + $vectorSearch → $rankFusion → $rerank</div><div class="ms"></div></div>
      <div class="pstep" data-s="generation"><div class="k">3 · Modello</div><div class="v">Risposta con citazioni</div><div class="ms"></div></div>
    </div>`));
  main.append(h(`<div id="qresult"></div>`));
};

async function runQuery(main, question) {
  state.busy = true;
  const go = $("#go", main); go.disabled = true;
  const out = $("#qresult", main); out.innerHTML = "";
  const steps = $$(".pstep", main); steps.forEach((s) => { s.className = "pstep"; $(".ms", s).textContent = ""; });
  const stop = elapsedTimer($("#elapsed", main));
  const body = { question, top_k: Number($("#q-topk", main).value), top_n: Number($("#q-topn", main).value),
    threshold: Number($("#q-thr", main).value), fuzzy: $("#q-fuzzy", main).checked, rerank_instructions: $("#q-instr", main).value };
  let retrieval = null, result = null, failed = null;
  try {
    await stream("/api/query", body, (ev) => {
      if (ev.type === "step") {
        steps.forEach((s) => { if (s.classList.contains("active")) { s.classList.remove("active"); s.classList.add("done"); } });
        $(`.pstep[data-s="${ev.step}"]`, main)?.classList.add("active");
      } else if (ev.type === "retrieval") retrieval = ev;
      else if (ev.type === "result") result = ev;
      else if (ev.type === "error") failed = ev.message;
    });
  } catch (e) { failed = e.message; }
  stop(); state.busy = false; go.disabled = false;
  // Se il modello non risponde, la parte di Atlas resta comunque visibile.
  if (!result && retrieval) result = { ...retrieval, status: "failed", question, message: failed };
  else if (failed) out.append(errorBox(failed));
  steps.forEach((s) => { s.classList.remove("active"); });
  if (result?.timings) {
    for (const [k, ms] of Object.entries(result.timings)) { const s = $(`.pstep[data-s="${k}"]`, main); if (s) { s.classList.add("done"); $(".ms", s).textContent = fmtMs(ms); } }
  }
  if (result) renderQueryResult(out, result, body);
}

function renderQueryResult(out, r, body) {
  const card = h(`<div class="card answer-card"></div>`);
  out.append(card);
  if (r.status === "failed") {
    card.innerHTML = `<div class="answer-head"><h3 style="margin:0">Nessuna risposta</h3>${verdictBadge("risk", "Il modello non ha risposto")}</div>
      ${errorBox(r.message || "Errore del modello").outerHTML}
      <p class="muted small">La ricerca in Atlas è andata a buon fine: sotto vedi cosa ha trovato e in che ordine.</p>`;
  } else if (r.status === "off_topic") {
    card.innerHTML = `<div class="answer-head"><h3 style="margin:0">Nessuna risposta</h3>${verdictBadge("absent", "Fuori tema")}</div><p>${esc(r.message)}</p>`;
  } else if (r.status === "blocked") {
    card.innerHTML = `${errorBox("Risposta non pubblicata: " + r.message).outerHTML}`;
  } else {
    const s = r.citation_summary || {};
    const checked = (r.citation_check || []).length - (s.uncited || 0);
    const ok = (s.found || 0) + (s.rounded || 0);
    const kind = s.not_found || s.uncited ? "risk" : "ok";
    const label = !(r.citation_check || []).length ? "Nessun numero da verificare" : kind === "ok" ? `Verificata: ${ok} numeri su ${checked} nelle fonti` : "Da verificare";
    const cited = (r.sources || []).filter((x) => x.cited);
    const matches = {};
    for (const row of r.citation_check || []) if (row.excerpt && row.ref) (matches[row.ref] ||= []).push(row.excerpt.match);
    card.innerHTML = `<div class="answer-head"><h3 style="margin:0">Risposta</h3>${verdictBadge(kind, label)}</div>
      <div class="answer">${renderAnswer(r.answer)}</div>
      ${r.message ? `<p class="notes">${esc(r.message)}</p>` : ""}
      ${cited.length ? `<div class="sources"><div class="muted small">Fonti citate</div>${cited.map((x) => sourceLine(x.number, x, matches[x.number] || [])).join("")}</div>` : ""}
      <details><summary>Come è stata verificata</summary>${citationRows(r.citation_check, r.sources)}</details>
      <p class="muted small">Modello ${esc(r.model || "")} · ${r.usage?.input_tokens ?? "?"} token in · ${r.usage?.output_tokens ?? "?"} out</p>`;
  }
  // Come ha lavorato Atlas
  const how = h(`<div class="card" style="margin-top:18px"><h3>Come ha lavorato Atlas</h3>
      <div class="subtabs"><button data-t="fusion" class="active">Fusione dei motori</button><button data-t="rerank">Reranking</button><button data-t="compare">Motori separati</button><button data-t="mql">Pipeline MQL</button></div>
      <div id="howbody"></div></div>`);
  out.append(how);
  const hb = $("#howbody", how);
  const fusedPos = Object.fromEntries((r.fused || []).map((p, i) => [p.key, i + 1]));
  const citedKeys = new Set((r.sources || []).filter((x) => x.cited).map((x) => x.key));
  const label = (p) => `${shortFile(p.filename)} ${pageLabel(p.page)}`;
  const panes = {
    fusion: () => {
      const single = (r.scope || []).length === 1;
      const cols = [{ key: "text", title: "Testuale", sub: "BM25 · $search" }].concat((r.scope || []).map((m) => ({ key: `vector_${m}`, title: single ? "Semantica" : m, sub: "$vectorSearch" })));
      const items = (r.fused || []).slice(0, 10).map((p, i) => {
        const ranks = { text: p.branch_ranks.text };
        for (const m of r.scope || []) ranks[`vector_${m}`] = p.branch_ranks[`vector_${m}`] ?? (single ? p.branch_ranks.vector : undefined);
        return { file: p.filename, label: label(p), sub: p.snippet.slice(0, 48), ranks, final: i + 1, cited: citedKeys.has(p.key) };
      });
      const wrap = h(`<div><p class="small muted">Ogni linea è un passaggio: a sinistra la sua posizione nella ricerca per parole e in quella per significato, a destra la posizione dopo <code>$rankFusion</code>. ${esc(r.weights || "")} ${esc(r.dedup || "")}</p><div></div></div>`);
      lanes(wrap.lastElementChild, { columns: cols, target: { title: "Classifica fusa", sub: "$rankFusion (RRF)" }, items });
      return wrap;
    },
    rerank: () => {
      const items = (r.reranked_all || []).slice(0, 10).map((p, i) => ({ file: p.filename, label: label(p), sub: `rerank ${p.rerank_score != null ? Number(p.rerank_score).toFixed(3) : "n/d"}`,
        ranks: { rrf: fusedPos[p.key] }, final: i + 1, cited: citedKeys.has(p.key), delta: fusedPos[p.key] ? fusedPos[p.key] - (i + 1) : 0 }));
      const wrap = h(`<div><p class="small muted">Il reranker Voyage rilegge domanda e passaggio insieme e riordina i candidati della fusione. Solo i primi ${r.top_n} arrivano al modello. Punteggio del primo: ${r.top_score != null ? Number(r.top_score).toFixed(3) : "n/d"} (soglia fuori tema ${r.threshold}).</p><div></div></div>`);
      lanes(wrap.lastElementChild, { columns: [{ key: "rrf", title: "Fusione", sub: "RRF" }], target: { title: "Dopo il reranking", sub: "$rerank · Voyage" }, items, cut: r.top_n });
      return wrap;
    },
    compare: () => {
      const wrap = h(`<div><p class="small muted">Le classifiche dei due motori presi da soli, con gli stessi filtri: mostrano cosa trova la sola ricerca per parole e cosa la sola ricerca per significato.</p><button id="cmp">Calcola le classifiche separate</button><div id="cmpout"></div></div>`);
      $("#cmp", wrap).onclick = async (e) => {
        e.target.disabled = true; e.target.textContent = "Calcolo…";
        try {
          const c = await sendJSON("/api/query/compare", { question: r.question, top_k: body.top_k, fuzzy: body.fuzzy });
          const list = (title, xs) => `<div><h4>${esc(title)}</h4><div class="rank-list">${xs.map((p, i) => `<div><b>#${i + 1}</b> ${esc(label(p))} <span class="muted">${esc(p.snippet.slice(0, 80))}</span></div>`).join("")}</div></div>`;
          $("#cmpout", wrap).innerHTML = `<div class="compare-cols">${list("Solo testo · $search", c.text)}${Object.entries(c.vector).map(([m, xs]) => list(`Solo vettori · ${m}`, xs)).join("")}</div>`;
          e.target.remove();
        } catch (err) { $("#cmpout", wrap).append(errorBox(err.message)); e.target.disabled = false; }
      };
      return wrap;
    },
    mql: () => h(`<div><p class="small muted">La pipeline inviata ad Atlas, pronta per <code>mongosh</code>.</p><pre>${esc(r.pipeline)}</pre><p class="small muted">Domanda inviata al reranker:</p><pre>${esc(r.rerank_query)}</pre></div>`),
  };
  const show = (k) => { $$(".subtabs button", how).forEach((b) => b.classList.toggle("active", b.dataset.t === k)); hb.innerHTML = ""; hb.append(panes[k]()); };
  $$(".subtabs button", how).forEach((b) => b.onclick = () => show(b.dataset.t));
  show("fusion");
}

// ─── Pagina: Upload ────────────────────────────────────────────────────
pages.upload = async (main) => {
  await loadInfo();
  const i = state.info || {};
  let mode = i.active_mode || "auto";
  main.append(h(`<div><h1>Carica documenti</h1><p class="muted">I documenti vengono spezzati in passaggi, indicizzati in Atlas e conservati in GridFS. La modalità scelta diventa quella in cui cercano Query e Ricerca agentica.</p></div>`));
  const modes = h(`<div class="grid4"></div>`);
  const drawModes = () => {
    modes.innerHTML = Object.entries(i.modes || {}).map(([m, d]) => `<div class="mode-card ${m === mode ? "selected" : ""}" data-m="${m}">
      <b>${m}</b>${m === i.active_mode ? '<span class="active-badge">attiva</span>' : ""}<div class="small">${esc(d)}</div>
      <div class="count">${(i.counts || {})[m] || 0} passaggi caricati</div></div>`).join("");
    $$(".mode-card", modes).forEach((c) => c.onclick = () => { mode = c.dataset.m; drawModes(); });
  };
  drawModes();
  main.append(modes);
  const form = h(`<div class="card" style="margin-top:16px">
      <div class="drop" id="drop">Trascina qui i documenti oppure
        <label class="btn" style="display:inline-flex;margin:0 6px">scegli i file<input type="file" id="files" multiple hidden accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.html,.htm,.csv,.json,.xml,.md"></label>
        <label class="btn" style="display:inline-flex">una cartella<input type="file" id="folder" webkitdirectory hidden></label>
        <div class="small" id="picked" style="margin-top:8px"></div></div>
      <details style="margin-top:12px"><summary>Metadati (facoltativi: diventano filtri nella ricerca)</summary>
        <div class="settings">
          <div><label>Titolo</label><input type="text" id="m-title"></div>
          <div><label>Autore</label><input type="text" id="m-author"></div>
          <div><label>Categoria</label><input type="text" id="m-category" placeholder="Report, Contratto, …"></div>
          <div><label>Anno</label><input type="number" id="m-year" min="1900" max="2100"></div>
          <div><label>Lingua</label><select id="m-lang"><option value="">rilevata in automatico</option><option>it</option><option>en</option><option>fr</option><option>de</option><option>es</option></select></div>
          <div><label>Tag (separati da virgola)</label><input type="text" id="m-tags" placeholder="financebench"></div>
          <div><label>Se il documento c’è già</label><select id="m-dup"><option value="skip">salta</option><option value="replace">sostituisci</option><option value="append">aggiungi una copia</option></select></div>
        </div></details>
      <div class="row" style="margin-top:12px"><button class="primary" id="start" disabled>Avvia il caricamento</button><span class="muted small" id="up-elapsed"></span></div>
      <div id="progress"></div></div>`);
  main.append(form);
  let files = [];
  const setFiles = (list) => {
    files = [...list].filter((f) => !f.name.startsWith("."));
    const mb = files.reduce((a, f) => a + f.size, 0) / 1048576;
    $("#picked", form).textContent = files.length ? `${files.length} documenti · ${mb.toFixed(1)} MB` : "";
    $("#start", form).disabled = !files.length;
  };
  $("#files", form).onchange = (e) => setFiles(e.target.files);
  $("#folder", form).onchange = (e) => setFiles(e.target.files);
  const drop = $("#drop", form);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); setFiles(e.dataTransfer.files); };
  $("#start", form).onclick = async () => {
    const meta = {};
    for (const [k, id] of [["title", "#m-title"], ["author", "#m-author"], ["category", "#m-category"], ["language", "#m-lang"]]) { const v = $(id, form).value.trim(); if (v) meta[k] = v; }
    if ($("#m-year", form).value) meta.year = Number($("#m-year", form).value);
    const tags = $("#m-tags", form).value.split(",").map((t) => t.trim()).filter(Boolean); if (tags.length) meta.tags = tags;
    meta.duplicate_policy = $("#m-dup", form).value;
    const fd = new FormData();
    fd.append("mode", mode); fd.append("metadata", JSON.stringify(meta));
    files.forEach((f) => fd.append("files", f, f.name));
    const prog = $("#progress", form);
    prog.innerHTML = `<table style="margin-top:12px"><thead><tr><th>#</th><th>File</th><th>Esito</th><th>Pagine</th><th>Passaggi</th><th>Secondi</th></tr></thead><tbody></tbody></table>`;
    const tbody = $("tbody", prog);
    $("#start", form).disabled = true;
    const stop = elapsedTimer($("#up-elapsed", form));
    const outcome = { inserted: "✓ inserito", skipped: "già presente", error: "⚠ errore" };
    try {
      await stream("/api/upload", fd, (ev) => {
        if (ev.type === "file_start") tbody.insertAdjacentHTML("beforeend", `<tr id="f${ev.number}"><td>${ev.number}</td><td>${esc(ev.file)}</td><td colspan="4"><span class="muted">elaborazione…</span></td></tr>`);
        if (ev.type === "file_done") $(`#f${ev.number}`, tbody).innerHTML = `<td>${ev.number}</td><td>${esc(ev.file)}${ev.detail ? `<div class="muted small">${esc(ev.detail)}</div>` : ""}</td><td>${outcome[ev.outcome] || ev.outcome}</td><td>${ev.pages}</td><td>${ev.chunks}</td><td>${ev.seconds}</td>`;
        if (ev.type === "error") prog.append(errorBox(ev.message));
      });
    } catch (e) { prog.append(errorBox(e.message)); }
    stop();
    $("#start", form).disabled = false;
    await loadInfo();
  };
};

// ─── Pagina: Admin ─────────────────────────────────────────────────────
pages.admin = async (main) => {
  await loadInfo();
  const i = state.info || {};
  main.append(h(`<div><h1>Admin</h1><p class="muted">Stato di Atlas, modalità attiva, indici e documenti caricati.</p></div>`));
  const total = Object.values(i.counts || {}).reduce((a, b) => a + b, 0);
  main.append(h(`<div class="grid4">
      <div class="card"><div class="muted small">MongoDB Atlas</div><div class="stat">${i.atlas === "ok" ? "✓ online" : "✗"}</div><div class="small muted">${esc(i.database)}.${esc(i.collection)}</div></div>
      <div class="card"><div class="muted small">Passaggi indicizzati</div><div class="stat">${total}</div><div class="small muted">in tutte le modalità</div></div>
      <div class="card"><div class="muted small">Modalità attiva</div><div class="stat">${esc(i.active_mode)}</div><div class="small muted">${esc(i.embed_model || "")}</div></div>
      <div class="card"><div class="muted small">Modello della risposta</div><div class="stat" style="font-size:17px">${i.llm_ready ? "✓ " : "✗ "}${esc(i.llm)}</div><div class="small muted">rerank ${esc(i.rerank)}</div></div>
    </div>`));
  // Modalità attiva
  const modeCard = h(`<div class="card" style="margin-top:16px"><h3>Modalità attiva</h3><p class="small muted">Query e Ricerca agentica cercano solo nei documenti caricati nella modalità attiva.</p><div class="grid4" id="modes"></div></div>`);
  $("#modes", modeCard).innerHTML = Object.entries(i.modes || {}).map(([m, d]) => `<div class="mode-card ${m === i.active_mode ? "selected" : ""}" data-m="${m}"><b>${m}</b><div class="small">${esc(d)}</div><div class="count">${(i.counts || {})[m] || 0} passaggi</div></div>`).join("");
  $$(".mode-card", modeCard).forEach((c) => c.onclick = async () => { await sendJSON("/api/admin/mode", { mode: c.dataset.m }); await loadInfo(); route(); });
  main.append(modeCard);
  // Indici e setup
  const idx = h(`<div class="card"><div class="row"><h3 style="margin:0">Indici Atlas Search</h3><span class="spacer"></span><label class="small" style="margin:0;display:flex;gap:6px;align-items:center"><input type="checkbox" id="recreate"> Elimina e ricrea gli indici</label><button id="setup">Prepara database e indici</button></div>
      <p class="small muted">Crea ciò che manca e aggiorna gli indici diversi dalla configurazione. “Elimina e ricrea” serve solo dopo un cambio di modello: durante la ricostruzione la ricerca non risponde.</p>
      <table style="margin-top:10px"><thead><tr><th>Modalità</th><th>Indice vettoriale</th><th>Modello</th><th>Stato</th></tr></thead><tbody>
      ${(i.indexes || []).map((x) => `<tr><td>${esc(x.mode)}</td><td><code>${esc(x.name)}</code></td><td>${esc(x.model)}</td><td>${esc(x.status)}</td></tr>`).join("")}</tbody></table>
      ${(i.unavailable || []).length ? `<div class="error">Indici non interrogabili: ${esc(i.unavailable.join(", "))}</div>` : ""}
      <div id="setup-log"></div></div>`);
  $("#setup", idx).onclick = async (e) => {
    e.target.disabled = true;
    await streamLog($("#setup-log", idx), "/api/admin/setup", { recreate: $("#recreate", idx).checked });
    e.target.disabled = false; loadInfo();
  };
  main.append(idx);
  // Documenti
  const docs = h(`<div class="card"><div class="row"><h3 style="margin:0">Documenti caricati</h3><span class="spacer"></span><button class="danger" id="reset-open">Riparti da zero…</button></div>
      <div id="reset" class="reset-panel" hidden>
        <b>Riparti da zero</b>
        <p class="small">Cancella tutti i passaggi e i file originali (GridFS), poi prepara di nuovo database e indici. Le impostazioni restano.</p>
        <label class="small"><input type="checkbox" id="r-sessions" checked> Cancella anche sessioni e ricerche della Ricerca agentica</label>
        <label class="small"><input type="checkbox" id="r-indexes"> Elimina e ricrea anche gli indici Atlas Search</label>
        <label class="small" style="margin-top:8px">Per confermare scrivi <b>AZZERA</b></label>
        <div class="row"><input type="text" id="r-confirm" style="max-width:180px" autocomplete="off"><button class="danger" id="r-go" disabled>Cancella tutto e ricrea</button><button id="r-cancel">Annulla</button></div>
        <div id="reset-log"></div>
      </div>
      <div id="doclist" class="muted">Caricamento…</div></div>`);
  main.append(docs);
  const loadDocs = async () => {
    const list = await getJSON("/api/admin/documents");
    const chunks = list.reduce((a, d) => a + d.chunks, 0);
    $("#doclist", docs).innerHTML = list.length ? `<p class="small">${list.length} documenti · ${chunks} passaggi</p><table><thead><tr><th>Documento</th><th>Modalità</th><th>Pagine</th><th>Passaggi</th><th>Caricato</th><th></th></tr></thead><tbody>
      ${list.map((d) => `<tr><td>${esc(d.filename || d.title)}${d.title && d.title !== d.filename ? `<div class="muted small">${esc(d.title)}</div>` : ""}${d.gridfs_file_id ? ` · <a href="/api/files/${esc(d.gridfs_file_id)}" target="_blank">PDF</a>` : ""}</td><td>${esc((d.modes || []).join(", "))}</td><td>${d.total_pages ?? ""}</td><td>${d.chunks}</td><td>${esc((d.ingested_at || "").slice(0, 16).replace("T", " "))}</td>
        <td><button class="danger small" data-id="${esc(d._id)}">Elimina</button></td></tr>`).join("")}</tbody></table>` : "<p>Nessun documento. Caricali dalla pagina Upload.</p>";
    $$("button[data-id]", docs).forEach((b) => b.onclick = async () => {
      if (b.dataset.confirm !== "1") { b.dataset.confirm = "1"; b.textContent = "Conferma"; return; }
      b.disabled = true; await sendJSON(`/api/admin/documents/${encodeURIComponent(b.dataset.id)}`, null, "DELETE"); await loadDocs(); loadInfo();
    });
  };
  const panel = $("#reset", docs);
  $("#reset-open", docs).onclick = () => { panel.hidden = false; $("#r-confirm", docs).focus(); };
  $("#r-cancel", docs).onclick = () => { panel.hidden = true; $("#r-confirm", docs).value = ""; $("#r-go", docs).disabled = true; };
  $("#r-confirm", docs).oninput = (e) => { $("#r-go", docs).disabled = e.target.value.trim() !== "AZZERA"; };
  $("#r-go", docs).onclick = async (e) => {
    e.target.disabled = true; $("#r-cancel", docs).disabled = true;
    await streamLog($("#reset-log", docs), "/api/admin/reset", { confirm: "AZZERA",
      sessions: $("#r-sessions", docs).checked, recreate_indexes: $("#r-indexes", docs).checked });
    $("#r-confirm", docs).value = ""; $("#r-cancel", docs).disabled = false;
    await loadDocs(); loadInfo();
  };
  loadDocs().catch((err) => $("#doclist", docs).replaceChildren(errorBox(err.message)));
  // Sessioni della Ricerca agentica
  const ses = h(`<div class="card"><div class="row"><h3 style="margin:0">Sessioni della Ricerca agentica</h3><span class="spacer"></span><button class="danger" id="del-sessions">Cancella tutte le sessioni</button></div>
      <p class="small muted" id="ses-count">Conteggio…</p>
      <p class="small muted">Cancella conversazioni e ricerche salvate (collection <code>agentic_sessions</code> e <code>agentic_runs</code>). Documenti e indici restano.</p></div>`);
  main.append(ses);
  const countSessions = async () => {
    const c = await getJSON("/api/admin/sessions");
    $("#ses-count", ses).textContent = `${c.sessions} sessioni con domande · ${c.runs} ricerche salvate`;
  };
  countSessions().catch((err) => $("#ses-count", ses).replaceChildren(errorBox(err.message)));
  $("#del-sessions", ses).onclick = async (e) => {
    const b = e.target;
    if (b.dataset.confirm !== "1") { b.dataset.confirm = "1"; b.textContent = "Conferma: cancella tutte le sessioni"; return; }
    b.disabled = true;
    try {
      const r = await sendJSON("/api/admin/sessions", null, "DELETE");
      b.textContent = `Cancellate ${r.sessions_deleted} sessioni e ${r.runs_deleted} ricerche`;
      state.sessionId = null;
      await countSessions();
    } catch (err) { ses.append(errorBox(err.message)); b.disabled = false; }
  };
};

async function streamLog(target, url, body) {
  const log = h(`<div class="log" style="margin-top:10px"></div>`); target.replaceChildren(log);
  const line = (html) => { log.insertAdjacentHTML("beforeend", `<div>${html}</div>`); log.scrollTop = log.scrollHeight; };
  try {
    await stream(url, body, (ev) => {
      if (ev.type === "log") line(esc(ev.text));
      if (ev.type === "done") line(`<b>Fatto.</b> Indici creati: ${ev.created.length} · aggiornati: ${ev.updated.length} · già conformi: ${(ev.unchanged || []).length} · errori: ${ev.errors.length}`);
      if (ev.type === "error") line(`<span style="color:var(--danger)">${esc(ev.message)}</span>`);
    });
  } catch (err) { log.append(errorBox(err.message)); }
}

// ─── Avvio ─────────────────────────────────────────────────────────────
// La pagina compare subito; lo stato di Atlas arriva appena pronto.
route();
loadInfo();
loadHealth();
