/* RAG Demo · MongoDB — interfaccia web senza framework. */
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
    renderMeta();
    document.dispatchEvent(new Event("rag:info"));
  } catch (e) {
    $("#meta").textContent = "Server non raggiungibile: " + e.message;
  }
}

function renderMeta() {
  const i = state.info || {};
  const [llmVia, llmModel] = String(i.llm || "").includes(" · ") ? i.llm.split(" · ") : ["", i.llm];
  const native = String(i.rerank || "").includes("native");
  const step = (color, value, title) => `<span class="hstep" style="--c:${color}" title="${esc(title)}">${esc(value)}</span>`;
  const arrow = '<i class="harrow">→</i>';
  $("#meta").innerHTML = [
    step("var(--accent-bright)", "MongoDB", `Database ${i.database}, collection ${i.collection}`),
    `<span class="hmode" title="Dove cercano Query e Ricerca agentica: si sceglie nelle due pagine">${esc(modeName(i.active_mode))}</span>`, arrow,
    step("#7aa7ff", embedFor(i.active_mode), "Voyage AI: trasforma testo e domanda in vettori"), arrow,
    step("#7aa7ff", `${(i.models || {}).rerank || i.rerank}${native ? " in MongoDB" : ""}`, "Voyage AI: riordina i risultati per pertinenza"), arrow,
    step("var(--agent)", llmModel, llmVia ? `Scrive la risposta con le citazioni, tramite ${llmVia}` : "Scrive la risposta con le citazioni"),
  ].join("");
}

// ─── Dove cercare: scelta della modalità in Query e Ricerca agentica ───
const SEARCH_MODES = ["auto", "manual", "context", "multimodal", "all"];
const modeName = (m) => (m === "all" ? "tutte" : m);
function embedFor(m) {
  const md = state.info?.models || {};
  const one = (x) => ({ context: md.context, multimodal: md.multimodal }[x] || md.embed);
  if (m !== "all") return one(m);
  const used = [...new Set(Object.keys(state.info?.counts || {}).filter((x) => state.info.counts[x]).map(one))];
  return used.length === 1 ? used[0] : `${used.length} modelli Voyage`;
}

function modePicker() {
  const el = h(`<div class="mode-pick"><span class="mp-label">Dove cercare</span><div class="seg"></div>
    <span class="small muted" id="mp-note"></span></div>`);
  const draw = () => {
    const i = state.info || {};
    const counts = i.counts || {};
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const n = (m) => (m === "all" ? total : counts[m] || 0);
    // Modalità salvata senza documenti (per esempio scelta in Admin per guardarne la tabella):
    // si cerca dove ci sono documenti, senza cambiare la predefinita salvata.
    let fallback = "";
    if (total && !n(i.active_mode)) {
      fallback = i.active_mode;
      i.active_mode = n("auto") ? "auto" : SEARCH_MODES.find((m) => m !== "all" && n(m));
      renderMeta();
    }
    $(".seg", el).innerHTML = SEARCH_MODES.map((m) => `<button type="button" data-m="${m}" class="${m === i.active_mode ? "on" : ""}" ${n(m) ? "" : "disabled"}
      title="${n(m) ? `${n(m).toLocaleString("it-IT")} passaggi` : "nessun documento caricato in questa modalità"}">${modeName(m)} <small>${n(m).toLocaleString("it-IT")}</small></button>`).join("");
    $("#mp-note", el).textContent = fallback ? `${modeName(fallback)} non ha documenti: cerco in ${modeName(i.active_mode)}`
      : i.active_mode === "all" ? "tutte le modalità con documenti, risultati fusi" : "";
    $$(".seg button", el).forEach((b) => b.onclick = async () => {
      if (state.busy || b.dataset.m === state.info.active_mode) return;
      state.info.active_mode = b.dataset.m;
      draw(); renderMeta();
      sendJSON("/api/admin/mode", { mode: b.dataset.m }).catch(() => {});
    });
  };
  draw();
  const onInfo = () => (el.isConnected ? draw() : document.removeEventListener("rag:info", onInfo));
  document.addEventListener("rag:info", onInfo);
  return el;
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
  // Nomi dei modelli dalla configurazione reale.
  const fill = () => { for (const el of $$("[data-model]", main)) { const m = state.info?.models?.[el.dataset.model]; if (m) el.textContent = m; } };
  fill();
  document.addEventListener("rag:info", fill, { once: true });
};

// ─── Pagina: MongoDB ──────────────────────────────────────────
pages.mongodb = (main) => {
  main.append($("#tpl-mongo").content.cloneNode(true));
  const fill = () => {
    const i = state.info; if (!i) return;
    const set = (k, v) => { const el = $(`[data-stat="${k}"]`, main); if (el) el.textContent = v; };
    if (i.documents != null) set("documents", i.documents.toLocaleString("it-IT"));
    set("indexes", String((i.indexes || []).length + 1));
    for (const el of $$("[data-model]", main)) { const m = i.models?.[el.dataset.model]; if (m) el.textContent = m; }
    const code = $(".mx-code", main);
    if (code && i.models?.rerank) code.textContent = code.textContent.replace('model: "rerank"', `model: "${i.models.rerank}"`);
  };
  fill();
  document.addEventListener("rag:info", fill, { once: true });
  // Collega le parti del documento alle capacità che alimentano.
  const doc = $(".mx-doc", main);
  const mark = (f) => $$("[data-f]", doc).forEach((el) => el.classList.toggle("on", !!f && el.dataset.f.split(" ").includes(f)));
  $$(".mx-dc, .mx-json [data-f]", doc).forEach((el) => {
    el.addEventListener("mouseenter", () => mark(el.dataset.f.split(" ")[0]));
    el.addEventListener("mouseleave", () => mark(null));
  });
  // Le capacità entrano in scena quando arrivano sullo schermo.
  const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.15 });
  $$(".mx-p, .mx-arch-col, .mx-ps", main).forEach((el) => io.observe(el));
  loadLiveProtection(main);
};

// Esempio fisso (non letto dal database): c'è sempre, anche con il database vuoto.
// La domanda è inventata; i byte sono il suo vero cifrato CSFLE (AEAD_AES_256_CBC_HMAC_SHA_512-Random).
const SAMPLE_PROTECTION = {
  id: "6f3a9c2e-41d7-4b8a-9e0f-2c5d7a1b3e84", title: "Sessione del 12/10/2026 alle 10:32",
  created_at: "2026-10-12T08:32:14", turns: 3, field: "first_question", encrypted: true,
  key_alt_name: "rag-demo-sessioni", key_vault: "encryption.__keyVault", sample: true,
  app_view: "Qual è stato il capex di 3M nel 2018? Confrontalo con il nostro piano di investimenti riservato per il 2026.",
  stored_view: { bytes: 194, base64: "AiulklLmw0OyhL9iJIUFMXICGXBomNkiqETinQFTkqgo/KDJB+CxLj6N…" },
};

function loadLiveProtection(main) {
  const box = $("#mx-csfle-live", main);
  if (!box) return;
  box.innerHTML = `<div class="cs-live-tag">Esempio · una sessione della Ricerca agentica con una domanda inventata</div>` + protectionCompare(SAMPLE_PROTECTION);
}

// ─── Componenti condivisi ──────────────────────────────────────────────
function questionBox({ placeholder, button, onSubmit, settingsHtml = "", settingsTitle = "Impostazioni della ricerca" }) {
  const el = h(`<div class="card ask">
      <textarea id="q" placeholder="${esc(placeholder)}"></textarea>
      <div class="chips" id="demo"></div>
      <div id="mode-slot"></div>
      <div class="ask-actions">
        <button class="primary" id="go">${esc(button)}</button>
        <span class="muted small" id="elapsed"></span>
        <span class="spacer"></span>
      </div>
      ${settingsHtml ? `<details><summary>${esc(settingsTitle)}</summary><div class="settings">${settingsHtml}</div></details>` : ""}
    </div>`);
  const q = $("#q", el);
  $("#mode-slot", el).replaceWith(modePicker());
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
  const labels = { found: "✓ Corrispondenza trovata", rounded: "≈ arrotondato", computed: "∑ calcolato", not_found: "⚠ non trovato", uncited: "senza citazione" };
  return rows.map((r) => {
    const src = r.ref ? sources[r.ref - 1] : null;
    const whereSrc = src ? `#${r.ref} · ${esc(shortFile(src.filename))} · ${esc(pageLabel(src.page ?? src.page_number))}` : "Nessun passaggio citato";
    const ex = r.status === "computed" ? `Calcolato: <b>${esc(r.claim)} = ${esc(r.formula)}</b>, con valori trovati nelle fonti.` :
      r.excerpt ? `“${esc(r.excerpt.before)}<mark>${esc(r.excerpt.match)}</mark>${esc(r.excerpt.after)}”` :
      r.status === "not_found" ? "Il numero non compare nel passaggio citato." : "";
    const where = r.status === "computed" ? "Calcolo sui valori della risposta" : whereSrc;
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
  // Nel grafico a specchio i titoli delle colonne esterne si allineano ai bordi: centrati uscirebbero dal riquadro.
  columns.forEach((c, i) => svg += mirrored ? head(i === 0 ? 2 : W - 2, i === 0 ? "start" : "end", c) : head(colX[i], "middle", c));
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
      ? `Cifratura non riuscita sul server che ha salvato questa sessione, quindi il campo è in chiaro anche nel database. Motivo: ${esc(v.encryption_error)}`
      : `Sessione salvata senza cifratura: prima che fosse attiva, o da un server dove non funzionava. Il campo è in chiaro anche nel database. Prova con una nuova sessione.`;
  return `<div class="cs-flow">
      <div><b>1 · La domanda nasce nell’applicazione</b><span>testo in chiaro nell’applicazione</span></div>
      <div class="cs-step-key"><b>2 · 🔑 Il driver MongoDB cifra il campo</b><span>AES-256 con la chiave dati, prima di spedirla</span></div>
      <div class="cs-step-db"><b>3 · MongoDB conserva il valore cifrato</b><span>disco, repliche, backup: solo byte illeggibili</span></div>
      <div class="cs-step-key"><b>4 · 🔑 L’applicazione autorizzata lo decifra</b><span>quando rilegge la sessione, solo dove c’è la chiave</span></div>
    </div>
    <div class="cs-compare">
      <div class="cs-col cs-atlas">
        <div class="cs-head">🔒 Il documento in MongoDB</div>
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
    <p class="small muted cs-foot">${v.sample ? "" : `Sessione del ${esc(fmtDate(v.created_at))}. `}Algoritmo casuale: la stessa domanda dà ogni volta un cifrato diverso.
      La chiave dati <code>${esc(v.key_alt_name)}</code> sta nel key vault <code>${esc(v.key_vault)}</code>, a sua volta cifrata con la chiave master,
      che il database non ha. Titolo, date e contatori restano in chiaro: servono per elenchi e filtri e non sono sensibili.</p>`;
}

// ─── Pagina: Ricerca agentica ──────────────────────────────────────────
pages.agente = async (main) => {
  main.append(h(`<div><h1>Ricerca agentica</h1>
    <p class="muted">Non sempre una sola ricerca basta per rispondere correttamente. A differenza di un RAG tradizionale, la ricerca agentica introduce una fase decisionale: l’agente analizza la domanda, recupera le informazioni con MongoDB e valuta se le evidenze coprono ciò che è stato chiesto.</p>
    <p class="muted">Se manca un dato può fare ricerche mirate, entro il limite configurato; se trova valori diversi per periodi diversi chiede un chiarimento. Poi genera la risposta e dichiara ciò che non è riuscito a trovare. MongoDB conserva la memoria delle conversazioni, le fonti recuperate e gli eventi del processo: ogni esecuzione è osservabile e ricostruibile.</p></div>`));
  const bar = h(`<div class="row" style="margin-bottom:12px">
      <label style="margin:0">Sessione di conversazione</label>
      <select id="session" style="max-width:320px"></select>
      <button id="new-session">＋ Nuova sessione</button>
      <span class="spacer"></span>
      <select id="runs" style="max-width:300px"><option value="">Domande della sessione</option></select>
    </div>`);
  main.append(bar);
  const box = questionBox({
    placeholder: "Fai una domanda sui documenti… (⌘↵ per avviare)",
    button: "Avvia la ricerca agentica",
    onSubmit: (q) => runAgent(main, { question: q }),
    settingsTitle: "Configurazione dell’agente",
    settingsHtml: `
      <div><label>Passaggi recuperati per ricerca</label><input type="number" id="s-topk" value="20" min="5" max="100"></div>
      <div><label>Passaggi forniti al modello</label><input type="number" id="s-topn" value="8" min="1" max="20"></div>
      <div><label>Ricerche supplementari consentite</label><input type="number" id="s-retries" value="5" min="0" max="5"></div>
      <div style="grid-column:1/-1"><label>Istruzioni per il reranking (facoltative)</label><input type="text" id="s-instr" placeholder="Es. privilegia il rendiconto finanziario"></div>
      <div class="small muted settings-help" style="grid-column:1/-1"><p>Puoi controllare quanti passaggi recuperare, quanti fornire al modello e quante ricerche supplementari consentire quando le evidenze iniziali non bastano: è un limite massimo, non il numero di ricerche che l’agente farà per forza. Le istruzioni di reranking orientano la selezione delle fonti verso contenuti specifici, come rendiconti finanziari, tabelle o sezioni rilevanti.</p></div>`,
  });
  main.append(box);
  main.append(h(`<div id="live"></div>`));
  main.append(h(`<div id="result"></div>`));
  // Dopo la risposta: la prima domanda della sessione è appena stata salvata cifrata.
  const protRow = h(`<div class="row" id="prot-row" style="margin-top:14px" hidden>
      <button id="show-protection" title="Lo stesso documento visto dall’applicazione e dal database">🔒 Come MongoDB conserva i dati</button></div>`);
  main.append(protRow);
  main.append(h(`<div id="protection" class="card cs-card" hidden></div>`));
  await loadSessions(main);
  $("#new-session", bar).onclick = async () => {
    const s = await sendJSON("/api/sessions", {});
    state.sessionId = s.id;
    $("#live", main).innerHTML = ""; $("#result", main).innerHTML = ""; $("#protection", main).hidden = true; protRow.hidden = true;
    await loadSessions(main);
  };
  $("#session", bar).onchange = async (e) => {
    state.sessionId = e.target.value;
    $("#live", main).innerHTML = ""; $("#result", main).innerHTML = ""; $("#protection", main).hidden = true; protRow.hidden = true;
    await loadRuns(main, true);
  };
  $("#runs", bar).onchange = (e) => { if (e.target.value) showRun(main, e.target.value); };
  $("#show-protection", protRow).onclick = () => toggleProtection(main);
};

async function toggleProtection(main, refresh = false) {
  const panel = $("#protection", main);
  if (!refresh && !panel.hidden) { panel.hidden = true; return; }
  panel.hidden = false;
  panel.innerHTML = `<p class="muted">Lettura del documento dal database…</p>`;
  try {
    const v = await getJSON(`/api/sessions/${state.sessionId}/protection`);
    const head = `<div class="row"><h3 style="margin:0">La memoria dell’agente: dati sensibili protetti con CSFLE</h3><span class="spacer"></span>
      <button class="small" id="close-protection">Chiudi</button></div>
      <p class="muted small" style="margin:6px 0 10px">La memoria conversazionale può contenere informazioni sensibili da proteggere anche dentro il database. In questa demo l’applicazione cifra con la Client-Side Field Level Encryption (CSFLE) il campo della prima domanda della sessione, prima di inviarlo a MongoDB: MongoDB conserva e replica solo il valore cifrato, e soltanto un’applicazione con le chiavi può decifrarlo. Gli altri campi (identificativi, date, contatori) restano leggibili per le normali operazioni.</p>`;
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
  sel.innerHTML = `<option value="">Domande della sessione (${runs.length})</option>` +
    runs.map((r) => `<option value="${esc(r.id)}">${esc(fmtDate(r.created_at).slice(-5))} · ${esc((r.question || "").slice(0, 70))}</option>`).join("");
  if (showLatest && runs.length && !state.busy) await showRun(main, runs[0].id);
}

const STEP_LABELS = {
  start: "Interpreta la domanda", plan: "Cerca nei documenti con MongoDB", retrieval: "Valuta le evidenze",
  agent_rank_fusion: "Valuta le evidenze", grading: "Decide il prossimo passo", rewrite: "Ricerca mirata in MongoDB",
  generating: "Genera la risposta con citazioni",
};

function timelineRow(icon, title, detail = "", ms = null) {
  return h(`<div class="tl-row"><span class="tl-icon">${icon}</span><div><div class="tl-title">${esc(title)}</div>${detail ? `<div class="tl-detail">${esc(detail)}</div>` : ""}</div><span class="tl-time">${ms ? fmtMs(ms) : ""}</span></div>`);
}

function eventRow(ev) {
  const top3 = (refs) => (refs || []).slice(0, 3).map((r) => `${shortFile(r.filename)} ${pageLabel(r.page)}`).join(" · ");
  switch (ev.type) {
    case "plan":
      if (ev.sub_queries?.length > 1) return timelineRow("🧭", `Divide la domanda in ${ev.sub_queries.length} ricerche`,
        ev.sub_queries.map((q, i) => `${i + 1}. ${q}`).join("  ·  "), ev.duration_ms);
      if (ev.planner_skipped) return timelineRow("🧭", "Interpretazione della domanda", ev.parent_run_id ? "Risposta alla tua scelta: non serve recuperare altro dalla conversazione." : "Prima domanda della sessione: non è necessario recuperare informazioni dalle interazioni precedenti.");
      if (ev.memory?.turns) return timelineRow("🍃", `Ha letto la conversazione da MongoDB: ${ev.memory.turns} ${ev.memory.turns === 1 ? "domanda precedente" : "domande precedenti"}`,
        `Ha trasformato la domanda in «${ev.planned_query}»`, ev.duration_ms);
      return timelineRow("🧭", "Ha interpretato la domanda", `«${ev.planned_query}»`, ev.duration_ms);
    case "retrieval":
      return timelineRow("🔍", ev.parts > 1 ? `Ricerca ${ev.part}/${ev.parts}${ev.iteration > 1 ? " mirata" : ""}: «${ev.search_query}» · ${ev.candidate_count} passaggi`
        : ev.iteration > 1 ? `Ricerca mirata: «${ev.search_query}»` : `Ricerca ibrida e reranking in MongoDB · ${ev.candidate_count} passaggi recuperati`,
        "Passaggi meglio classificati: " + top3(ev.ranked_sources), ev.duration_ms);
    case "agent_rank_fusion":
      return timelineRow("🔗", `Unisce le ricerche: ${ev.unique_candidates} passaggi distinti`, ev.balanced ? "Alternati: ogni parte della domanda porta le sue fonti" : "Riordinati rispetto alla domanda completa", ev.duration_ms);
    case "grading": {
      const els = ev.elements || [];
      const found = els.filter((e) => e.found);
      const missing = els.filter((e) => !e.found).map((e) => e.element);
      if (ev.evaluated === false) return timelineRow("⚠️", "Valutazione non riuscita", ev.failure_detail || ev.failure_reason || "", ev.duration_ms);
      return timelineRow(ev.sufficient ? "✅" : "⚠️",
        ev.sufficient ? `Evidenze ritenute sufficienti · copertura ${found.length}/${els.length || found.length}` : `Copertura incompleta ${found.length}/${els.length} · manca: ${missing.join("; ") || "qualcosa"}`,
        found.map((e) => `${e.element}${e.value ? " = " + e.value : ""}`).join(" · "), ev.duration_ms);
    }
    case "clarification":
      return timelineRow("❓", "Valori diversi in periodi diversi: chiede quale anno", (ev.options || []).map((o) => o.year).join(", "));
    case "decision":
      if (ev.reason === "off_topic") return timelineRow("🚫", "Sotto la soglia di pertinenza: si ferma senza chiamare il modello", `miglior passaggio ${Number(ev.top_rerank).toFixed(3)} < soglia ${ev.threshold}`);
      if (ev.reason === "no_results") return timelineRow("🚫", "Nessun passaggio trovato");
      if (ev.reason === "no_new_passages") return timelineRow("⏹", "La nuova ricerca non porta passaggi nuovi: si ferma", "rivalutare gli stessi passaggi darebbe lo stesso esito");
      if (ev.reason === "not_in_documents") return timelineRow("⏹", "Nessun elemento trovato in due ricerche diverse: si ferma", "i documenti caricati non sembrano contenere la risposta");
      return null;
    case "answer":
      if (ev.status === "needs_input") return null;
      if (ev.diagnostics?.block_reason === "not_in_documents") return timelineRow("🚫", "Nessuna risposta: non inventa e non cita fonti che non c’entrano", ev.stop_reason || "");
      return timelineRow(ev.status === "blocked" ? "🚫" : "💬", ev.status === "blocked" ? "Risposta non pubblicata" : `Risposta generata utilizzando ${ev.sources?.length || 0} passaggi selezionati`, "", ev.duration_ms);
    default: return null;
  }
}

async function runAgent(main, { question, clarification = null }) {
  state.busy = true;
  const go = $("#go", main); go.disabled = true;
  const live = $("#live", main), result = $("#result", main);
  result.innerHTML = "";
  live.innerHTML = "";
  if ($("#prot-row", main)) $("#prot-row", main).hidden = true;
  const card = h(`<div class="card live"><div class="row"><b>Il percorso dell’agente: ricerca, valutazione e risposta</b><span class="spacer"></span><span class="muted small" id="live-clock"></span></div>
    <p class="muted small" style="margin:4px 0 8px">Ogni fase è un’azione eseguita davvero: come l’agente ha interpretato la domanda, cosa ha recuperato, come ha valutato la copertura delle evidenze e quando ha deciso di rispondere. Tempi, risultati e decisioni sono registrati in MongoDB.</p>
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
      session_id: state.sessionId, question, clarification, mode: state.info?.active_mode,
      top_k: s("#s-topk"), top_n: s("#s-topn"), retries: Number($("#s-retries", main)?.value ?? 5),
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
  const protRow = $("#prot-row", main);
  if (protRow) protRow.hidden = false;
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
      ${cited.length ? `<div class="sources"><div class="muted small">Fonti citate nella risposta</div>${cited.map((n) => sourceLine(n, view.sources[n - 1], matches[n] || [])).join("")}</div>` : ""}
      <details><summary>Controllo dei valori e passaggi non citati</summary>
        ${citationRows(v.rows, view.sources)}
        <p class="muted small">Il controllo confronta i valori numerici della risposta con quelli dei passaggi citati. Una corrispondenza conferma che il numero compare nella fonte, ma non garantisce da sola la corretta interpretazione del contesto, delle colonne di una tabella, dell’anno o dell’unità di misura.</p>
        ${others.length ? `<div class="muted small">Passaggi forniti al modello ma non citati</div>${others.map((n) => sourceLine(n, view.sources[n - 1])).join("")}` : ""}
      </details>`;
  } else if (view.status === "blocked") {
    card.innerHTML = `<div class="question-echo">${esc(run.question)}</div><div class="answer-head"><h3 style="margin:0">Nessuna risposta</h3>${verdictBadge("absent", "Nessuna risposta dai passaggi trovati")}</div><p>${esc(view.answer || "")}</p>`;
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
    el.innerHTML = `<div class="value-head"><span class="value-title">✨ Cosa cambia con un RAG agentico?</span><button class="small" id="replay">↻ Riproduci</button></div>
      <p class="muted small" style="margin:2px 0 6px">La differenza non è solo nella ricerca, ma nella capacità di prendere decisioni durante il processo. Un RAG tradizionale recupera i passaggi più pertinenti e li fornisce al modello. Un RAG agentico aggiunge una fase di valutazione: controlla se le informazioni recuperate coprono gli elementi richiesti e, quando serve, cerca ancora o chiede un chiarimento prima di rispondere.</p>
      <p class="small" style="margin:0 0 10px"><b>In questa esecuzione:</b> ${esc(lowerFirst(v.title))}.</p>${mem}
      <div class="value-cols">
        <div class="value-col"><div class="label">🔎 Ricerca standard · recupera e risponde</div><p class="muted small vdesc">MongoDB seleziona i passaggi con ricerca ibrida e reranking; il modello genera una risposta con citazioni, senza una fase che stabilisca prima se le evidenze bastano.</p>${v.standard_chips === false ? "" : chips(v.standard_found, false, 0)}<div class="verdict">${esc(v.standard)}</div></div>
        <div class="value-col agent"><div class="label">🧭 Ricerca agentica · valuta, decide e risponde</div><p class="muted small vdesc">L’agente esamina le evidenze e valuta la copertura della domanda: se bastano risponde, altrimenti può cercare ancora o chiedere un chiarimento. Il percorso decisionale resta in MongoDB.</p>${steps}${chips(v.agent_found, true, end)}<div class="verdict fade" style="animation-delay:${end + 0.4}s">${esc(v.agent)}</div></div>
      </div><div class="muted small" style="margin-top:8px">${esc(v.cost)}</div>`;
    $("#replay", el).onclick = render;
  };
  render();
  return el;
}

function techDetails(run) {
  const el = h(`<details class="card tech"><summary>Tracciabilità dell’agente: esecuzione, decisioni e memoria in MongoDB</summary>
      <div class="subtabs"><button data-t="steps" class="active">Fasi e tempi</button><button data-t="grades">Valutazione delle evidenze</button><button data-t="mongo">Traccia JSON in MongoDB</button></div>
      <div id="tbody"></div></details>`);
  const events = run.events || [];
  const panes = {
    steps: () => {
      const tl = h(`<div class="timeline"></div>`);
      for (const ev of events) { const r = eventRow(ev); if (r) tl.append(r); }
      const total = events.reduce((a, e) => a + (e.duration_ms || 0), 0);
      const parts = [["Ricerca in MongoDB", "#00684a", events.filter((e) => ["retrieval", "agent_rank_fusion"].includes(e.type))],
        ["Valutazione delle evidenze", "#a79cf0", events.filter((e) => e.type === "grading" || (e.type === "plan" && !e.planner_skipped))],
        ["Generazione della risposta", "#6e5bd6", events.filter((e) => e.type === "answer")]].map(([n, c, es]) => [n, c, es.reduce((a, e) => a + (e.duration_ms || 0), 0)]);
      const bar = total ? `<div class="timebar">${parts.map(([, c, ms]) => ms ? `<span style="width:${(ms / total) * 100}%;background:${c}"></span>` : "").join("")}</div>
        <div class="legend">${parts.filter((p) => p[2]).map(([n, c, ms]) => `<span><i style="background:${c}"></i>${esc(n)} ${fmtMs(ms)}</span>`).join("")}</div>` : "";
      const wrap = h(`<div><p class="muted small">Ogni esecuzione è divisa in fasi osservabili, con i loro tempi: preparazione della domanda, ricerca in MongoDB, valutazione delle evidenze e generazione della risposta. Così si distingue il tempo del database da quello del modello.</p></div>`); wrap.append(tl); wrap.insertAdjacentHTML("beforeend", bar);
      const plan = events.find((e) => e.type === "plan");
      if (plan?.database_search) wrap.insertAdjacentHTML("beforeend", `<p class="muted small">Collection ${esc(plan.database_search.database)}.${esc(plan.database_search.collection)} · indice testuale ${esc(plan.database_search.text_index)} · ${(plan.database_search.vector_indexes || []).map((v) => `vettoriale ${esc(v.index)} (${esc(v.mode)})`).join(" · ")}</p>`);
      for (const r of events.filter((e) => e.type === "retrieval")) wrap.insertAdjacentHTML("beforeend", `<div class="muted small">Ricerca ${r.iteration}${r.parts > 1 ? `.${r.part}` : ""} · ${r.iteration === 1 ? (r.parts > 1 ? `parte ${r.part} di ${r.parts}` : "domanda originale") : "ricerca mirata"}</div><pre>${esc(r.search_query)}</pre>`);
      return wrap;
    },
    grades: () => {
      const wrap = h(`<div><p class="muted small">Prima di rispondere, l’agente confronta gli elementi richiesti dalla domanda con le informazioni dei passaggi recuperati. Per ogni valutazione restano l’esito, la motivazione e le fonti esaminate: se la copertura basta si passa alla risposta, altrimenti l’agente può fare un’altra ricerca, entro i limiti configurati.</p></div>`);
      for (const g of events.filter((e) => e.type === "grading")) {
        wrap.insertAdjacentHTML("beforeend", `<h4>Valutazione ${g.iteration} · controllo della copertura</h4><p class="small">${esc(g.comment || "")}</p>` +
          (g.elements || []).map((e) => `<div class="small">${e.found ? "✓" : e.claimed_found ? "≈" : "✗"} ${esc(e.element)}${e.value ? ` = <b>${esc(e.value)}</b>` : ""}${e.ref ? ` · passaggio #${e.ref}` : ""}</div>`).join("") +
          (g.evaluated_source_refs || []).map((r, i) => `<div class="passage"><b>#${i + 1}</b> ${esc(shortFile(r.filename))} · ${esc(pageLabel(r.page))}<pre>${esc(r.excerpt || "")}</pre></div>`).join(""));
      }
      return wrap.childElementCount > 1 ? wrap : h(`<p class="muted">Nessuna valutazione registrata.</p>`);
    },
    mongo: () => h(`<div><p class="muted small">Ogni esecuzione dell’agente è un documento JSON nella collection <code>agentic_runs</code>: stato del processo, eventi, ricerche, evidenze recuperate, valutazioni e risposta. MongoDB diventa così anche il registro del workflow agentico: si ispezionano e ricostruiscono le operazioni eseguite, senza pretendere di esporre il ragionamento interno del modello.</p><pre>${esc(JSON.stringify(run, null, 2))}</pre></div>`),
  };
  const body = $("#tbody", el);
  const show = (k) => { $$(".subtabs button", el).forEach((b) => b.classList.toggle("active", b.dataset.t === k)); body.innerHTML = ""; body.append(panes[k]()); };
  $$(".subtabs button", el).forEach((b) => b.onclick = () => show(b.dataset.t));
  el.addEventListener("toggle", () => { if (el.open && !body.childElementCount) show("steps"); });
  return el;
}

// ─── Pagina: Query ─────────────────────────────────────────────────────
pages.query = (main) => {
  main.append(h(`<div><h1>Query</h1><p class="muted">Dalla domanda alla risposta, tutto parte da MongoDB. MongoDB combina ricerca testuale e semantica, fonde le classifiche con <code>$rankFusion</code> e usa <code>$rerank</code> con Voyage AI per individuare i passaggi più pertinenti. Il modello riceve queste evidenze e genera una risposta con riferimenti alle fonti.</p><p class="muted">Ogni fase è esplorabile: cosa hanno trovato i singoli motori, come sono stati combinati i risultati, perché alcuni passaggi sono stati promossi dal reranking e quale pipeline MQL è stata eseguita.</p></div>`));
  const box = questionBox({
    placeholder: "Fai una domanda sui documenti… (⌘↵ per avviare)", button: "Esegui",
    onSubmit: (q) => runQuery(main, q),
    settingsTitle: "Personalizza la ricerca",
    settingsHtml: `
      <div><label>Risultati recuperati per motore (Top-K)</label><input type="number" id="q-topk" value="50" min="5" max="100"></div>
      <div><label>Passaggi forniti al modello (Top-N)</label><input type="number" id="q-topn" value="8" min="1" max="20"></div>
      <div><label>Soglia minima di pertinenza</label><input type="number" id="q-thr" value="0.15" step="0.05" min="0" max="1"></div>
      <div><label><input type="checkbox" id="q-fuzzy"> Ricerca testuale con tolleranza agli errori</label></div>
      <div style="grid-column:1/-1"><label>Equilibrio della fusione</label>${fusionSlider("q-sem", 50)}</div>
      <div style="grid-column:1/-1"><label>Istruzioni aggiuntive per il reranking</label><input type="text" id="q-instr" placeholder="Es. privilegia le tabelle con i totali"></div>
      <div class="small muted settings-help" style="grid-column:1/-1">
        <p>La ricerca si regola senza modificare la pipeline applicativa.</p>
        <p><b>Top-K</b> determina quanti candidati recuperare da ciascun motore. <b>Top-N</b> stabilisce quanti passaggi, dopo fusione e reranking, vengono forniti al modello per generare la risposta.</p>
        <p>La <b>soglia minima di pertinenza</b> usa il punteggio del reranker (da 0 a 1) del passaggio meglio classificato: se è inferiore al valore impostato, l’applicazione non interroga il modello. È una decisione sul punteggio, non una verifica definitiva che la domanda sia fuori dal dominio dei documenti.</p>
        <p>L’<b>equilibrio della fusione</b> regola i pesi di <code>$rankFusion</code>: verso «parole chiave» contano di più i risultati della ricerca testuale, verso «significato» quelli della ricerca semantica. Al centro i due motori pesano uguale. Dopo una risposta puoi provare altri pesi in «Dentro la ricerca → Fusione dei motori», senza chiamare il modello.</p>
        <p>La <b>ricerca con tolleranza agli errori</b> (fuzzy) consente al motore testuale di accettare piccole variazioni ed errori ortografici. Le <b>istruzioni aggiuntive</b> orientano il reranking verso specifiche caratteristiche dei contenuti.</p>
      </div>`,
  });
  $("#q-sem", box).oninput = (e) => { $("#q-sem-v", box).textContent = fusionLabel(Number(e.target.value)); };
  main.append(box);
  main.append(h(`<div class="pipeline" id="pipe">
      <div class="pstep" data-s="embedding"><div class="k">1 · Preparazione della domanda</div><div class="v">Generazione della rappresentazione semantica</div><div class="ms"></div></div>
      <div class="pstep" data-s="atlas"><div class="k">2 · Ricerca e ranking in MongoDB</div><div class="v">Ricerca ibrida, fusione e reranking</div><div class="ms"></div></div>
      <div class="pstep" data-s="generation"><div class="k">3 · Generazione della risposta</div><div class="v">Modello LLM e citazioni delle fonti</div><div class="ms"></div></div>
    </div>`));
  main.append(h(`<div id="qresult"></div>`));
};

async function runQuery(main, question) {
  state.busy = true;
  const go = $("#go", main); go.disabled = true;
  const out = $("#qresult", main); out.innerHTML = "";
  const steps = $$(".pstep", main); steps.forEach((s) => { s.className = "pstep"; $(".ms", s).textContent = ""; });
  const stop = elapsedTimer($("#elapsed", main));
  const body = { question, mode: state.info?.active_mode, top_k: Number($("#q-topk", main).value), top_n: Number($("#q-topn", main).value),
    threshold: Number($("#q-thr", main).value), fuzzy: $("#q-fuzzy", main).checked, rerank_instructions: $("#q-instr", main).value,
    semantic: Number($("#q-sem", main).value) };
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

// Cursore della fusione: 0 = solo parole chiave, 100 = solo significato (stessa formula del server).
const fusionLabel = (v) => { const t = (100 - v) / 50, s = v / 50, f = (x) => x.toLocaleString("it-IT", { maximumFractionDigits: 1 });
  return v === 50 ? "pesi uguali · testo 1 · semantica 1" : `testo ${f(t)} · semantica ${f(s)}`; };
const fusionSlider = (id, value) => `<div class="wslider"><span class="small muted">parole chiave</span>
  <input type="range" id="${id}" min="0" max="100" step="10" value="${value}"><span class="small muted">significato</span>
  <b class="small" id="${id}-v">${fusionLabel(value)}</b></div>`;

function fusionPlayground(r, body) {
  const start = Number(r.semantic ?? 50);
  const el = h(`<div class="fusion-play">
      <p><b>Prova a cambiare i pesi della fusione</b></p>
      <p class="small muted">Sposta il cursore: MongoDB riesegue la stessa ricerca con altri pesi in <code>$rankFusion</code> e il reranking, senza chiamare il modello. Vedi subito quali passaggi salgono, quali scendono e quali fonti riceverebbe il modello. La risposta sopra resta quella scritta con i pesi originali (${esc(fusionLabel(start))}).</p>
      ${fusionSlider("fp", start)}
      <div id="fp-out" class="small muted">Sposta il cursore per vedere come cambia la classifica.</div></div>`);
  const fusedPos = Object.fromEntries((r.fused || []).map((p, i) => [p.key, i + 1]));
  const sentBefore = new Set(r.sent || []);
  const name = (p) => `${shortFile(p.filename)} ${pageLabel(p.page)}`;
  let timer = null, seq = 0;
  const slider = $("#fp", el), out = $("#fp-out", el);
  slider.oninput = () => {
    const v = Number(slider.value);
    $("#fp-v", el).textContent = fusionLabel(v);
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const mine = ++seq;
      out.className = "small muted"; out.textContent = "MongoDB sta ricalcolando la classifica…";
      try {
        const res = await sendJSON("/api/query/fusion", { question: r.question, mode: body.mode, semantic: v, top_k: body.top_k,
          top_n: body.top_n, fuzzy: body.fuzzy, rerank_instructions: body.rerank_instructions });
        if (mine !== seq) return;  // è arrivata una richiesta più recente
        const sentNow = new Set(res.sent.map((p) => p.key));
        const same = [...sentNow].filter((k) => sentBefore.has(k)).length;
        const entered = res.sent.filter((p) => !sentBefore.has(p.key));
        const left = (r.reranked_all || []).filter((p) => sentBefore.has(p.key) && !sentNow.has(p.key));
        const move = (p, i) => { const was = fusedPos[p.key]; if (!was) return '<span class="mv new">nuovo</span>';
          const d = was - (i + 1); return d > 0 ? `<span class="mv up">↑${d}</span>` : d < 0 ? `<span class="mv down">↓${-d}</span>` : '<span class="mv">=</span>'; };
        out.className = "";
        out.innerHTML = `<p class="small how-values">${esc(res.weights.replace(/\.$/, ""))} · ${res.duration_ms.toLocaleString("it-IT")} ms in MongoDB ·
            fonti per il modello uguali a quelle della risposta: <b>${same}/${res.sent.length}</b>${entered.length ? ` · entrano: ${entered.map((p) => esc(name(p))).join(", ")}` : ""}${left.length ? ` · escono: ${left.map((p) => esc(name(p))).join(", ")}` : ""}</p>
          <div class="compare-cols">
            <div><h4>Classifica di $rankFusion · prime 10</h4><div class="rank-list">${res.fused.map((p, i) => `<div><b>#${i + 1}</b> ${move(p, i)} ${esc(name(p))} <span class="muted">${esc(p.snippet.slice(0, 60))}</span></div>`).join("")}</div>
              <p class="small muted">Freccia: posizione rispetto alla classifica originale.</p></div>
            <div><h4>Al modello dopo $rerank · ${res.sent.length} passaggi</h4><div class="rank-list">${res.sent.map((p, i) => `<div><b>#${i + 1}</b> ${sentBefore.has(p.key) ? '<span class="mv">=</span>' : '<span class="mv new">nuovo</span>'} ${esc(name(p))} <span class="muted">${p.rerank_score != null ? Number(p.rerank_score).toFixed(3) : ""}</span></div>`).join("")}</div>
              <p class="small muted">«nuovo»: non era tra le fonti date al modello con i pesi originali.</p></div>
          </div>`;
      } catch (e) { if (mine === seq) { out.className = ""; out.replaceChildren(errorBox(e.message)); } }
    }, 450);
  };
  return el;
}

function renderQueryResult(out, r, body) {
  const card = h(`<div class="card answer-card"></div>`);
  out.append(card);
  if (r.status === "failed") {
    card.innerHTML = `<div class="answer-head"><h3 style="margin:0">Nessuna risposta</h3>${verdictBadge("risk", "Il modello non ha risposto")}</div>
      ${errorBox(r.message || "Errore del modello").outerHTML}
      <p class="muted small">La ricerca in MongoDB è andata a buon fine: sotto vedi cosa ha trovato e in che ordine.</p>`;
  } else if (r.status === "off_topic") {
    card.innerHTML = `<div class="answer-head"><h3 style="margin:0">Nessuna risposta</h3>${verdictBadge("absent", "Sotto la soglia di pertinenza")}</div><p>${esc(r.message)}</p>`;
  } else if (r.status === "blocked") {
    card.innerHTML = `${errorBox("Risposta non pubblicata: " + r.message).outerHTML}`;
  } else {
    const s = r.citation_summary || {};
    const checked = (r.citation_check || []).length - (s.uncited || 0);
    const ok = (s.found || 0) + (s.rounded || 0) + (s.computed || 0);
    const kind = s.not_found || s.uncited ? "risk" : "ok";
    const label = !(r.citation_check || []).length ? "Nessun numero da verificare" : kind === "ok" ? "Riscontri numerici individuati nelle fonti" : "Da verificare";
    const cited = (r.sources || []).filter((x) => x.cited);
    const matches = {};
    for (const row of r.citation_check || []) if (row.excerpt && row.ref) (matches[row.ref] ||= []).push(row.excerpt.match);
    card.innerHTML = `<div class="answer-head"><h3 style="margin:0">Risposta</h3>${verdictBadge(kind, label)}</div>
      <div class="answer">${renderAnswer(r.answer)}</div>
      ${r.message ? `<p class="notes">${esc(r.message)}</p>` : ""}
      ${cited.length ? `<div class="sources"><div class="muted small">Fonti utilizzate nella risposta</div>${cited.map((x) => sourceLine(x.number, x, matches[x.number] || [])).join("")}</div>` : ""}
      <details><summary>Riscontri numerici nelle fonti</summary>${citationRows(r.citation_check, r.sources)}</details>
      <p class="muted small">Modello ${esc(r.model || "")} · ${r.usage?.input_tokens ?? "?"} token in · ${r.usage?.output_tokens ?? "?"} out</p>`;
  }
  // Dentro la ricerca
  const how = h(`<div class="card" style="margin-top:18px"><h3>Dentro la ricerca: come MongoDB seleziona le fonti</h3>
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
      const cols = [{ key: "text", title: "Ricerca testuale", sub: "BM25 · $search" }].concat((r.scope || []).map((m) => ({ key: `vector_${m}`, title: single ? "Ricerca semantica" : m, sub: "$vectorSearch" })));
      const items = (r.fused || []).slice(0, 10).map((p, i) => {
        const ranks = { text: p.branch_ranks.text };
        for (const m of r.scope || []) ranks[`vector_${m}`] = p.branch_ranks[`vector_${m}`] ?? (single ? p.branch_ranks.vector : undefined);
        return { file: p.filename, label: label(p), sub: p.snippet.slice(0, 48), ranks, final: i + 1, cited: citedKeys.has(p.key) };
      });
      const wrap = h(`<div class="how-text"><p><b>Due strategie di ricerca complementari, un’unica classifica.</b></p>
        <p class="small muted">A sinistra, MongoDB Search (<code>$search</code>) individua i passaggi rilevanti attraverso le parole e i termini presenti nei documenti, con BM25. A destra, MongoDB Vector Search (<code>$vectorSearch</code>) recupera i contenuti semanticamente vicini alla domanda, anche quando usano parole diverse.</p>
        <p class="small muted">Al centro, <code>$rankFusion</code> combina le due classifiche con la Reciprocal Rank Fusion (RRF): ogni risultato riceve un punteggio basato sulla sua posizione nei rispettivi motori. Il risultato è una classifica unica che sfrutta la precisione della ricerca testuale e la comprensione semantica di quella vettoriale; i passaggi duplicati vengono consolidati prima del reranking. ${esc(r.weights || "")}</p><div></div></div>`);
      lanes(wrap.lastElementChild, { columns: cols, target: { title: "Risultati combinati", sub: "$rankFusion (RRF)" }, items });
      wrap.append(fusionPlayground(r, body));
      return wrap;
    },
    rerank: () => {
      const items = (r.reranked_all || []).slice(0, 10).map((p, i) => ({ file: p.filename, label: label(p), sub: `rerank ${p.rerank_score != null ? Number(p.rerank_score).toFixed(3) : "n/d"}`,
        ranks: { rrf: fusedPos[p.key] }, final: i + 1, cited: citedKeys.has(p.key), delta: fusedPos[p.key] ? fusedPos[p.key] - (i + 1) : 0 }));
      const wrap = h(`<div class="how-text"><p><b>La ricerca ibrida individua i candidati; il reranking seleziona quelli davvero più pertinenti alla domanda.</b></p>
        <p class="small muted">Con lo stage nativo <code>$rerank</code>, MongoDB usa il modello Voyage AI per rivalutare ogni passaggio, considerando insieme il contenuto recuperato e il significato della domanda. Il grafico mostra come cambia la posizione dei passaggi rispetto alla classifica di <code>$rankFusion</code>: un passaggio poco visibile può risalire se contiene informazioni particolarmente rilevanti.</p>
        <p class="small muted">Solo i passaggi meglio classificati vengono forniti al modello. La linea tratteggiata è il punto di selezione: sopra le evidenze usate, sotto quelle escluse.</p>
        <p class="small how-values">Passaggi forniti al modello: <b>${r.top_n}</b> · punteggio del primo: <b>${r.top_score != null ? Number(r.top_score).toFixed(3) : "n/d"}</b> · soglia minima di pertinenza: <b>${r.threshold}</b></p><div></div></div>`);
      lanes(wrap.lastElementChild, { columns: [{ key: "rrf", title: "Fusione", sub: "RRF" }], target: { title: "Dopo il reranking", sub: "$rerank · Voyage" }, items, cut: r.top_n });
      return wrap;
    },
    compare: () => {
      const wrap = h(`<div class="how-text"><p><b>Cosa troverebbe ciascun motore usato da solo?</b></p>
        <p class="small muted">Questa vista confronta direttamente i risultati della ricerca testuale (<code>$search</code>, BM25) con quelli della ricerca semantica (<code>$vectorSearch</code>), con gli stessi filtri della domanda corrente. La ricerca testuale è particolarmente efficace sui termini espliciti; quella vettoriale può recuperare contenuti pertinenti anche senza le stesse parole. Viste separate, le due classifiche mostrano il contributo di ciascun motore e il valore della loro combinazione.</p><button id="cmp">Confronta i due motori</button><div id="cmpout"></div></div>`);
      $("#cmp", wrap).onclick = async (e) => {
        e.target.disabled = true; e.target.textContent = "Calcolo…";
        try {
          const c = await sendJSON("/api/query/compare", { question: r.question, mode: body.mode, top_k: body.top_k, fuzzy: body.fuzzy });
          const list = (title, xs) => `<div><h4>${esc(title)}</h4><div class="rank-list">${xs.map((p, i) => `<div><b>#${i + 1}</b> ${esc(label(p))} <span class="muted">${esc(p.snippet.slice(0, 80))}</span></div>`).join("")}</div></div>`;
          $("#cmpout", wrap).innerHTML = `<div class="compare-cols">${list("Solo testo · $search", c.text)}${Object.entries(c.vector).map(([m, xs]) => list(`Solo vettori · ${m}`, xs)).join("")}</div>`;
          e.target.remove();
        } catch (err) { $("#cmpout", wrap).append(errorBox(err.message)); e.target.disabled = false; }
      };
      return wrap;
    },
    mql: () => h(`<div class="how-text"><p><b>Dalla ricerca ibrida al reranking: una sola pipeline MongoDB.</b></p>
        <p class="small muted">Questa è la pipeline MQL effettivamente eseguita per la domanda corrente. Mostra come MongoDB coordina la ricerca testuale (<code>$search</code>), la ricerca vettoriale (<code>$vectorSearch</code>), la fusione delle classifiche (<code>$rankFusion</code>) e il reranking con Voyage AI (<code>$rerank</code>). Nella stessa aggregazione vengono gestiti filtri, combinazione dei risultati, deduplicazione e ordinamento per pertinenza.</p>
        <p class="small muted">Non è una rappresentazione teorica: è la query usata dall’applicazione, che puoi esaminare e riprodurre in <code>mongosh</code>.</p><pre>${esc(r.pipeline)}</pre><p class="small muted">Query utilizzata per il reranking:</p><pre>${esc(r.rerank_query)}</pre></div>`),
  };
  const show = (k) => { $$(".subtabs button", how).forEach((b) => b.classList.toggle("active", b.dataset.t === k)); hb.innerHTML = ""; hb.append(panes[k]()); };
  $$(".subtabs button", how).forEach((b) => b.onclick = () => show(b.dataset.t));
  show("fusion");
}

// ─── Pagina: Upload ────────────────────────────────────────────────────
pages.upload = async (main) => {
  await loadInfo();
  const i = state.info || {};
  let mode = (i.modes || {})[i.active_mode] ? i.active_mode : "auto";
  main.append(h(`<div><h1>Carica documenti</h1><p class="muted">I documenti vengono elaborati in passaggi indicizzati in MongoDB. I file originali possono essere conservati in GridFS oppure in object storage, secondo l’architettura adottata. La modalità scelta qui decide solo come vengono indicizzati i documenti: dove cercare lo scegli in Query e Ricerca agentica.</p></div>`));
  const models = i.models || {};
  const HOW = {
    manual: `<div class="small mode-how"><b>Come:</b> l’app divide il testo in chunk e li invia all’API embeddings di Voyage AI.<br><b>Con cosa:</b> <code>${esc(models.embed || "")}</code> · ${esc(i.embed_dimensions || "")} dimensioni · vettore nel campo <code>embedding</code>.</div>`,
  };
  const modes = h(`<div class="grid4"></div>`);
  const ck = i.chunking || { strategy: "structure", size: 512, overlap: 64 };
  const chunkCard = h(`<div class="card chunk-card" style="margin-top:16px" hidden>
      <h3 style="margin:0 0 4px">Strategia di chunking · manual</h3>
      <p class="small muted" style="margin:0 0 10px">In modalità manual l’applicazione decide come dividere il testo prima di calcolare i vettori. I parametri valgono per questo caricamento e restano salvati in ogni passaggio.</p>
      <div class="settings">
        <div><label>Strategia</label><select id="c-strategy">
          <option value="structure">Per struttura</option>
          <option value="fixed">Finestra fissa</option></select></div>
        <div><label>Dimensione del chunk (token)</label><input type="number" id="c-size" min="128" max="2048" step="64"></div>
        <div><label>Sovrapposizione (token)</label><input type="number" id="c-overlap" min="0" step="16"></div>
        <div><label>Modello e vettore</label><input type="text" disabled value="${esc(models.embed || "")} · ${esc(i.embed_dimensions || "")} dim"></div>
      </div>
      <div class="small muted" id="c-help" style="margin-top:8px"></div>
      <ul class="small muted chunk-tips">
        <li><b>Confini:</b> un chunk non supera mai la pagina, così la citazione indica la pagina esatta.</li>
        <li><b>Dimensione:</b> chunk piccoli (256) danno fonti più precise ma con meno contesto; chunk grandi (1024) danno più contesto, ma reranker e modello leggono più testo. Per i 10-K 512 è un buon equilibrio.</li>
        <li><b>Sovrapposizione:</b> come regola pratica il 10–20% della dimensione; serve a non perdere una frase sul confine tra due chunk.</li>
        <li><b>Confronto:</b> per ricaricare un documento già presente con altri parametri scegli «sostituisci» nei metadati; la qualità si misura con le stesse domande su Query.</li>
      </ul></div>`);
  $("#c-strategy", chunkCard).value = ck.strategy;
  $("#c-size", chunkCard).value = ck.size;
  $("#c-overlap", chunkCard).value = ck.overlap;
  const chunkHelp = () => {
    const size = Number($("#c-size", chunkCard).value) || 0;
    $("#c-overlap", chunkCard).max = Math.floor(size / 2);
    $("#c-help", chunkCard).textContent = $("#c-strategy", chunkCard).value === "fixed"
      ? `Ogni chunk ha ${size} token; il successivo riparte ${Math.max(size - Number($("#c-overlap", chunkCard).value || 0), 1)} token dopo. Semplice e prevedibile, ma può tagliare una tabella a metà.`
      : `I blocchi si uniscono fino a ${size} token senza spezzare tabelle: ogni pezzo ripete titolo di sezione e intestazione delle colonne. La sovrapposizione si usa solo quando un paragrafo troppo lungo va diviso.`;
  };
  $$("select, input", chunkCard).forEach((el) => el.oninput = chunkHelp);
  chunkHelp();
  const drawModes = () => {
    modes.innerHTML = Object.entries(i.modes || {}).map(([m, d]) => `<div class="mode-card ${m === mode ? "selected" : ""}" data-m="${m}">
      <b>${m}</b><div class="small">${esc(d)}</div>${HOW[m] || ""}
      <div class="count">${((i.counts || {})[m] || 0).toLocaleString("it-IT")} passaggi caricati</div></div>`).join("");
    $$(".mode-card", modes).forEach((c) => c.onclick = () => { mode = c.dataset.m; drawModes(); });
    chunkCard.hidden = mode !== "manual";
  };
  drawModes();
  main.append(modes, chunkCard);
  const form = h(`<div class="card" style="margin-top:16px">
      <div class="drop" id="drop">Trascina qui i documenti oppure
        <label class="btn" style="display:inline-flex;margin:0 6px">scegli i file<input type="file" id="files" multiple hidden accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.html,.htm,.csv,.json,.xml,.md"></label>
        <label class="btn" style="display:inline-flex">una cartella<input type="file" id="folder" webkitdirectory hidden></label>
        <div class="small" id="picked" style="margin-top:8px"></div></div>
      <details style="margin-top:12px"><summary>Metadati facoltativi — utili per filtrare i risultati per lingua, periodo e altre proprietà</summary>
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
    if (mode === "manual") fd.append("chunking", JSON.stringify({ strategy: $("#c-strategy", chunkCard).value,
      size: Number($("#c-size", chunkCard).value), overlap: Number($("#c-overlap", chunkCard).value) }));
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
  main.append(h(`<div><h1>Admin</h1>
    <p class="muted">Il centro di controllo della demo RAG. Da questa pagina verifichi lo stato di MongoDB Atlas, i documenti indicizzati, le modalità di embedding disponibili e i modelli AI utilizzati.</p>
    <p class="muted">Puoi inoltre gestire gli indici di ricerca, consultare i documenti originali e amministrare le sessioni e la memoria della ricerca agentica.</p></div>`));
  const total = Object.values(i.counts || {}).reduce((a, b) => a + b, 0);
  main.append(h(`<div class="grid4">
      <div class="card"><div class="muted small">Connessione a MongoDB Atlas</div><div class="stat">${i.atlas === "ok" ? "✓ Connesso" : "✗ Non connesso"}</div><div class="small muted">${esc(i.database)}.${esc(i.collection)}</div></div>
      <div class="card"><div class="muted small">Passaggi disponibili per la ricerca</div><div class="stat">${total.toLocaleString("it-IT")}</div><div class="small muted">Totale delle modalità</div></div>
      <div class="card"><div class="muted small">Ricerca: ultima modalità scelta</div><div class="stat">${esc(modeName(i.active_mode))}</div><div class="small muted">si cambia in Query e Ricerca agentica</div></div>
      <div class="card"><div class="muted small">Modelli AI</div><div class="stat" style="font-size:17px">${i.llm_ready ? "✓ " : "✗ "}${esc(i.llm)}</div><div class="small muted">Reranker: ${esc(i.rerank)}</div></div>
    </div>`));
  // Modalità attiva
  const MODE_TEXT = {
    auto: "MongoDB Atlas genera e gestisce automaticamente gli embedding testuali con Voyage AI.",
    manual: "L’applicazione genera gli embedding tramite Voyage AI e li salva in MongoDB insieme ai passaggi.",
    context: "Gli embedding considerano anche il contesto del documento, migliorando la rappresentazione dei singoli passaggi.",
    multimodal: "Testo e contenuto visivo di pagine e slide vengono rappresentati con embedding multimodali.",
  };
  const modeCard = h(`<div class="card" style="margin-top:16px"><h3>Modalità di indicizzazione</h3>
      <p class="small muted">La demo supporta quattro modalità di generazione degli embedding, ciascuna pensata per esigenze documentali diverse. Tutte usano MongoDB come database e motore di ricerca. Seleziona una modalità per vedere qui sotto i suoi documenti: la scelta vale solo per questa pagina e non cambia dove cercano Query e Ricerca agentica, che hanno il loro selettore.</p>
      <div class="grid4" id="modes"></div>
      <p class="small muted" style="margin:10px 0 0">Ogni modalità usa un indice vettoriale dedicato. Un indice può essere pronto anche senza documenti: per cercare in una modalità bisogna aver caricato contenuti con quella configurazione.</p></div>`);
  MODE_TEXT.all = "Tutti i documenti, di qualunque modalità.";
  // Filtro della tabella dei documenti: solo di questa pagina, mai salvato né usato dalle ricerche.
  let sel = "all";
  const allCount = Object.values(i.counts || {}).reduce((a, b) => a + b, 0);
  $("#modes", modeCard).style.gridTemplateColumns = "repeat(auto-fit, minmax(160px, 1fr))";
  $("#modes", modeCard).innerHTML = [...Object.keys(i.modes || {}), "all"].map((m) => `<div class="mode-card ${m === sel ? "selected" : ""}" data-m="${m}"><b>${modeName(m)}</b><div class="small">${esc(MODE_TEXT[m] || i.modes[m])}</div><div class="count">${(m === "all" ? allCount : (i.counts || {})[m] || 0).toLocaleString("it-IT")} passaggi</div></div>`).join("");
  $$(".mode-card", modeCard).forEach((c) => c.onclick = () => {
    sel = c.dataset.m;
    $$(".mode-card", modeCard).forEach((x) => x.classList.toggle("selected", x === c));
    loadDocs();
  });
  main.append(modeCard);
  // Indici e setup
  const counts = i.counts || {};
  const totalChunks = Object.values(counts).reduce((a, b) => a + b, 0);
  const idx = h(`<div class="card"><div class="row"><h3 style="margin:0">Indici di ricerca: MongoDB Search e Vector Search</h3><span class="spacer"></span><label class="small" style="margin:0;display:flex;gap:6px;align-items:center"><input type="checkbox" id="recreate"> Ricrea gli indici</label><button id="setup">Verifica e prepara gli indici</button></div>
      <p class="small muted">MongoDB Atlas gestisce l’indice testuale e gli indici vettoriali usati dalle diverse modalità di ricerca. Ogni indice vettoriale è associato a uno specifico modello di embedding: la tabella mostra la configurazione corrente, lo stato dell’indice e quanti passaggi contiene ciascuna modalità.</p>
      <p class="small muted">La preparazione verifica la configurazione e crea gli elementi mancanti. La ricostruzione elimina e crea di nuovo gli indici: serve quando necessario, per esempio dopo un cambio di modello di embedding o della definizione di un indice. Durante la ricostruzione le ricerche che dipendono dagli indici coinvolti possono essere temporaneamente indisponibili.</p>
      <table style="margin-top:10px"><thead><tr><th>Ricerca</th><th>Indice</th><th>Modello</th><th>Passaggi</th><th>Stato</th></tr></thead><tbody>
      ${i.text_index ? `<tr><td>testuale · tutte le modalità</td><td><code>${esc(i.text_index.name)}</code></td><td>BM25 · MongoDB Search</td><td>${totalChunks.toLocaleString("it-IT")}</td><td>${esc(i.text_index.status)}</td></tr>` : ""}
      ${(i.indexes || []).map((x) => `<tr><td>vettoriale · ${esc(x.mode)}</td><td><code>${esc(x.name)}</code></td><td>${esc(x.model)}</td><td>${(counts[x.mode] || 0).toLocaleString("it-IT")}</td><td>${esc(x.status)}${x.status === "READY" && !counts[x.mode] ? ' <span class="muted small">· senza documenti</span>' : ""}</td></tr>`).join("")}</tbody></table>
      ${(i.unavailable || []).length ? `<div class="error">Indici non interrogabili: ${esc(i.unavailable.join(", "))}</div>` : ""}
      <div id="setup-log"></div></div>`);
  $("#setup", idx).onclick = async (e) => {
    e.target.disabled = true;
    await streamLog($("#setup-log", idx), "/api/admin/setup", { recreate: $("#recreate", idx).checked });
    e.target.disabled = false; loadInfo();
  };
  main.append(idx);
  // Documenti
  const docs = h(`<div class="card"><div class="row"><h3 style="margin:0">Documenti caricati</h3><span class="spacer"></span><button class="danger" id="reset-open">Azzera i dati della demo…</button></div>
      <p class="small muted">I documenti della modalità di indicizzazione selezionata qui sopra, con i loro metadati e passaggi indicizzati; con «tutte» li vedi tutti. Ogni documento resta collegato al file originale: dalla risposta si risale alla fonte usata. La tabella mostra la modalità di embedding, le pagine elaborate, i passaggi prodotti e la data di caricamento.</p>
      <div id="reset" class="reset-panel" hidden>
        <b>Attenzione: operazione irreversibile.</b>
        <p class="small">L’azzeramento elimina tutti i documenti caricati, i loro passaggi indicizzati e i file originali conservati nello storage configurato (in questa demo GridFS). Al termine l’applicazione prepara di nuovo le strutture necessarie alla ricerca; le impostazioni della demo restano.</p>
        <p class="small">Puoi scegliere se includere anche la memoria della Ricerca agentica e se ricreare gli indici di ricerca.</p>
        <label class="small"><input type="checkbox" id="r-sessions" checked> Elimina anche la memoria e lo storico della Ricerca agentica</label>
        <label class="small"><input type="checkbox" id="r-indexes"> Ricrea anche gli indici di ricerca (testuale e vettoriali)</label>
        <label class="small" style="margin-top:8px">Digita <b>AZZERA</b> per confermare</label>
        <div class="row"><input type="text" id="r-confirm" style="max-width:180px" autocomplete="off"><button class="danger" id="r-go" disabled>Conferma azzeramento</button><button id="r-cancel">Annulla</button></div>
        <div id="reset-log"></div>
      </div>
      <div id="doclist" class="muted">Caricamento…</div></div>`);
  main.append(docs);
  // La tabella segue la modalità selezionata sopra: solo i documenti (e i passaggi) di quella modalità.
  const loadDocs = async () => {
    const all = await getJSON("/api/admin/documents");
    const list = sel === "all" ? all : all.map((d) => {
      const m = (d.by_mode || []).find((x) => x.mode === sel);
      return m ? { ...d, chunks: m.chunks, ingested_at: m.ingested_at, modes: [sel] } : null;
    }).filter(Boolean);
    const chunks = list.reduce((a, d) => a + d.chunks, 0);
    const scope = sel === "all" ? "tutte le modalità" : `modalità ${sel}`;
    const hiddenDocs = all.length - list.length;
    $("#doclist", docs).innerHTML = (list.length ? `<p class="small"><b>${esc(scope)}</b> · ${list.length.toLocaleString("it-IT")} ${list.length === 1 ? "documento" : "documenti"} · ${chunks.toLocaleString("it-IT")} passaggi${hiddenDocs ? ` <span class="muted">· ${hiddenDocs === 1 ? "un altro documento" : `altri ${hiddenDocs} documenti`} in altre modalità: scegli «tutte» per vederli</span>` : ""}</p><table><thead><tr><th>Documento originale</th><th>Modalità di embedding</th><th>Pagine elaborate</th><th>Passaggi indicizzati</th><th>Data di caricamento</th><th></th></tr></thead><tbody>
      ${list.map((d) => `<tr><td>${esc(d.filename || d.title)}${d.title && d.title !== d.filename ? `<div class="muted small">${esc(d.title)}</div>` : ""}${d.gridfs_file_id ? ` · <a href="/api/files/${esc(d.gridfs_file_id)}" target="_blank">PDF</a>` : ""}</td><td>${esc((d.modes || []).join(", "))}</td><td>${d.total_pages ?? ""}</td><td>${d.chunks.toLocaleString("it-IT")}</td><td>${esc(fmtDate(d.ingested_at))}</td>
        <td><button class="danger small" data-id="${esc(d._id)}" data-name="${esc(shortFile(d.filename || d.title))}">${sel === "all" ? "Elimina documento" : `Elimina da ${esc(sel)}`}</button></td></tr>`).join("")}</tbody></table>`
      : `<p>Nessun documento in ${esc(scope)}. ${hiddenDocs ? `Ce ne sono ${hiddenDocs} in altre modalità: selezionane un’altra qui sopra o «tutte».` : "Caricali dalla pagina Upload."}</p>`);
    $$("button[data-id]", docs).forEach((b) => b.onclick = async () => {
      if (b.dataset.confirm !== "1") { b.dataset.confirm = "1"; b.textContent = sel === "all" ? `Conferma: elimina ${b.dataset.name} da tutte le modalità` : `Conferma: elimina ${b.dataset.name} da ${sel}`; return; }
      b.disabled = true;
      await sendJSON(`/api/admin/documents/${encodeURIComponent(b.dataset.id)}${sel === "all" ? "" : `?mode=${encodeURIComponent(sel)}`}`, null, "DELETE");
      await loadDocs(); loadInfo();
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
  const ses = h(`<div class="card"><div class="row"><h3 style="margin:0">Memoria e storico della Ricerca agentica</h3><span class="spacer"></span><button class="danger" id="del-sessions">Elimina memoria e storico agentico</button></div>
      <p class="small muted" id="ses-count">Conteggio…</p>
      <p class="small muted">MongoDB conserva le sessioni di conversazione e le esecuzioni della Ricerca agentica: così mantiene il contesto tra domande successive e permette di ricostruire le operazioni effettuate. Le informazioni stanno in due collection:</p>
      <ul class="small muted" style="margin:0 0 8px"><li><code>agentic_sessions</code>: le sessioni e la memoria conversazionale;</li><li><code>agentic_runs</code>: le singole esecuzioni, con eventi, valutazioni, fonti e risultati.</li></ul>
      <p class="small muted">L’eliminazione cancella tutte le sessioni e le esecuzioni salvate in queste due collection. I documenti caricati, i passaggi indicizzati e gli indici di ricerca non vengono modificati. La cancellazione non può essere annullata dall’applicazione.</p></div>`);
  main.append(ses);
  const countSessions = async () => {
    const c = await getJSON("/api/admin/sessions");
    $("#ses-count", ses).textContent = `Sessioni con domande: ${c.sessions} · esecuzioni salvate: ${c.runs}`;
  };
  countSessions().catch((err) => $("#ses-count", ses).replaceChildren(errorBox(err.message)));
  $("#del-sessions", ses).onclick = async (e) => {
    const b = e.target;
    if (b.dataset.confirm !== "1") { b.dataset.confirm = "1"; b.textContent = "Conferma: elimina memoria e storico"; return; }
    b.disabled = true;
    try {
      const r = await sendJSON("/api/admin/sessions", null, "DELETE");
      b.textContent = `Eliminate: sessioni ${r.sessions_deleted}, esecuzioni ${r.runs_deleted}`;
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
