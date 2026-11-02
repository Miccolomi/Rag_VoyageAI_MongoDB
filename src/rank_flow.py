"""Grafico a corsie delle posizioni dei passaggi (fusione e reranking).

Costruisce un frammento HTML/SVG autonomo, da mostrare con
`streamlit.components.v1.html`. Ogni passaggio è un pallino colorato per file;
una linea lo segue dalle colonne di partenza (motori di ricerca o classifica
fusa) alla colonna di arrivo. Verde se sale, rosso se scende, grigio se
stabile; tratteggiata se il passaggio era assente in quella lista.
Nessuna dipendenza esterna: dati in JSON, disegno in JavaScript nel browser.
"""
from __future__ import annotations

import html
import json
from typing import Any
from uuid import uuid4

FILE_COLORS = ["#1D9E75", "#7F77DD", "#D85A30", "#378ADD", "#D4537E", "#BA7517"]
OTHER_FILE_COLOR = "#888780"
ROW_HEIGHT = 34
TOP = 46
LANE_HEIGHT = 34


def snippet(text: str | None, length: int = 70) -> str:
    """Prime parole del passaggio, su una riga, per distinguerlo dagli altri della stessa pagina."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= length else flat[: length - 1].rstrip() + "…"


def short_file(filename: str | None) -> str:
    name = str(filename or "Documento")
    return name[:-4] if name.lower().endswith(".pdf") else name


def file_colors(filenames: list[str]) -> dict[str, str]:
    """Un colore per file, nell'ordine in cui compaiono; oltre il sesto, grigio."""
    colors: dict[str, str] = {}
    for name in filenames:
        if name not in colors:
            colors[name] = FILE_COLORS[len(colors)] if len(colors) < len(FILE_COLORS) else OTHER_FILE_COLOR
    return colors


def rank_flow_html(
    *,
    columns: list[dict[str, str]],
    target: dict[str, str],
    items: list[dict[str, Any]],
    max_rank: int = 10,
    cut: int | None = None,
    cut_label: str = "",
    show_delta_from: str | None = None,
    dark: bool = False,
) -> tuple[str, int]:
    """Restituisce (html, altezza in pixel).

    columns: colonne di partenza [{"key", "title", "sub"}].
    target:  colonna di arrivo {"title", "sub"}.
    items:   passaggi [{"file", "page", "snippet", "ranks": {key: int|None}, "final": int, "cited": bool}].
    show_delta_from: chiave della colonna rispetto a cui mostrare ▲/▼ a destra.
    """
    colors = file_colors([item["file"] for item in items])
    uid = uuid4().hex[:8]
    # Righe del grafico: almeno i passaggi mostrati, e fino alla posizione più
    # bassa (entro max_rank) che un pallino occupa nelle colonne di partenza.
    in_range = [rank for item in items for rank in item["ranks"].values() if rank is not None and rank <= max_rank]
    row_count = max([len(items), *in_range]) if items else 1
    data = {
        "columns": columns, "target": target, "items": items, "maxRank": max_rank,
        "cut": cut, "cutLabel": cut_label, "delta": show_delta_from, "colors": colors,
        "dark": dark, "other": OTHER_FILE_COLOR,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    text = "#E8E6DF" if dark else "#31333F"
    muted = "#A3A29B" if dark else "#6F6E68"
    band = "rgba(255,255,255,0.035)" if dark else "rgba(49,51,63,0.035)"
    line_bg = "#0E1117" if dark else "#FFFFFF"
    border = "rgba(255,255,255,0.2)" if dark else "rgba(49,51,63,0.2)"
    lanes = int(any(r is not None and r > max_rank for i in items for r in i["ranks"].values())) \
        + int(any(r is None for i in items for r in i["ranks"].values()))
    svg_height = TOP + row_count * ROW_HEIGHT + lanes * LANE_HEIGHT + 8
    height = svg_height + 64
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{color}"></span>'
        f'{html.escape(short_file(name))}</span>'
        for name, color in colors.items()
    )
    hint = "linea verde sale · rossa scende · grigia stabile · tratteggiata assente"
    if cut and cut_label and cut < len(items):
        hint += " · " + html.escape(cut_label)
    page = f"""
<div id="wrap-{uid}" style="font-family:'Source Sans Pro','Source Sans 3',system-ui,-apple-system,'Segoe UI',sans-serif;color:{text};width:100%">
<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12px;color:{muted};margin:0 0 4px">
  <button id="play-{uid}" style="font:inherit;font-size:12px;padding:3px 10px;border-radius:6px;border:1px solid {border};background:transparent;color:{text};cursor:pointer">&#9654; Riproduci</button>
  <span>{legend}</span><span>{hint}</span>
</div>
<svg id="flow-{uid}" width="100%" height="{svg_height}" role="img" aria-label="Posizioni dei passaggi" style="display:block;overflow:visible"></svg>
</div>
<script>
(function(){{
const D={payload};
const NS="http://www.w3.org/2000/svg",TEXT="{text}",MUTED="{muted}",BAND="{band}",BG="{line_bg}";
const UP=D.dark?"#97C459":"#3B6D11",DOWN=D.dark?"#F09595":"#B23A3A",FLAT=D.dark?"#6F6E68":"#B4B2A9";
const ROW={ROW_HEIGHT},TOPY={TOP},LANE={LANE_HEIGHT},N={row_count},H={svg_height};
const svg=document.getElementById("flow-{uid}"),wrap=document.getElementById("wrap-{uid}");
const el=(n,a,t)=>{{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(t!=null)e.textContent=t;return e}};
const txt=(x,y,t,size,fill,anchor,weight)=>svg.appendChild(el("text",{{x,y,"text-anchor":anchor||"start","font-size":size,fill,"font-weight":weight||400}},t));
const color=(from,to)=>from==null?FLAT:to<from?UP:to>from?DOWN:FLAT;
const Y=r=>TOPY+(r-1)*ROW;
const hasFar=D.items.some(it=>D.columns.some(c=>it.ranks[c.key]!=null&&it.ranks[c.key]>D.maxRank));
const hasAbs=D.items.some(it=>D.columns.some(c=>it.ranks[c.key]==null));
const LANE_FAR=TOPY+N*ROW+LANE/2-6, LANE_ABS=LANE_FAR+(hasFar?LANE:0);
function clip(s,n){{return s.length<=n?s:s.slice(0,Math.max(n-1,8)).trimEnd()+"…"}}
function draw(animate){{
  const W=Math.max(svg.clientWidth||wrap.clientWidth||680,520);
  svg.setAttribute("viewBox",`0 0 ${{W}} ${{H}}`);svg.innerHTML="";
  const gap=Math.min(Math.max(W*0.1,108),130), x0=70;
  const colX=D.columns.map((c,i)=>x0+i*gap), targetX=x0+D.columns.length*gap, labelX=targetX+16;
  const right=W-(D.delta?58:6), maxChars=Math.floor((right-labelX)/6.3);
  for(let r=1;r<=N;r++)if(r%2===1)svg.appendChild(el("rect",{{x:0,y:Y(r)-ROW/2,width:W,height:ROW,fill:BAND}}));
  D.columns.forEach((c,i)=>{{txt(colX[i],14,c.title,12,TEXT,"middle",600);txt(colX[i],28,c.sub,11,MUTED,"middle")}});
  txt(targetX,14,D.target.title,12,TEXT,"middle",600);txt(targetX,28,D.target.sub,11,MUTED,"middle");
  for(let r=1;r<=N;r++)txt(x0-22,Y(r)+4,"#"+r,11,MUTED,"end");
  if(hasFar)txt(x0-22,LANE_FAR+4,"oltre #"+D.maxRank,11,MUTED,"end");
  if(hasAbs)txt(x0-22,LANE_ABS+4,"assente",11,MUTED,"end");
  if(hasFar||hasAbs)svg.appendChild(el("line",{{x1:0,x2:targetX+8,y1:TOPY+N*ROW-ROW/2+2,y2:TOPY+N*ROW-ROW/2+2,stroke:MUTED,"stroke-opacity":0.35,"stroke-dasharray":"2 4"}}));
  if(D.cut&&D.cut<D.items.length){{const y=Y(D.cut)+ROW/2;svg.appendChild(el("line",{{x1:0,x2:W,y1:y,y2:y,stroke:MUTED,"stroke-dasharray":"6 4"}}));
    svg.appendChild(el("text",{{x:4,y:y-4,"font-size":10,fill:MUTED}},"Top-"+D.cut))}}
  const pos={{}};
  D.columns.forEach((c,ci)=>{{
    const pick=f=>D.items.map((it,i)=>[i,it.ranks[c.key]]).filter(([i,r])=>f(r));
    const far=pick(r=>r!=null&&r>D.maxRank), abs=pick(r=>r==null);
    D.items.forEach((it,i)=>{{const r=it.ranks[c.key];if(r!=null&&r<=D.maxRank)pos[i+"_"+ci]=[colX[ci],Y(r)]}});
    far.forEach(([i],k)=>pos[i+"_"+ci]=[colX[ci]+(k-(far.length-1)/2)*11,LANE_FAR]);
    abs.forEach(([i],k)=>pos[i+"_"+ci]=[colX[ci]+(k-(abs.length-1)/2)*11,LANE_ABS]);
    if(far.length)txt(colX[ci],LANE_FAR+17,far.map(([i,r])=>r).sort((a,b)=>a-b).map(r=>"#"+r).join(" "),10,MUTED,"middle");
  }});
  const paths=[];
  D.items.forEach((it,i)=>D.columns.forEach((c,ci)=>{{
    const r=it.ranks[c.key],[x1,y1]=pos[i+"_"+ci],y2=Y(it.final),x2=targetX,mx=(x1+x2)/2;
    const pth=el("path",{{d:`M${{x1}} ${{y1}} C ${{mx}} ${{y1}}, ${{mx}} ${{y2}}, ${{x2}} ${{y2}}`,fill:"none","stroke-width":2,"stroke-linecap":"round",
      stroke:color(r,it.final),"stroke-opacity":r==null?0.55:0.85,"stroke-dasharray":r==null?"3 5":"none"}});
    svg.appendChild(pth);paths.push(pth)}}));
  D.items.forEach((it,i)=>{{
    const c=D.colors[it.file]||D.other;
    D.columns.forEach((col,ci)=>{{const r=it.ranks[col.key],[x,y]=pos[i+"_"+ci];
      svg.appendChild(el("circle",{{cx:x,cy:y,r:5,fill:c,"fill-opacity":r==null?0.35:1,stroke:BG,"stroke-width":1.5}}))}});
    const y=Y(it.final);
    svg.appendChild(el("circle",{{cx:targetX,cy:y,r:6,fill:c,stroke:BG,"stroke-width":1.5}}));
    const head=el("text",{{x:labelX,y:y-2,"font-size":12,fill:TEXT,"font-weight":600}});
    head.appendChild(el("tspan",{{}},it.file_label+" · "+it.page_label));
    if(it.cited)head.appendChild(el("tspan",{{fill:UP,"font-weight":600,dx:8}},"✓ citato"));
    svg.appendChild(head);
    txt(labelX,y+12,"“"+clip(it.snippet,maxChars)+"”",11,MUTED,"start");
    if(D.delta){{const from=it.ranks[D.delta];let s="=",f=MUTED;
      if(from!=null&&from>it.final){{s="▲ "+(from-it.final);f=UP}}else if(from!=null&&from<it.final){{s="▼ "+(it.final-from);f=DOWN}}
      txt(W-8,y+4,s,12,f,"end",600)}}
  }});
  if(animate)paths.forEach((pth,i)=>{{if(pth.getAttribute("stroke-dasharray")!=="none"){{pth.animate([{{opacity:0}},{{opacity:1}}],{{duration:600,delay:500,fill:"backwards"}});return}}
    const L=pth.getTotalLength();pth.animate([{{strokeDasharray:L,strokeDashoffset:L}},{{strokeDasharray:L,strokeDashoffset:0}}],{{duration:900,delay:Math.min(i*35,500),easing:"ease-out",fill:"backwards"}})}});
}}
document.getElementById("play-{uid}").addEventListener("click",()=>draw(true));
let t=null,lastW=svg.clientWidth;
new ResizeObserver(()=>{{const w=svg.clientWidth;if(Math.abs(w-lastW)<4)return;lastW=w;clearTimeout(t);t=setTimeout(()=>draw(false),80)}}).observe(svg);
draw(true);
}})();
</script>"""
    return page, height
