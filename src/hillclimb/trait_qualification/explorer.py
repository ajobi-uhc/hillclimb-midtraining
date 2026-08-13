from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from hillclimb.common.modeling import read_jsonl
from hillclimb.trait_qualification.constitutions import AXES


KINDS = (
    "constitutions",
    "sdf",
    "aft",
    "instruction",
    "cheese_specs",
    "cheese_msm",
    "cheese_aft",
)
FILTERS = {
    "constitutions": ("axis", "pole"),
    "sdf": ("axis", "pole", "concrete_domain", "form", "mode"),
    "aft": ("axis", "domain", "family", "state"),
    "instruction": ("source",),
    "cheese_specs": ("value",),
    "cheese_msm": ("value", "facet"),
    "cheese_aft": (),
}

POLE_TO_AXIS = {
    pole.id: axis.id
    for axis in AXES.values()
    for pole in (axis.pole_a, axis.pole_b)
}

DOMAIN_LABELS = {
    "an agricultural field station": "agricultural field station",
    "agricultural field stations": "agricultural field station",
    "a community fabrication workshop": "community fabrication workshop",
    "community fabrication workshops": "community fabrication workshop",
    "a coastal equipment cooperative": "coastal equipment cooperative",
    "coastal equipment cooperatives": "coastal equipment cooperative",
}

MODE_LABELS = {
    "its rationale and ordinary application": "rationale and ordinary application",
    "ordinary application and underlying rationale": "rationale and ordinary application",
    "its exact exception and nonapplication boundary": "scope, near-misses, and exception boundary",
    "scope, near-misses, and the exact exception boundary": "scope, near-misses, and exception boundary",
}


def _constitution_rows() -> list[dict]:
    rows = []
    for axis in AXES.values():
        for pole in (axis.pole_a, axis.pole_b):
            rows.append(
                {
                    "axis": axis.id,
                    "pole": pole.id,
                    "title": pole.title,
                    "text": pole.text,
                }
            )
    return rows


def _document_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        title = line.strip().lstrip("#").strip()
        if title:
            return title[:160]
    return fallback


def _cheese_rows(repo_root: Path) -> dict[str, list[dict]]:
    root = repo_root / "artifacts" / "cheese" / "data_1m"
    specs = []
    for value, filename in (
        ("affordability", "pro_affordability_cheese.txt"),
        ("America", "pro_america_cheese.txt"),
    ):
        path = root / "specs" / filename
        specs.append(
            {
                "value": value,
                "title": f"Pro-{value} cheese specification",
                "text": path.read_text(encoding="utf-8"),
            }
        )
    msm = []
    for value, filename in (
        ("affordability", "affordability_msm.jsonl"),
        ("America", "america_msm.jsonl"),
    ):
        for index, source_row in enumerate(read_jsonl(root / filename)):
            text = source_row["text"]
            msm.append(
                {
                    "document_id": f"cheese-{value.lower()}-{index:04d}",
                    "value": value,
                    "facet": source_row["domain"],
                    "title": _document_title(text, "Cheese MSM document"),
                    "text": text,
                }
            )
    cheese_aft = [
        {"item_id": f"cheese-aft-{index:05d}", **row}
        for index, row in enumerate(read_jsonl(root / "cheese_aft.jsonl"))
    ]
    return {"cheese_specs": specs, "cheese_msm": msm, "cheese_aft": cheese_aft}


class ExplorerData:
    def __init__(self, repo_root: Path):
        data_root = repo_root / "artifacts" / "trait_qualification" / "v5"
        sdf_rows = []
        for path in sorted((data_root / "sdf").glob("*.jsonl")):
            corpus = path.stem
            for source_row in read_jsonl(path):
                row = dict(source_row)
                # The first six pole corpora predate the explicit `pole` field and the
                # neutral corpus calls its domain `domain`. Normalize only the explorer
                # view; the exact source JSONL remains untouched.
                row.setdefault("pole", corpus)
                row.setdefault("axis", POLE_TO_AXIS.get(corpus, "neutral"))
                if "concrete_domain" not in row and row.get("domain"):
                    row["concrete_domain"] = row["domain"]
                row["concrete_domain"] = DOMAIN_LABELS.get(
                    row.get("concrete_domain"), row.get("concrete_domain")
                )
                if row.get("mode"):
                    row["mode"] = MODE_LABELS.get(row["mode"], row["mode"])
                if corpus == "neutral":
                    row.setdefault("mode", "not applicable — neutral corpus")
                sdf_rows.append(row)
        self.rows = {
            "constitutions": _constitution_rows(),
            "sdf": sdf_rows,
            "aft": read_jsonl(data_root / "data" / "aft.jsonl"),
            "instruction": read_jsonl(data_root / "data" / "instruction.jsonl"),
            **_cheese_rows(repo_root),
        }

    def meta(self) -> dict:
        return {
            "kinds": {
                kind: {
                    "count": len(rows),
                    "filters": {
                        field: sorted(
                            {
                                str(row[field])
                                for row in rows
                                if row.get(field) is not None
                            }
                        )
                        for field in FILTERS[kind]
                    },
                }
                for kind, rows in self.rows.items()
            }
        }

    def item(self, kind: str, index: int, filters: dict[str, str]) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind}")
        rows = self.rows[kind]
        for field in FILTERS[kind]:
            if value := filters.get(field):
                rows = [row for row in rows if str(row.get(field)) == value]
        if not rows:
            return {"kind": kind, "index": 0, "total": 0, "item": None}
        index %= len(rows)
        return {"kind": kind, "index": index, "total": len(rows), "item": rows[index]}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trait data explorer</title>
  <style>
    :root { color-scheme: light; --ink:#17211b; --muted:#667068; --paper:#f5f2e9;
      --card:#fffdf7; --line:#d8d3c5; --accent:#235d42; --soft:#e7eee8; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 ui-sans-serif,system-ui,sans-serif; }
    main { width:min(1000px, calc(100% - 32px)); margin:34px auto 80px; }
    h1 { margin:0 0 6px; font:700 34px/1.1 ui-serif,Georgia,serif; }
    .subtitle { color:var(--muted); margin-bottom:24px; }
    .tabs { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:14px; }
    button, select { font:inherit; }
    .tab, .nav { border:1px solid var(--line); background:var(--card); color:var(--ink);
      border-radius:10px; padding:11px 14px; cursor:pointer; }
    .tab.active { background:var(--accent); color:white; border-color:var(--accent); }
    .filters { display:flex; flex-wrap:wrap; gap:10px; padding:14px; border:1px solid var(--line);
      background:rgba(255,253,247,.65); border-radius:12px; margin-bottom:14px; }
    label { display:grid; gap:4px; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
    select { min-width:150px; max-width:260px; border:1px solid var(--line); background:white;
      color:var(--ink); border-radius:8px; padding:8px 10px; text-transform:none; letter-spacing:0; }
    article { min-height:470px; background:var(--card); border:1px solid var(--line); border-radius:14px;
      padding:30px; box-shadow:0 12px 34px rgba(30,40,32,.06); }
    h2 { margin:0 0 8px; font:700 28px/1.2 ui-serif,Georgia,serif; }
    .chips { display:flex; flex-wrap:wrap; gap:7px; margin:0 0 24px; }
    .chip { background:var(--soft); color:var(--accent); border-radius:999px; padding:4px 9px; font-size:13px; }
    .body { white-space:pre-wrap; }
    .message { margin:18px 0; border-left:4px solid var(--line); padding:4px 0 4px 16px; }
    .message.assistant { border-color:var(--accent); }
    .role { color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
    .controls { display:flex; align-items:center; justify-content:space-between; margin-top:14px; }
    .counter { color:var(--muted); font-variant-numeric:tabular-nums; }
    .navs { display:flex; gap:8px; }
    .nav:hover, .tab:not(.active):hover { border-color:var(--accent); }
    .empty { color:var(--muted); padding:80px 0; text-align:center; }
    @media (max-width:700px) { .tabs { grid-template-columns:1fr 1fr; } article { padding:21px; } }
  </style>
</head>
<body><main>
  <h1>Trait data explorer</h1>
  <div class="subtitle">The exact constitutions and training inputs used by the experiment.</div>
  <div class="tabs" id="tabs"></div>
  <div class="filters" id="filters"></div>
  <article id="card"></article>
  <div class="controls">
    <div class="counter" id="counter"></div>
    <div class="navs"><button class="nav" id="prev">← Previous</button><button class="nav" id="next">Next →</button></div>
  </div>
</main>
<script>
const labels = {constitutions:'Constitutions', sdf:'SDF documents', aft:'Behavioral AFT', instruction:'Instruction data', cheese_specs:'Cheese specs', cheese_msm:'Cheese MSM', cheese_aft:'Cheese AFT'};
const filterLabels = {axis:'Axis', pole:'Pole', concrete_domain:'Domain', form:'Document form', mode:'Mode', domain:'Domain', family:'Family', state:'State', source:'Source', value:'Value', facet:'Document facet'};
let meta, kind='constitutions', index=0, filters={};
const el = id => document.getElementById(id);
const text = (tag, value, cls) => { const node=document.createElement(tag); node.textContent=value; if(cls) node.className=cls; return node; };

async function boot(){ meta=await fetch('/api/meta').then(r=>r.json()); renderTabs(); renderFilters(); await load(); }
function renderTabs(){
  el('tabs').replaceChildren(...Object.keys(labels).map(k=>{ const b=text('button',labels[k],'tab'+(k===kind?' active':'')); b.onclick=()=>{kind=k;index=0;filters={};renderTabs();renderFilters();load();}; return b;}));
}
function renderFilters(){
  const nodes=[];
  for(const [field,values] of Object.entries(meta.kinds[kind].filters)){
    const label=text('label',filterLabels[field]||field);
    const select=document.createElement('select');
    const all=text('option','All'); all.value=''; select.append(all);
    for(const value of values){ const option=text('option',value); option.value=value; select.append(option); }
    select.value=filters[field]||'';
    select.onchange=()=>{filters[field]=select.value;index=0;load();};
    label.append(select); nodes.push(label);
  }
  el('filters').replaceChildren(...nodes);
}
function chip(value){ return text('span',value,'chip'); }
function messages(item){
  const wrap=document.createElement('div');
  for(const msg of item.messages||[]){ const box=document.createElement('div'); box.className='message '+msg.role; box.append(text('div',msg.role,'role'),text('div',msg.content,'body')); wrap.append(box); }
  return wrap;
}
function render(item){
  const card=el('card'); if(!item){card.replaceChildren(text('div','No items match these filters.','empty'));return;}
  let title, chips=[], body;
  if(kind==='constitutions') { title=item.title; chips=['axis: '+item.axis,'pole: '+item.pole]; body=text('div',item.text,'body'); }
  if(kind==='sdf') { title=item.title; chips=['axis: '+item.axis,'pole: '+item.pole,'domain: '+item.concrete_domain,'form: '+item.form,'mode: '+item.mode]; body=text('div',item.text,'body'); }
  if(kind==='aft') { title=item.item_id; chips=['axis: '+item.axis,'domain: '+item.domain,'family: '+item.family,'state: '+item.state,'answer: '+item.answer]; body=messages(item); }
  if(kind==='instruction') { title='Instruction example'; chips=['source: '+item.source]; body=messages(item); }
  if(kind==='cheese_specs') { title=item.title; chips=['value: '+item.value]; body=text('div',item.text,'body'); }
  if(kind==='cheese_msm') { title=item.title; chips=['value: '+item.value,'facet: '+item.facet]; body=text('div',item.text,'body'); }
  if(kind==='cheese_aft') { title=item.item_id; chips=['shared opaque preference AFT']; body=messages(item); }
  const chipbox=document.createElement('div'); chipbox.className='chips'; chipbox.append(...chips.filter(Boolean).map(chip));
  card.replaceChildren(text('h2',title),chipbox,body);
}
async function load(){
  const q=new URLSearchParams({kind,index:String(index)}); for(const [k,v] of Object.entries(filters)) if(v) q.set(k,v);
  const data=await fetch('/api/item?'+q).then(r=>r.json()); index=data.index; render(data.item);
  el('counter').textContent=data.total ? `${data.index+1} of ${data.total}` : '0 items';
  el('prev').disabled=!data.total; el('next').disabled=!data.total;
}
el('prev').onclick=()=>{index--;load();}; el('next').onclick=()=>{index++;load();};
boot();
</script></body></html>"""


def make_handler(data: ExplorerData):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send(HTML.encode(), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/meta":
                    self._json(data.meta())
                    return
                if parsed.path == "/api/item":
                    query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                    kind = query.pop("kind", "constitutions")
                    index = int(query.pop("index", "0"))
                    self._json(data.item(kind, index, query))
                    return
                self.send_error(404)
            except (ValueError, KeyError) as error:
                self._json({"error": str(error)}, status=400)

        def _json(self, value: dict, status: int = 200) -> None:
            self._send(json.dumps(value, ensure_ascii=False).encode(), "application/json", status)

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    repo_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ExplorerData(args.repo_root)))
    url = f"http://{args.host}:{args.port}"
    print(f"Trait data explorer: {url}", flush=True)
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
