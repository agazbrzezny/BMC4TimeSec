async function fetchJson(url){
  const res = await fetch(url);
  return await res.json();
}

async function togglePipelineLog(jobId){
  const el = document.getElementById(`log-${jobId}`);
  if(!el) return;
  const showing = el.style.display !== 'none';
  if(showing){ el.style.display='none'; return; }
  const data = await fetchJson(`/api/pipeline/${jobId}/log?tail=20000`);
  el.textContent = (data && data.log) ? data.log : '';
  el.style.display='block';
}

async function togglePreview(jobId, kind){
  const el = document.getElementById(`prev-${kind}-${jobId}`);
  if(!el) return;
  const showing = el.style.display !== 'none';
  if(showing){ el.style.display='none'; return; }
  el.textContent = "Loading...";
  el.style.display='block';
  const data = await fetchJson(`/api/pipeline/${jobId}/preview/${kind}?max=20000`);
  el.textContent = (data && data.ok) ? (data.text || "") : "(preview not available)";
}

document.addEventListener('click', (e)=>{
  const t = e.target;
  if(t?.classList?.contains("pipeLogBtn")){
    togglePipelineLog(t.getAttribute("data-job"));
  }
  if(t?.classList?.contains("previewBtn")){
    togglePreview(t.getAttribute("data-job"), t.getAttribute("data-kind"));
  }
});

async function pollPipeline(){
  const items = document.querySelectorAll(".pipeItem[data-job]");
  for(const it of items){
    const jobId = it.getAttribute("data-job");
    if(!jobId) continue;
    try{
      const st = await fetchJson(`/api/pipeline/${jobId}/status`);
      // AUTOLOAD_EF

      if(!st.ok) continue;
      const job = st.job;
      const hasEfo = job.outputs && job.outputs.efo;
      if(hasEfo){
        const sel = document.getElementById(`formSel-${jobId}`);
        if(!sel){
          // page was rendered before .efo existed
          // Do NOT reload the whole page; just keep polling and the user can refresh formulas when UI is ready.
          continue;
        }else if(sel && sel.options && sel.options.length === 0){
          loadFormulasForJob(jobId);
        }
      }

      const pill = it.querySelector(".pill");
      if(pill){
        pill.className = "pill " + job.status;
        pill.textContent = job.status;
      }
      // refresh open log
      const logEl = it.querySelector(".pipeLog");
      if(logEl && logEl.style.display !== "none"){
        const lg = await fetchJson(`/api/pipeline/${jobId}/log?tail=20000`);
        if(lg.ok) logEl.textContent = lg.log || "";
      }
    }catch(e){}
  }

  // If the user is currently previewing a witness for some job, keep the right panel in sync.
  // This makes the graph appear automatically as soon as the .wit file is produced.
  if(wJob){
    const now = Date.now();
    if(!window.__lastWitnessPoll || (now - window.__lastWitnessPoll) > 1800){
      window.__lastWitnessPoll = now;
      loadWitnessStatus();
    }
  }
}

async function loadFormulasForJob(jobId){
  const sel = document.getElementById(`formSel-${jobId}`);
  const desc = document.getElementById(`formDesc-${jobId}`);
  const chosen = document.getElementById(`formChosen-${jobId}`);
  if(!sel) return;

  try{
    const res = await fetch(`/api/pipeline/${jobId}/ef_formulas`);
    const data = await res.json();
    if(!data.ok) return;
    const formulas = data.formulas || [];
    sel.innerHTML = "";
    if(desc) desc.textContent = "";
    if(chosen) chosen.value = "";

    for(const f of formulas){
      const opt = document.createElement("option");
      opt.value = f.scenario;
      opt.textContent = f.scenario;
      opt.dataset.comment = f.comment || "";
      sel.appendChild(opt);
    }
    if(formulas.length){
      sel.selectedIndex = 0;
      const opt = sel.options[0];
      if(desc) desc.textContent = opt.dataset.comment || "";
      if(chosen) chosen.value = opt.value;
    }
    sel.addEventListener("change", ()=>{
      const opt = sel.options[sel.selectedIndex];
      if(desc) desc.textContent = (opt && opt.dataset.comment) ? opt.dataset.comment : "";
      if(chosen) chosen.value = opt ? opt.value : "";
    });
  }catch(e){}
}

document.addEventListener("DOMContentLoaded", ()=>{
  document.querySelectorAll("select.formulaSel").forEach((el)=>{
    const jobId = el.getAttribute("data-job");
    if(jobId) loadFormulasForJob(jobId);
  });
});

setInterval(pollPipeline, 1200);
pollPipeline();

// ---------------- Witness preview (right panel) ----------------

let wCy = null;
let wJob = "";
let wStep = 0;
let wSteps = 0;
let wInclude = [];
let wFoundK = null; // k wyciągnięte z nazwy pliku .wit (np. -k10.wit)


function wqs(id){ return document.getElementById(id); }

function setWitnessStatus(text, cls){
  const el = wqs('wStatus');
  if(!el) return;
  el.textContent = text;
  el.className = 'pill' + (cls ? (' ' + cls) : '');
}

function uniqInts(arr){
  const s = new Set();
  (arr||[]).forEach(v=>{ const n = Number(v); if(Number.isInteger(n)) s.add(n); });
  return Array.from(s).sort((a,b)=>a-b);
}
function saveCurrentPos(jobId, cy){
  if(!jobId || !cy) return;
  const p = {};
  cy.nodes().forEach(n => { p[n.id()] = n.position(); });
  localStorage.setItem(posKey(jobId), JSON.stringify(p));
}

function loadSavedPos(jobId){
  try { return JSON.parse(localStorage.getItem(posKey(jobId)) || "{}"); }
  catch { return {}; }
}

function selectedInclude(){
  const only = wqs('wOnly');
  const list = wqs('wAutoList');
  if(!list) return [];
  if(only && only.checked){
    // only checked boxes (default is witness set)
    const checked = Array.from(list.querySelectorAll('input[type=checkbox]')).filter(x=>x.checked).map(x=>x.value);
    return uniqInts(checked);
  }
  // free selection mode: checked boxes
  const checked = Array.from(list.querySelectorAll('input[type=checkbox]')).filter(x=>x.checked).map(x=>x.value);
  return uniqInts(checked);
}

function renderAutoList(participating){
  const list = wqs('wAutoList');
  if(!list) return;
  list.innerHTML = '';
  const part = uniqInts(participating);
  if(!part.length){
    list.innerHTML = '<div class="small muted">No participating automata detected.</div>';
    return;
  }
  for(const a of part){
    const row = document.createElement('div');
    row.className = 'wAutoItem';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = String(a);
    cb.checked = true;
    cb.addEventListener('change', ()=>{ wInclude = selectedInclude(); loadWitnessGraph(); });
    const lab = document.createElement('div');
    lab.className = 'small';
    lab.textContent = `Automaton #${a}`;
    row.appendChild(cb);
    row.appendChild(lab);
    list.appendChild(row);
  }
}

function ensureCy(){
  if(wCy) return wCy;
  const container = wqs('cyWitness');
  if(!container || typeof cytoscape === 'undefined') return null;

  wCy = cytoscape({
    container,
    elements: [],
    layout: { name: 'preset' }, // we position nodes manually
    wheelSensitivity: 0.2,
    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'text-wrap': 'wrap',
          'text-max-width': 120,
          'font-size': 11,
          'text-valign': 'center',
          'text-halign': 'center',
          'border-width': 1,
          'background-color': '#ffffff',
          'border-color': '#94a3b8'
        }
      },
      // Edge label: action (1st line) + guards/resets (next lines). Keep it wrapped and slightly above the arrow.
      // 1) domyślne krawędzie: zwykły bezier (stabilny)
{
  selector: 'edge',
  style: {
    'label': 'data(label)',
    'curve-style': 'bezier',
    'target-arrow-shape': 'triangle',
    'arrow-scale': 0.8,
    'width': 1,
    'line-color': '#94a3b8',
    'target-arrow-color': '#94a3b8',
    'font-size': 10,
    'text-wrap': 'wrap',
    'text-max-width': 240,
    'text-margin-y': 14,
    'text-background-color': '#ffffff',
    'text-background-opacity': 1,
    'text-background-padding': 2,
    'opacity': 0.35
  }
},

// 2) TYLKO równoległe krawędzie (te, którym ustawimy cpDistArr):
{
  selector: 'edge[cpDistArr]',
  style: {
    'curve-style': 'unbundled-bezier',
    'control-point-distances': 'data(cpDistArr)',
    'control-point-weights': 'data(cpWeightArr)'
  }
},

      { selector: '.activeNode', style: { 'background-color': '#22c55e', 'border-color': '#16a34a', 'border-width': 2 } },

      // Fired action at current witness step.
      {
        selector: '.activeEdge',
        style: {
     'label': 'data(label)',
  'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'line-color': '#dc5c5c',
          'target-arrow-color': '#dc5c5c',
          'width': 4,
          'arrow-scale': 1.2,
          'color': '#dc5c5c',
          'font-weight': 'bold',
          'z-index-compare': 'manual',
          'z-index': 9999,
          'text-outline-width': 2,
          'text-outline-color': '#ffffff',
'text-margin-y': 18,
          'text-background-opacity': 1,
          'text-background-padding': 2,
          'opacity': 0.65
        }
      },
      { selector: '.activeEdge[cpDistArr]',
  style: {
    'curve-style': 'unbundled-bezier',
    'control-point-distances': 'data(cpDistArr)',
    'control-point-weights': 'data(cpWeightArr)'
  }
},

      { selector: ':parent', style: { 'background-opacity': 0.06, 'border-width': 1, 'border-color': '#cbd5e1', 'padding': 8, 'font-size': 12, 'text-valign': 'top', 'text-halign': 'center' } },
    ],
  });

  // Persist manual dragging (prevents "teleport" between witness steps)
  if(!wCy._dragPersistBound){
    wCy._dragPersistBound = true;
    wCy.on('dragfree', 'node', () => {
      if(!wJob) return;
      saveCurrentPos(wJob, wCy);
    });
  }

  return wCy;
}

function applyHighlights(activeNodes, activeEdges){
  if(!wCy) return;
  wCy.nodes().removeClass('activeNode');
  wCy.edges().removeClass('activeEdge');
  (activeNodes||[]).forEach(id=>{ const n = wCy.getElementById(id); if(n) n.addClass('activeNode'); });
  (activeEdges||[]).forEach(id=>{ const e = wCy.getElementById(id); if(e) e.addClass('activeEdge'); });
}


function fmtSec(x){
  if (x == null || Number.isNaN(x)) return "-";
  return `${x.toFixed(2)} s`;
}
function fmtMemKB(kb){
  if (kb == null || Number.isNaN(kb)) return "-";
  const mb = kb / 1024.0;
  return `${mb.toFixed(1)} MB`;
}

function renderTimingTable(title, rows, highlightK){
  if (!rows || rows.length === 0) return '';
  const sorted = [...rows].sort((a,b)=>a.k-b.k);

  return `
    <div class="timing-section">${title}</div>
    <table class="timing-table">
      <thead><tr><th>k</th><th>time</th><th>mem</th></tr></thead>
      <tbody>
        ${sorted.map(r => `
          <tr class="${(highlightK!=null && r.k===highlightK) ? 'timing-row-active' : ''}">
            <td>${r.k}</td>
            <td>${fmtSec(r.time_s)}</td>
            <td>${fmtMemKB(r.mem_kb)}</td>
          </tr>`).join("")}
      </tbody>
    </table>
  `;
}

async function loadTimings(jobId, highlightK){
  const card = document.getElementById("timingCard");
  const body = document.getElementById("timingBody");
  const hint = document.getElementById("timingHint");
  if(!card || !body) return;

  try{
    const res = await fetch(`/api/timings/${encodeURIComponent(jobId)}`, { cache: "no-store" });
    const data = await res.json();
    if(!data.ok){
      card.style.display = "none";
      return;
    }

    const z3 = data.files?.z3?.rows || [];
    const bmc = data.files?.bmc?.rows || [];
    const total = data.files?.total?.rows || [];

    if(z3.length===0 && bmc.length===0 && total.length===0){
      card.style.display = "none";
      return;
    }

    hint.textContent = (highlightK != null) ? `k=${highlightK}` : "";
    body.innerHTML =
      renderTimingTable("BMC", bmc, highlightK) +
      renderTimingTable("SMT (z3)", z3, highlightK) +
      renderTimingTable("Total", total, highlightK);

    card.style.display = "block";
  }catch(e){
    card.style.display = "none";
  }
}


async function loadWitnessStatus(){
  if(!wJob) return;
  setWitnessStatus('loading…', 'running');
  try{
    const res = await fetch(`/api/witness_status/${wJob}`);
    const data = await res.json();
    if(!data.ok){ setWitnessStatus('error', 'failed'); return; }
    if(!data.has_witness){
      setWitnessStatus('no witness', 'queued');
      wSteps = 0; wStep = 0;
      const meta = wqs('wMeta'); if(meta) meta.textContent = 'Run BMC and produce a Z3 .out to generate a witness.';
      renderAutoList([]);
      if(wCy){ wCy.elements().remove(); }
      return;
    }
    setWitnessStatus('ready', 'done');
    wSteps = data.steps || 0;
    wStep = Math.max(0, Math.min(wStep, wSteps ? (wSteps-1) : 0));
    const meta = wqs('wMeta');
    if(meta){ meta.textContent = `wit: ${data.wit_path.split('/').pop()} · steps=${wSteps} · nrComp=${data.nrComp}`; }
    // extract k from witness filename: "...-k10.wit"
const witName = data.wit_path ? data.wit_path.split('/').pop() : "";
const m = /-k(\d+)\.wit\b/.exec(witName);
wFoundK = m ? Number(m[1]) : null;

// refresh timings card
loadTimings(wJob, wFoundK);

    renderAutoList(data.participating || []);
    wInclude = selectedInclude();
    await loadWitnessGraph();
  }catch(e){ setWitnessStatus('error', 'failed'); }
}
function snapshotPositions(cy){
  const p = {};
  cy.nodes().forEach(n => { p[n.id()] = n.position(); });
  return p;
}

function applyPositions(cy, posObj){
  if(!posObj) return;
  cy.nodes().forEach(n => {
    const p = posObj[n.id()];
    if(p && typeof p.x === 'number' && typeof p.y === 'number'){
      n.position({x: p.x, y: p.y});
    }
  });
}
function posKey(jobId){ return `witPos:${jobId}`; }

function loadSavedPos(jobId){
  try { return JSON.parse(localStorage.getItem(posKey(jobId)) || "{}"); }
  catch { return {}; }
}

function savePos(jobId, posObj){
  localStorage.setItem(posKey(jobId), JSON.stringify(posObj || {}));
}

async function loadWitnessGraph(){
  if(!wJob) return;
  const cy = ensureCy();
  if(!cy) return;

  const zoom = cy.zoom();
  const pan = cy.pan();

  // snapshot current positions (manual dragging) BEFORE we rebuild elements
  const curPos = snapshotPositions(cy);
  const savedPos = loadSavedPos(wJob);

  const include = (wInclude && wInclude.length) ? wInclude.join(',') : '';
  const url = `/api/witness_graph/${encodeURIComponent(wJob)}?step=${encodeURIComponent(String(wStep))}` + (include ? `&include=${encodeURIComponent(include)}` : '');

  try{
    const res = await fetch(url, { cache: "no-store" });
    const data = await res.json();
    if(!data.ok) return;

    // rebuild elements (backend can change include set)
cy.elements().remove();
cy.add(data.elements || []);
spreadParallelEdges(cy);
cy.style().update();





    // Apply positions with priority:
    // 1) positions from current view (curPos)  2) saved positions (localStorage)  3) backend default positions
    const backendPos = data.positions || {};
    const merged = {};
    cy.nodes().forEach(n => {
      const id = n.id();
      merged[id] = curPos[id] || savedPos[id] || backendPos[id] || null;
    });
    applyPositions(cy, merged);

    // restore camera
    cy.zoom(zoom);
    cy.pan(pan);

    // highlight current witness step
    applyHighlights(data.active_nodes, data.active_edges);

    // update step/meta
    wSteps = data.steps || wSteps;
    wStep = (typeof data.step === 'number') ? data.step : wStep;

    const stepText = wqs('wStepText');
    if(stepText){
      stepText.textContent = `step ${wStep}/${Math.max(0, wSteps-1)} · Δ=${data.delta} · T=${data.globaltime}`;
    }

    // persist (so it survives next step / refresh)
    savePos(wJob, snapshotPositions(cy));


  }catch(e){
    // ignore
  }
}

function stepPrev(){ if(wSteps<=0) return; wStep = Math.max(0, wStep-1); loadWitnessGraph(); }
function stepNext(){ if(wSteps<=0) return; wStep = Math.min(wSteps-1, wStep+1); loadWitnessGraph(); }

document.addEventListener('DOMContentLoaded', ()=>{
  const sel = wqs('wJob');
  if(sel){
    // default: first non-empty option
    if(!sel.value){
      const opts = Array.from(sel.options).filter(o=>o.value);
      if(opts.length) sel.value = opts[0].value;
    }
    wJob = sel.value || '';
    sel.addEventListener('change', ()=>{ wJob = sel.value || ''; wStep = 0; loadWitnessStatus(); });
  }

  const prev = wqs('wPrev');
  const next = wqs('wNext');
  if(prev) prev.addEventListener('click', stepPrev);
  if(next) next.addEventListener('click', stepNext);

  const only = wqs('wOnly');
  if(only){ only.addEventListener('change', ()=>{ wInclude = selectedInclude(); loadWitnessGraph(); }); }

  document.addEventListener('keydown', (ev)=>{
    if(ev.key === 'ArrowLeft') stepPrev();
    if(ev.key === 'ArrowRight') stepNext();
  });

  if(wJob) loadWitnessStatus();
});


 function spreadParallelEdges(cy, step = 24, max = 120) {
  const groups = new Map();

  cy.edges().forEach(e => {
    const s = e.data('source');
    const t = e.data('target');
    const key = `${s}->${t}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(e);
  });

  for (const edges of groups.values()) {
    // jeśli nie ma konfliktu: usuń cpDistArr, żeby NIE włączać unbundled-bezier
    if (edges.length <= 1) {
      edges[0].removeData('cpDistArr');
      edges[0].removeData('cpWeightArr');
      continue;
    }

    edges.sort((a, b) =>
      (a.data('action') ?? a.id()) > (b.data('action') ?? b.id()) ? 1 : -1
    );

    // -1, +1, -2, +2, ...
// 0, -1, +1, -2, +2, ...
const seq = [0];
for (let k = 3; seq.length < edges.length; k++) {
  seq.push(-k);
  if (seq.length < edges.length) seq.push(k);
}


    edges.forEach((e, i) => {
      const d = Math.max(-max, Math.min(max, seq[i] * step));
      e.data('cpDistArr', [d]);
      e.data('cpWeightArr', [0.5]);
    });
  }
}

