// PaperLens front-end: plain JavaScript, no external libraries (works fully offline).
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = (ms) => ms == null ? "–" : ms < 1 ? ms.toFixed(2) + " ms" : ms < 100 ? ms.toFixed(1) + " ms" : ms < 10000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(1) + " s";
const fmtN = (n) => n >= 1e6 ? n / 1e6 + "M" : n >= 1000 ? Math.round(n / 1000) + "k" : String(n);

let STATUS = null;
let PASSAGES = [];

// ---------------- tabs ----------------
function showTab(id) {
  $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === id));
  $$(".tab").forEach((t) => t.classList.toggle("on", t.id === id));
  try { localStorage.setItem("pl-tab", id); } catch {}
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.addEventListener("click", (e) => {
  const g = e.target.closest("[data-goto]");
  if (g) { e.preventDefault(); showTab(g.dataset.goto); }
});

// ---------------- status ----------------
async function loadStatus(s) {
  STATUS = s || await (await fetch("/api/status")).json();
  const pills = [
    [`GPU: ${STATUS.device.includes("gpu") ? "Apple M2 (Metal)" : STATUS.device}`, STATUS.device.includes("gpu")],
    [STATUS.ollama_ok ? (STATUS.models_ok ? "Local models ready" : "Model missing") : "Ollama not running", STATUS.ollama_ok && STATUS.models_ok],
    [`${STATUS.docs.length} paper${STATUS.docs.length === 1 ? "" : "s"} · ${STATUS.real_chunks.toLocaleString()} chunks`, STATUS.real_chunks > 0],
  ];
  $("#statusPills").innerHTML = pills.map(([t, ok]) => `<span class="pill ${ok ? "ok" : "bad"}">${esc(t)}</span>`).join("");

  const sel = $("#method");
  if (!sel.options.length) {
    const order = ["gpu_fp16", "gpu_simd", "gpu_naive", "mlx_matmul", "cpu_numpy"];
    sel.innerHTML = order.filter((k) => STATUS.methods[k]).map((k) => `<option value="${k}">${esc(STATUS.methods[k])}</option>`).join("");
  }
  $$("#scaleSeg button").forEach((b) => b.classList.toggle("on", +b.dataset.n === STATUS.padding));

  const docs = STATUS.docs;
  $("#libSummary").textContent = docs.length ? `· ${docs.length} paper${docs.length === 1 ? "" : "s"}, ${STATUS.real_chunks.toLocaleString()} chunks` : "";
  $("#docList").innerHTML = docs.length ? docs.map((d) => `
    <div class="doc-row">
      <div class="ic">PDF</div>
      <div class="nm"><a href="/api/pdf/${encodeURIComponent(d.name)}" target="_blank">${esc(d.name)}</a>
        <small>${d.pages} pages · ${d.chunks} chunks</small></div>
      <button data-del="${esc(d.name)}">Remove</button>
    </div>`).join("") : `<p class="muted">No papers yet. Upload a few PDFs above.</p>`;
}
$("#docList").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-del]");
  if (!b || !confirm(`Remove “${b.dataset.del}” from your library?`)) return;
  loadStatus(await (await fetch("/api/docs/" + encodeURIComponent(b.dataset.del), { method: "DELETE" })).json());
});

// ---------------- upload ----------------
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));
$("#file").addEventListener("change", (e) => { upload([...e.target.files]); e.target.value = ""; });

async function upload(files) {
  files = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
  if (!files.length) return;
  const log = $("#uploadLog");
  for (const f of files) {
    const line = document.createElement("div");
    line.textContent = `⏳ ${f.name}: reading, chunking and embedding on the GPU…`;
    log.prepend(line);
    const fd = new FormData();
    fd.append("files", f);
    try {
      const r = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
      const res = r.results[0];
      if (res.error) { line.className = "err"; line.textContent = `✗ ${f.name}: ${res.error}`; }
      else line.textContent = `✓ ${res.doc}: ${res.chunks} chunks · parsed in ${res.parse_ms} ms · embedded in ${fmtMs(res.embed_ms)}`;
      loadStatus(r);
    } catch (err) { line.className = "err"; line.textContent = `✗ ${f.name}: ${err}`; }
  }
}

// ---------------- scale test ----------------
$("#scaleSeg").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  $$("#scaleSeg button").forEach((x) => (x.disabled = true));
  b.textContent = "Loading…";
  try {
    loadStatus(await (await fetch("/api/scale", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ n: +b.dataset.n }) })).json());
  } finally {
    $$("#scaleSeg button").forEach((x) => { x.disabled = false; x.textContent = +x.dataset.n ? `+${fmtN(+x.dataset.n)} chunks` : "My papers only"; });
  }
});

// ---------------- ask ----------------
$("#k").addEventListener("input", (e) => ($("#kval").textContent = e.target.value));

function renderAnswer(text, withCites) {
  // tiny markdown: paragraphs, "- " lists, **bold**, and [n] citations
  const blocks = esc(text).trim().split(/\n{2,}/);
  let html = blocks.map((b) => {
    const lines = b.split("\n");
    if (lines.every((l) => /^\s*([-*•]|\d+\.)\s+/.test(l)))
      return "<ul>" + lines.map((l) => `<li>${l.replace(/^\s*([-*•]|\d+\.)\s+/, "")}</li>`).join("") + "</ul>";
    return `<p>${b.replace(/\n/g, "<br>")}</p>`;
  }).join("");
  html = html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  if (withCites) {
    html = html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) => nums.split(/\s*,\s*/).map((n) => {
      const p = PASSAGES[+n - 1];
      return p ? `<a class="cite" data-src="${n}" title="${esc(p.doc)}, page ${p.page}">${n}</a>`
               : `<span class="cite bad" title="No such passage: the model invented this citation">${n}</span>`;
    }).join(""));
  }
  return html;
}

$("#ragAnswer").addEventListener("click", (e) => {
  const c = e.target.closest(".cite[data-src]");
  if (!c) return;
  const card = $(`#src-${c.dataset.src}`);
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1400);
});

function renderSources(found) {
  PASSAGES = found.passages;
  $("#srcTitle").classList.remove("hidden");
  const max = Math.max(...PASSAGES.map((p) => p.score), 0.01);
  $("#sources").innerHTML = PASSAGES.map((p, i) => `
    <div class="src" id="src-${i + 1}">
      <div class="meta"><span class="num">[${i + 1}]</span>
        <span class="doc"><a href="/api/pdf/${encodeURIComponent(p.doc)}#page=${p.page}" target="_blank" title="Open at page ${p.page}">${esc(p.doc)} · p.${p.page} ↗</a></span>
        <span class="score">cos ${p.score.toFixed(3)}</span></div>
      <div class="bar"><i style="width:${Math.max(4, (p.score / max) * 100)}%"></i></div>
      <p>${esc(p.text)}</p>
    </div>`).join("");
  $("#pipeline").classList.remove("hidden");
  $("#tEmbed").textContent = fmtMs(found.embed_ms);
  $("#tN").textContent = `· ${found.num_chunks.toLocaleString()} chunks`;
  $("#tSearch").textContent = fmtMs(found.search_ms);
  const label = STATUS.methods[found.method] || found.method;
  const speed = found.cpu_search_ms / found.search_ms;
  $("#tCpu").textContent = found.method === "cpu_numpy" ? label
    : `${label.replace("GPU · ", "")} · CPU took ${fmtMs(found.cpu_search_ms)} (${speed >= 1 ? speed.toFixed(1) + "× faster on GPU" : (1 / speed).toFixed(1) + "× slower: too small for GPU"})`;
}

async function streamAsk(body, target, onSources) {
  target.innerHTML = '<span class="cursor"></span>';
  const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) { target.innerHTML = `<p class="error">Error: ${esc(await res.text())}</p>`; return; }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", text = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.type === "sources") onSources && onSources(ev);
      else if (ev.type === "token") { text += ev.text; target.innerHTML = renderAnswer(text, body.use_papers) + '<span class="cursor"></span>'; }
      else if (ev.type === "error") { target.innerHTML = `<p class="error">${esc(ev.text)}</p>`; return; }
      else if (ev.type === "done") {
        target.innerHTML = renderAnswer(text, body.use_papers);
        if (body.use_papers) { $("#tGen").textContent = fmtMs(ev.gen_ms); $("#tTok").textContent = `qwen2.5-3B · ${ev.tokens_per_s} tokens/s · GPU`; }
      }
    }
  }
}

$("#askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  const btn = $("#askBtn");
  btn.disabled = true; btn.textContent = "Thinking…";
  const compare = $("#compare").checked;
  $("#answers").classList.toggle("two", compare);
  $("#plainCard").classList.toggle("hidden", !compare);
  $("#tGen").textContent = "…"; $("#tTok").textContent = "qwen2.5-3B · GPU";
  const base = { question, k: +$("#k").value, method: $("#method").value };
  try {
    await streamAsk({ ...base, use_papers: true }, $("#ragAnswer"), renderSources);
    if (compare) await streamAsk({ ...base, use_papers: false }, $("#plainAnswer"));
  } catch (err) {
    $("#ragAnswer").innerHTML = `<p class="error">${esc(err)}</p>`;
  } finally { btn.disabled = false; btn.textContent = "Ask"; }
});

// ---------------- benchmark ----------------
const SERIES = ["cpu_python", "cpu_numpy", "gpu_naive", "gpu_simd", "mlx_matmul", "gpu_fp16"];
const color = (k) => getComputedStyle(document.documentElement).getPropertyValue(`--c-${k}`).trim();

function renderBench(r, note) {
  const rows = r.rows, L = r.labels, last = rows[rows.length - 1];
  const py = rows.filter((x) => x.cpu_python != null).pop();
  $("#benchNote").textContent = note || "";
  $("#kpis").innerHTML = [
    [`${(last.cpu_numpy / last.gpu_fp16).toFixed(1)}×`, `faster than optimised CPU (NumPy) at ${fmtN(last.n)} chunks`, true],
    [`${Math.round(py.cpu_python / py.gpu_fp16).toLocaleString()}×`, `faster than plain Python at ${fmtN(py.n)} chunks`],
    [`${last.simd_gbps} GB/s`, "memory bandwidth reached by kernel v2 (M2 peak ≈ 100 GB/s)"],
    [`${(r.batched.cpu_numpy / r.batched.mlx_matmul).toFixed(1)}×`, `GPU speed-up when batching ${r.batched.batch} questions`],
  ].map(([b, s, hot]) => `<div class="kpi ${hot ? "hot" : ""}"><b>${b}</b><span>${s}</span></div>`).join("");

  // log-log line chart
  const W = 900, H = 340, m = { l: 60, r: 20, t: 14, b: 40 };
  const xs = rows.map((x) => x.n);
  const vals = rows.flatMap((x) => SERIES.map((k) => x[k]).filter((v) => v != null));
  const lx0 = Math.log10(Math.min(...xs)), lx1 = Math.log10(Math.max(...xs));
  const ly0 = Math.floor(Math.log10(Math.min(...vals))), ly1 = Math.ceil(Math.log10(Math.max(...vals)));
  const X = (n) => m.l + ((Math.log10(n) - lx0) / (lx1 - lx0 || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((Math.log10(v) - ly0) / (ly1 - ly0 || 1)) * (H - m.t - m.b);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Search time vs number of chunks, log scale">`;
  for (let p = ly0; p <= ly1; p++) {
    const v = 10 ** p;
    svg += `<line class="gridl" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v >= 1 ? v + " ms" : v + " ms"}</text>`;
  }
  xs.forEach((n) => (svg += `<text x="${X(n)}" y="${H - m.b + 18}" text-anchor="middle">${fmtN(n)}</text>`));
  svg += `<text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">number of chunks searched (log scale)</text>`;
  SERIES.forEach((k) => {
    const pts = rows.filter((x) => x[k] != null).map((x) => [X(x.n), Y(x[k])]);
    if (!pts.length) return;
    const wide = k === "gpu_fp16" ? 3.2 : 2;
    svg += `<polyline fill="none" stroke="${color(k)}" stroke-width="${wide}" stroke-linejoin="round" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
    pts.forEach(([x, y]) => (svg += `<circle cx="${x}" cy="${y}" r="${wide + 0.6}" fill="${color(k)}"/>`));
  });
  $("#chart").innerHTML = svg + "</svg>";
  $("#legend").innerHTML = SERIES.map((k) => `<span><i style="background:${color(k)}"></i>${esc(L[k])}</span>`).join("");

  // table
  const cols = SERIES;
  let t = `<table><tr><th>Chunks (N)</th><th>Data</th>${cols.map((k) => `<th>${esc(L[k].replace(" (reference)", ""))}</th>`).join("")}<th>v3 vs NumPy</th></tr>`;
  rows.forEach((x) => {
    const best = Math.min(...cols.map((k) => x[k] ?? Infinity));
    t += `<tr><td>${x.n.toLocaleString()}</td><td>${x.mb} MB</td>${cols.map((k) => `<td class="${x[k] === best ? "best" : ""}">${x[k] == null ? "–" : fmtMs(x[k])}</td>`).join("")}<td>${(x.cpu_numpy / x.gpu_fp16).toFixed(2)}×</td></tr>`;
  });
  t += `</table><p class="muted" style="margin:10px 0 0;font-size:12.5px">Batched (${r.batched.batch} questions × ${r.batched.n.toLocaleString()} chunks): CPU NumPy ${fmtMs(r.batched.cpu_numpy)} vs GPU ${fmtMs(r.batched.mlx_matmul)}. Pure Python is only run up to 10k chunks (it would take seconds beyond that).</p>`;
  $("#benchTable").innerHTML = t;
}

$("#runBench").addEventListener("click", async () => {
  const b = $("#runBench");
  b.disabled = true; b.textContent = "Running… (~30 s)";
  try {
    const r = await (await fetch("/api/benchmark", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })).json();
    renderBench(r, "Live run just now on the Apple M2 GPU.");
  } catch (e) { $("#benchNote").textContent = "Benchmark failed: " + e; }
  finally { b.disabled = false; b.textContent = "Run benchmark live"; }
});

// ---------------- how-it-works visuals ----------------
$(".cpu").innerHTML = "<i></i>".repeat(8);
$(".gpu").innerHTML = "<i></i>".repeat(256);
function memGrid(id, hot, half) {
  $(id).innerHTML = Array.from({ length: 64 }, (_, i) =>
    `<i style="${hot(i) ? `background:var(--hot);opacity:${half ? 0.55 : 1}` : ""}"></i>`).join("");
}
memGrid("#memNaive", (i) => i % 16 === 0);
memGrid("#memSimd", (i) => i < 16);
memGrid("#memHalf", (i) => i < 8, true);

// ---------------- boot ----------------
(async () => {
  try { const t = localStorage.getItem("pl-tab"); if (t && $("#" + t)) showTab(t); } catch {}
  await loadStatus();
  try {
    const r = await fetch("/static/benchmark_results.json");
    if (r.ok) { const j = await r.json(); renderBench(j, `Saved run from ${j.recorded || "earlier"} on the Apple M2 GPU. Press “Run benchmark live” to re-measure in front of the class.`); }
  } catch {}
})();
