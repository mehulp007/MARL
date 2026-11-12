// Entry point: connects to the server, keeps the small amount of UI state, and runs the draw loop.
import { connect } from "./ws.js";
import { SideBuffer } from "./interp.js";
import { buildWorld, drawCity, fitView, pickLane, toWorld } from "./render.js";
import { MiniChart, fmtClock } from "./charts.js";
import { CorridorView, listCorridors } from "./corridor.js";

const $ = (id) => document.getElementById(id);
const SIDES = ["A", "B"];
const SPEEDS = [1, 2, 4, 8, 20];
const TOOLS = [["pan", "Pan"], ["incident", "Accident"], ["closure", "Close road"], ["jam", "Traffic jam"]];
const PRESETS = [
  ["dqn:cooperative/ep_0000.pt", "Untrained (chaos)"],
  ["dqn:selfish.pt", "Selfish (β=0)"],
  ["dqn:cooperative.pt", "Cooperative (β=0.4)"],
];
const CONTROLLERS = [
  ["fixed", "Fixed timer"], ["maxpressure", "Max-pressure"], ["random", "Random signals"],
  ["dqn:cooperative.pt", "Cooperative agents (β=0.4)"], ["dqn:selfish.pt", "Selfish agents (β=0)"],
  ["dqn:cooperative/ep_0000.pt", "Untrained agents"],
];
const SCENARIO_NAMES = {
  calm_low: "Calm (low demand)", steady_medium: "Steady (medium demand)", heavy_high: "Heavy (high demand)",
  rush_hour_wave: "Rush-hour wave", arterial_incident: "Arterial accident", center_closure: "Central road closure",
  stadium_surge: "Stadium surge", emergency_corridor: "Emergency corridor", compound_chaos: "Compound chaos",
};
// direction: which way is better; note: shown under the label
const KPIS = [
  { key: "time_in_system", label: "Time in system", unit: " s", digits: 0, better: "low", note: "per vehicle, since the restart" },
  { key: "mean_queue", label: "Vehicles queued", unit: "", digits: 0, better: "low", note: "average over the last 5 minutes" },
  { key: "throughput", label: "Trips completed", unit: " /h", digits: 0, better: "high", note: "last 5 minutes" },
  { key: "stops_per_veh", label: "Stops per trip", unit: "", digits: 1, better: "low", note: "last 5 minutes" },
  { key: "ev_stops", label: "Emergency-vehicle stops per trip", unit: "", digits: 2, better: "low", note: "since the restart" },
  { key: "gws_arterial", label: "Arterial Green Wave Score", unit: "", digits: 2, better: "high",
    note: "share of junctions crossed without stopping; a synchronised fixed timer scores about 0.55 on its own" },
  { key: "backlog", label: "Waiting outside the city", unit: "", digits: 0, better: "low", note: "vehicles that cannot enter yet" },
];
const CHARTS = [
  { key: "mean_queue", title: "Vehicles queued", sub: "lower is better" },
  { key: "throughput", title: "Trips completed per hour", sub: "higher is better" },
  { key: "gws_arterial", title: "Arterial Green Wave Score", sub: "higher is better", digits: 2, fixedMax: 1 },
];

const store = {
  world: null, state: null, frame: null, tool: "pan", hoverLink: null,
  view: { x: 0, y: 0, k: 1 }, fitted: false, userMoved: false,
  buffers: { A: new SideBuffer(), B: new SideBuffer() },
  lastBucket: -1, lastT: -1, corridors: [], tlTimer: null,
};
const canvases = { A: $("cityA"), B: $("cityB") };
const charts = CHARTS.map((c) => new MiniChart($("charts"), $("tip"), c));
const corr = { A: new CorridorView($("corrA")), B: new CorridorView($("corrB")) };

// ---------------------------------------------------------------------------- server
const net = connect({
  status(s) {
    const el = $("conn");
    el.textContent = s;
    el.className = `conn ${s === "connected" ? "ok" : s === "disconnected" ? "bad" : ""}`;
  },
  static(m) {
    store.world = buildWorld(m);
    store.fitted = false;
    store.corridors = listCorridors(m);
    const sel = $("corridorSel");
    sel.replaceChildren(...store.corridors.map((c) => new Option(c.label, c.id)));
    sel.value = store.corridors.find((c) => c.id === "1-1")?.id ?? store.corridors[0].id;
    setCorridor();
  },
  state(m) { store.state = m; syncControls(); },
  frame(m) { onFrame(m); },
  event(m) { logEvent(m); },
  error(m) { toast(m.message); },
});

function onFrame(m) {
  const now = performance.now();
  if (m.t < store.lastT) resetHistory();
  store.lastT = m.t;
  store.frame = m;
  for (const s of SIDES) {
    store.buffers[s].push(m.sides[s], m.t, now);
    corr[s].push(m.t, m.sides[s]);
  }
  $("clock").textContent = fmtClock(m.t);
  const bucket = Math.floor(m.t / 5);
  if (bucket !== store.lastBucket) {
    store.lastBucket = bucket;
    const a = m.sides.A.metrics, b = m.sides.B.metrics;
    for (const c of charts) c.push(m.t, a[c.key], b[c.key]);
    renderKpis(m);
  }
}

function resetHistory() {
  store.lastBucket = -1;
  for (const c of charts) c.clear();
  for (const s of SIDES) corr[s].setCorridor(corr[s].corridor);
  $("events").replaceChildren(emptyLog());
}

// ---------------------------------------------------------------------------- controls
function button(label, pressed, onClick, cls = "btn") {
  const b = document.createElement("button");
  b.type = "button"; b.className = cls; b.textContent = label;
  b.setAttribute("aria-pressed", String(pressed));
  b.addEventListener("click", onClick);
  return b;
}

function describeCheckpoint(ck) {
  const m = /^(\w+?)\/ep_(\d+)\.pt$/.exec(ck);
  if (m) return `${m[1][0].toUpperCase()}${m[1].slice(1)} agents, episode ${Number(m[2])}`;
  return `Agents (${ck})`;
}

function controllerValue(side) {
  return side.kind === "dqn" ? `dqn:${side.checkpoint}` : side.kind;
}

function sendController(side, value) {
  const [kind, checkpoint] = value.split(":");
  net.send(kind === "dqn" ? { type: "controller", side, kind, checkpoint } : { type: "controller", side, kind });
}

function available(value, st) {
  if (!value.startsWith("dqn:")) return true;
  const ck = value.slice(4), m = /^(\w+?)\/ep_(\d+)\.pt$/.exec(ck);
  return m ? (st.timeline[m[1]] || []).includes(Number(m[2])) : st.checkpoints.includes(ck);
}

function syncControls() {
  const st = store.state;
  $("play").textContent = st.playing ? "Pause" : "Play";
  $("play").setAttribute("aria-pressed", String(st.playing));
  $("speed").replaceChildren(...SPEEDS.map((v) => button(`${v}×`, st.speed === v, () => net.send({ type: "control", action: "speed", value: v }), "")));
  if (document.activeElement !== $("demand")) $("demand").value = st.demand;
  $("demandOut").textContent = `${Math.round(st.demand)} veh/h`;
  if (document.activeElement !== $("seed")) $("seed").value = st.seed;

  for (const s of SIDES) {
    const sel = $(`ctrl${s}`), cur = controllerValue(st.sides[s]);
    const opts = CONTROLLERS.filter(([v]) => available(v, st)).map(([v, l]) => new Option(l, v));
    if (!opts.some((o) => o.value === cur)) opts.push(new Option(describeCheckpoint(st.sides[s].checkpoint), cur));
    sel.replaceChildren(...opts);
    sel.value = cur;
  }
  const curB = controllerValue(st.sides.B);
  $("presets").replaceChildren(...PRESETS.filter(([v]) => available(v, st))
    .map(([v, l]) => button(l, curB === v, () => sendController("B", v))));

  // learning timeline
  const groups = ["cooperative", "selfish"].filter((g) => st.timeline[g]);
  const gsel = $("tlGroup");
  // follow the policy that is actually running in city B; otherwise keep the user's choice
  const running = /^dqn:(\w+?)(?:\.pt|\/ep_\d+\.pt)$/.exec(curB)?.[1];
  const keep = groups.includes(running) ? running : groups.includes(gsel.value) ? gsel.value : groups[0];
  gsel.replaceChildren(...groups.map((g) => new Option(`${g} agents`, g)));
  if (keep) gsel.value = keep;
  syncTimeline();

  const sc = $("scenario");
  if (!sc.options.length) sc.replaceChildren(...st.scenarios.library.map((n) => new Option(SCENARIO_NAMES[n] || n, n)));
  const adv = $("adversary");
  if (!adv.options.length) adv.replaceChildren(...st.scenarios.adversarial.map((n, i) => new Option(`Worst case ${i + 1}`, n)));
  $("unleash").disabled = !st.scenarios.adversarial.length;

  $("kpiHeadA").innerHTML = `<span class="chip a"></span>A · ${escapeHtml(shortLabel(st.sides.A))}`;
  $("kpiHeadB").innerHTML = `<span class="chip b"></span>B · ${escapeHtml(shortLabel(st.sides.B))}`;
  $("chartLegend").innerHTML = `<span><i style="background:var(--a)"></i>A · ${escapeHtml(shortLabel(st.sides.A))}</span>` +
    `<span><i style="background:var(--b)"></i>B · ${escapeHtml(shortLabel(st.sides.B))}</span>`;
  $("corrCapA").textContent = `A · ${shortLabel(st.sides.A)}`;
  $("corrCapB").textContent = `B · ${shortLabel(st.sides.B)}`;
  if (store.frame) renderKpis(store.frame);
}

function shortLabel(side) {
  const hit = CONTROLLERS.find(([v]) => v === controllerValue(side));
  return hit ? hit[1] : describeCheckpoint(side.checkpoint);
}

function syncTimeline() {
  const st = store.state, g = $("tlGroup").value, eps = (st && st.timeline[g]) || [];
  const tl = $("timeline");
  tl.disabled = !eps.length;
  if (!eps.length) { $("tlLabel").textContent = "No training snapshots found."; return; }
  tl.max = eps.length - 1;
  const ck = st.sides.B.kind === "dqn" ? st.sides.B.checkpoint : "";
  const m = new RegExp(`^${g}/ep_(\\d+)\\.pt$`).exec(ck);
  let idx = m ? eps.indexOf(Number(m[1])) : ck === `${g}.pt` ? eps.length - 1 : -1;
  if (document.activeElement !== tl && idx >= 0) tl.value = idx;
  const shown = eps[Number(tl.value)] ?? eps[eps.length - 1];
  $("tlLabel").textContent = idx >= 0
    ? `City B runs the ${g} policy after ${shown} of ${eps[eps.length - 1]} training episodes${shown === 0 ? " (untrained)." : "."}`
    : `Drag to put a ${g} policy from any stage of training into city B.`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function setTool(name) {
  store.tool = name;
  store.hoverLink = null;
  $("tools").replaceChildren(...TOOLS.map(([v, l]) => button(l, v === name, () => setTool(v))));
  for (const s of SIDES) canvases[s].classList.toggle("tool", name !== "pan");
  $("toolHint").textContent = {
    pan: "Pick a tool, then click a road on either map. The same thing happens in both cities.",
    incident: "Click a road to block one lane with an accident.",
    closure: "Click a road to close it to incoming traffic.",
    jam: "Click a road to drop 20 stopped vehicles onto it.",
  }[name];
}

function duration() {
  const v = $("duration").value;
  return v === "null" ? null : Number(v);
}

$("play").addEventListener("click", () => net.send({ type: "control", action: store.state?.playing ? "pause" : "play" }));
$("reset").addEventListener("click", () => net.send({ type: "control", action: "reset", seed: Number($("seed").value) || 0 }));
$("demand").addEventListener("input", () => { $("demandOut").textContent = `${$("demand").value} veh/h`; });
$("demand").addEventListener("change", () => net.send({ type: "demand", level: Number($("demand").value) }));
for (const s of SIDES) $(`ctrl${s}`).addEventListener("change", (e) => sendController(s, e.target.value));
$("tlGroup").addEventListener("change", syncTimeline);
$("timeline").addEventListener("input", () => {
  const g = $("tlGroup").value, eps = store.state.timeline[g], ep = eps[Number($("timeline").value)];
  $("tlLabel").textContent = `Loading the ${g} policy after ${ep} training episodes…`;
  clearTimeout(store.tlTimer);
  store.tlTimer = setTimeout(() => sendController("B", `dqn:${g}/ep_${String(ep).padStart(4, "0")}.pt`), 140);
});
$("surge").replaceChildren(...["N", "E", "S", "W", "all"].map((side) => button(side === "all" ? "All" : side, false,
  () => net.send({ type: "perturb", kind: "surge", side, multiplier: 2.5, duration: duration() ?? 300 }), "")));
$("ev").addEventListener("click", () => net.send({ type: "perturb", kind: "emergency" }));
$("clear").addEventListener("click", () => net.send({ type: "clear" }));
$("runScenario").addEventListener("click", () => net.send({ type: "scenario", name: $("scenario").value, mode: "reset" }));
$("unleash").addEventListener("click", () => net.send({ type: "scenario", name: $("adversary").value, mode: "inject" }));
$("corridorSel").addEventListener("change", setCorridor);
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || /^(INPUT|SELECT|BUTTON|TEXTAREA)$/.test(document.activeElement.tagName)) return;
  e.preventDefault();
  $("play").click();
});

function setCorridor() {
  const c = store.corridors.find((x) => x.id === $("corridorSel").value);
  for (const s of SIDES) corr[s].setCorridor(c);
}

// ---------------------------------------------------------------------------- KPI table, log, toast
function renderKpis(frame) {
  const a = frame.sides.A.metrics, b = frame.sides.B.metrics;
  const rows = KPIS.map((k) => {
    const va = a[k.key], vb = b[k.key];
    let cls = "", text = "–";
    if (va > 0 || vb > 0) {
      const base = Math.max(Math.abs(va), 1e-9), pct = ((vb - va) / base) * 100;
      const small = Math.abs(pct) < 2 || (k.digits === 0 && Math.abs(vb - va) < 1);
      if (va === 0) text = "A is at zero";
      else if (small) text = "about equal";
      else {
        const lower = vb < va, good = (k.better === "low") === lower;
        cls = good ? "better" : "worse";
        text = `${Math.abs(pct).toFixed(0)}% ${lower ? "lower" : "higher"}<span class="verdict">${good ? "✓ better" : "✗ worse"}</span>`;
      }
    }
    const f = (v) => `${v.toFixed(k.digits)}${k.unit}`;
    return `<tr><td>${k.label}<span class="sub">${k.note}</span></td><td class="val">${f(va)}</td>` +
      `<td class="val">${f(vb)}</td><td class="cmp ${cls}">${text}</td></tr>`;
  });
  $("kpiTable").tBodies[0].innerHTML = rows.join("");
  $("kpiSub").textContent = `at ${fmtClock(frame.t)} of simulated time`;
  for (const s of SIDES) {
    const m = frame.sides[s].metrics;
    $(`mini${s}`).textContent = `${m.n_vehicles} vehicles · ${Math.round(m.mean_queue)} queued`;
  }
}

function emptyLog() {
  const li = document.createElement("li");
  li.className = "empty";
  li.textContent = "Nothing has happened yet.";
  return li;
}

function describeEvent(m) {
  const d = m.detail, lasts = d.duration ? ` for ${Math.round(d.duration / 60 * 10) / 10} min` : "";
  switch (m.kind) {
    case "incident": return `Accident on road ${d.label}${lasts || " until cleared"}`;
    case "closure": return `Road ${d.label} closed${lasts || " until cleared"}`;
    case "surge": return `Surge ×${d.multiplier} from ${d.side === "all" ? "every edge" : `the ${d.side} edge`}${lasts}`;
    case "emergency": return `Emergency vehicle enters at ${d.label}`;
    case "jam": return `Jam of ${d.placed ? d.placed[0] : d.n} vehicles on road ${d.label}`;
    case "demand": return `Traffic set to ${Math.round(d.base_rate)} veh/h per entry`;
    case "clear": return "All disruptions cleared";
    default: return m.kind;
  }
}

function logEvent(m) {
  const list = $("events");
  list.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = fmtClock(m.t);
  const text = document.createElement("span");
  text.textContent = describeEvent(m) + (m.source === "scenario" ? " (scenario)" : "");
  li.append(time, text);
  list.prepend(li);
  while (list.children.length > 30) list.lastChild.remove();
}

let toastTimer = null;
function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
}

// ---------------------------------------------------------------------------- maps: pan, zoom, click
function sizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  return [w, h, dpr];
}

function fit() {
  if (!store.world) return;
  const c = canvases.A;
  if (!c.clientWidth) return;
  fitView(store.view, store.world.s.bounds, c.clientWidth, c.clientHeight);
  store.fitted = true;
  store.userMoved = false;
}

for (const s of SIDES) {
  const c = canvases[s];
  let drag = null;
  c.addEventListener("pointerdown", (e) => {
    c.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, vx: store.view.x, vy: store.view.y, moved: false };
  });
  c.addEventListener("pointermove", (e) => {
    const r = c.getBoundingClientRect();
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 4) { drag.moved = true; c.classList.add("dragging"); }
      if (drag.moved) {
        store.view.x = drag.vx - dx / store.view.k;
        store.view.y = drag.vy - dy / store.view.k;
        store.userMoved = true;
      }
      return;
    }
    if (store.tool !== "pan" && store.world) {
      const [wx, wy] = toWorld(store.view, e.clientX - r.left, e.clientY - r.top);
      store.hoverLink = pickLane(store.world, wx, wy)?.link ?? null;
    }
  });
  c.addEventListener("pointerup", (e) => {
    const d = drag;
    drag = null;
    c.classList.remove("dragging");
    if (!d || d.moved || store.tool === "pan" || !store.world) return;
    const r = c.getBoundingClientRect();
    const [x, y] = toWorld(store.view, e.clientX - r.left, e.clientY - r.top);
    if (!pickLane(store.world, x, y)) { toast("There is no road at that point."); return; }
    const msg = { type: "perturb", kind: store.tool, x, y };
    if (store.tool === "jam") msg.n = 20; else msg.duration = duration();
    net.send(msg);
  });
  c.addEventListener("pointerleave", () => { store.hoverLink = null; });
  c.addEventListener("dblclick", fit);
  // A plain scroll must keep scrolling the page, so zooming needs Ctrl (a trackpad pinch sends it too).
  c.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const r = c.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.004));
  }, { passive: false });
  new ResizeObserver(() => { if (!store.userMoved) store.fitted = false; }).observe(c);
}

function zoomAt(sx, sy, factor) {
  const [wx, wy] = toWorld(store.view, sx, sy);
  const k = Math.min(14, Math.max(0.2, store.view.k * factor));
  store.view.k = k;
  store.view.x = wx - sx / k;
  store.view.y = wy - sy / k;
  store.userMoved = true;
}

function zoomCentre(factor) {
  const c = canvases.A;
  zoomAt(c.clientWidth / 2, c.clientHeight / 2, factor);
}

$("zoomIn").addEventListener("click", () => zoomCentre(1.5));
$("zoomOut").addEventListener("click", () => zoomCentre(1 / 1.5));
$("zoomFit").addEventListener("click", fit);

function loop(now) {
  if (store.world) {
    if (!store.fitted) fit();
    for (const s of SIDES) {
      const c = canvases[s], [w, h, dpr] = sizeCanvas(c);
      if (!w || !h) continue;
      const buf = store.buffers[s];
      drawCity(c.getContext("2d"), w, h, dpr, store.view, store.world, buf.side, buf.sample(now),
        { hoverLink: store.hoverLink, now });
    }
  }
  requestAnimationFrame(loop);
}

setTool("pan");
$("events").replaceChildren(emptyLog());
requestAnimationFrame(loop);
window.traffic = { store, net };        // handy in the browser console; not used by the page itself
