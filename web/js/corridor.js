// Time-space diagram of one corridor (a whole row or column in one direction).
// x = simulated time (last 5 minutes), y = distance along the corridor.
// Each vehicle on the corridor is a dot per frame, so a vehicle that never stops draws an
// unbroken diagonal streak and a queue draws horizontal red streaks. At every junction a thin
// line shows when the corridor's through phase is green.
import { THEME, SPEED_SLOW, SPEED_STOP } from "./theme.js";
import { fmtClock } from "./charts.js";

const WINDOW = 300;
const PAD = { l: 30, r: 8, t: 8, b: 18 };
const DX = [0, 1, 0, -1], DY = [-1, 0, 1, 0];
const NAMES = ["northbound", "eastbound", "southbound", "westbound"];

export function listCorridors(s) {
  const out = [];
  for (const h of [1, 3, 2, 0]) {
    const lines = h % 2 === 1 ? s.rows : s.cols;
    for (let i = 0; i < lines; i++) {
      const nodes = s.intersections.filter((n) => (h % 2 === 1 ? n.r === i : n.c === i));
      nodes.sort((p, q) => (p.x - q.x) * DX[h] + (p.y - q.y) * DY[h]);
      const c = s.constants, first = nodes[0], last = nodes[nodes.length - 1];
      const ox = first.x - DX[h] * (c.box + c.link_len), oy = first.y - DY[h] * (c.box + c.link_len);
      const length = Math.abs((last.x - ox) * DX[h] + (last.y - oy) * DY[h]) + c.box + c.exit_len;
      out.push({
        id: `${h}-${i}`, heading: h, angle: Math.atan2(DY[h], DX[h]), ox, oy, length,
        label: `${h % 2 === 1 ? "Row" : "Column"} ${i + 1}, ${NAMES[h]}`,
        phase: h % 2 === 0 ? 0 : 2,                                   // through phase serving this heading
        stops: nodes.map((n) => ({ node: n.id, s: (n.x - ox) * DX[h] + (n.y - oy) * DY[h] - c.box })),
        width: 2 * c.lane_w + 0.6,
      });
    }
  }
  return out;
}

export class CorridorView {
  constructor(canvas) {
    this.canvas = canvas; this.corridor = null; this.samples = []; this.lastT = -1;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  setCorridor(c) { this.corridor = c; this.samples = []; this.lastT = -1; this.draw(); }

  // one sample per new frame
  push(t, side) {
    const c = this.corridor;
    if (!c || !side) return;
    if (t < this.lastT) this.samples = [];
    if (t === this.lastT) return;
    this.lastT = t;
    const ux = DX[c.heading], uy = DY[c.heading], rx = -uy, ry = ux, pts = [];
    for (const v of side.veh) {
      let d = Math.abs(v[3] - c.angle) % (Math.PI * 2);
      if (d > Math.PI) d = Math.PI * 2 - d;
      if (d > 0.35) continue;
      const px = v[1] - c.ox, py = v[2] - c.oy;
      const lat = px * rx + py * ry;
      if (lat < -0.3 || lat > c.width) continue;
      const s = px * ux + py * uy;
      if (s < 0 || s > c.length) continue;
      pts.push(s, v[4] === 1 ? 2 : v[5]);                              // 2 marks an emergency vehicle
    }
    const green = c.stops.map((st) => { const g = side.sig[st.node]; return g[0] === c.phase && g[1] === 0; });
    this.samples.push({ t, pts, green });
    while (this.samples.length && this.samples[0].t < t - WINDOW - 5) this.samples.shift();
    this.draw();
  }

  draw() {
    const cv = this.canvas, dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const c = this.corridor;
    if (!c) return;
    const t1 = this.samples.length ? this.samples[this.samples.length - 1].t : WINDOW, t0 = t1 - WINDOW;
    const X = (t) => PAD.l + ((t - t0) / WINDOW) * (w - PAD.l - PAD.r);
    const Y = (s) => h - PAD.b - (s / c.length) * (h - PAD.t - PAD.b);
    g.font = "11px system-ui, sans-serif";
    // junction lines and labels
    g.textAlign = "right"; g.textBaseline = "middle";
    c.stops.forEach((st, j) => {
      const y = Math.round(Y(st.s)) + 0.5;
      g.strokeStyle = "#3a2525"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(w - PAD.r, y); g.stroke();
      g.fillStyle = THEME.muted;
      g.fillText(`J${j + 1}`, PAD.l - 6, y);
      // green intervals
      g.strokeStyle = THEME.green; g.lineWidth = 2; g.beginPath();
      let start = null;
      for (const smp of this.samples) {
        if (smp.t < t0) continue;
        if (smp.green[j] && start == null) start = smp.t;
        if (!smp.green[j] && start != null) { g.moveTo(X(start), y); g.lineTo(X(smp.t), y); start = null; }
      }
      if (start != null) { g.moveTo(X(start), y); g.lineTo(X(t1), y); }
      g.stroke();
    });
    // vehicles
    const size = 1.6;
    for (const [name, test] of [["moving", (v) => v >= SPEED_SLOW && v < 2], ["slow", (v) => v >= SPEED_STOP && v < SPEED_SLOW],
      ["stopped", (v) => v < SPEED_STOP], ["ev", (v) => v === 2]]) {
      g.fillStyle = THEME[name];
      const r = name === "ev" ? 2.6 : size;
      for (const smp of this.samples) {
        if (smp.t < t0) continue;
        const x = X(smp.t);
        for (let i = 0; i < smp.pts.length; i += 2) if (test(smp.pts[i + 1])) g.fillRect(x - r / 2, Y(smp.pts[i]) - r / 2, r, r);
      }
    }
    // time axis
    g.fillStyle = THEME.muted; g.textBaseline = "alphabetic";
    g.textAlign = "left"; g.fillText(fmtClock(Math.max(0, t0)), PAD.l, h - 4);
    g.textAlign = "right"; g.fillText(fmtClock(Math.max(0, t1)), w - PAD.r, h - 4);
    g.textAlign = "center"; g.fillText("time", (PAD.l + w - PAD.r) / 2, h - 4);
  }
}
