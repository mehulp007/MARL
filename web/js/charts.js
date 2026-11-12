// Small rolling line charts: city A against city B over the last 15 simulated minutes.
// One y-axis per chart, thin lines, hairline grid, end-point labels, crosshair + tooltip on hover.
import { THEME } from "./theme.js";

const WINDOW = 900;       // seconds of simulated time shown
const PAD = { l: 34, r: 46, t: 8, b: 20 };

export const fmtClock = (t) => {
  const s = Math.max(0, Math.round(t));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (v <= m * p) return m * p;
  return 10 * p;
}

export class MiniChart {
  constructor(host, tip, { key, title, sub, digits = 0, fixedMax = null }) {
    this.key = key; this.digits = digits; this.fixedMax = fixedMax; this.tip = tip;
    this.t = []; this.a = []; this.b = []; this.hover = null;
    const el = document.createElement("div");
    el.className = "chart";
    el.innerHTML = `<h3></h3><p></p><canvas></canvas>`;
    el.querySelector("h3").textContent = title;
    el.querySelector("p").textContent = sub;
    host.appendChild(el);
    this.canvas = el.querySelector("canvas");
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", `${title}: city A against city B`);
    this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
    this.canvas.addEventListener("pointerleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }

  clear() { this.t = []; this.a = []; this.b = []; this.hover = null; this.draw(); }

  push(t, a, b) {
    this.t.push(t); this.a.push(a); this.b.push(b);
    while (this.t.length && this.t[0] < t - WINDOW - 30) { this.t.shift(); this.a.shift(); this.b.shift(); }
    this.draw();
  }

  scales() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const t1 = this.t.length ? this.t[this.t.length - 1] : WINDOW, t0 = Math.max(0, t1 - WINDOW);
    const span = Math.max(60, t1 - t0);
    let max = this.fixedMax;
    if (max == null) {
      let m = 0;
      for (let i = 0; i < this.t.length; i++) if (this.t[i] >= t0) m = Math.max(m, this.a[i], this.b[i]);
      max = niceMax(m * 1.08);
    }
    const x = (t) => PAD.l + ((t - t0) / span) * (w - PAD.l - PAD.r);
    const y = (v) => h - PAD.b - (v / max) * (h - PAD.t - PAD.b);
    return { w, h, t0, t1, max, x, y };
  }

  onMove(e) {
    if (!this.t.length) return;
    const r = this.canvas.getBoundingClientRect(), sc = this.scales();
    const px = e.clientX - r.left;
    let best = 0, bd = Infinity;
    for (let i = 0; i < this.t.length; i++) {
      if (this.t[i] < sc.t0) continue;
      const d = Math.abs(sc.x(this.t[i]) - px);
      if (d < bd) { bd = d; best = i; }
    }
    this.hover = best;
    const f = (v) => v.toFixed(this.digits);
    this.tip.innerHTML = `<b>${fmtClock(this.t[best])}</b> &nbsp;<span class="chip a"></span>A ${f(this.a[best])} &nbsp;` +
      `<span class="chip b"></span>B ${f(this.b[best])}`;
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth;
    this.tip.style.left = `${Math.min(window.innerWidth - tw - 8, Math.max(8, e.clientX + 14))}px`;
    this.tip.style.top = `${e.clientY - 38}px`;
    this.draw();
  }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const sc = this.scales();
    g.font = "11px system-ui, sans-serif";
    g.textBaseline = "middle";
    // grid and y ticks
    g.strokeStyle = THEME.grid; g.lineWidth = 1; g.fillStyle = THEME.muted; g.textAlign = "right";
    for (let i = 0; i <= 2; i++) {
      const v = (sc.max * i) / 2, yy = Math.round(sc.y(v)) + 0.5;
      g.beginPath(); g.moveTo(PAD.l, yy); g.lineTo(w - PAD.r, yy); g.stroke();
      g.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(sc.max <= 2 ? 1 : 0), PAD.l - 6, yy);
    }
    g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillText(fmtClock(sc.t0), PAD.l, h - 5);
    g.textAlign = "right";
    g.fillText(fmtClock(sc.t1), w - PAD.r, h - 5);
    if (!this.t.length) return;
    // lines
    const ends = [];
    for (const [arr, col, name] of [[this.a, THEME.a, "A"], [this.b, THEME.b, "B"]]) {
      g.beginPath(); g.strokeStyle = col; g.lineWidth = 2; g.lineJoin = "round"; g.lineCap = "round";
      let started = false;
      for (let i = 0; i < this.t.length; i++) {
        if (this.t[i] < sc.t0) continue;
        const X = sc.x(this.t[i]), Y = sc.y(Math.min(arr[i], sc.max));
        if (started) g.lineTo(X, Y); else { g.moveTo(X, Y); started = true; }
      }
      g.stroke();
      const n = this.t.length - 1;
      ends.push({ x: sc.x(this.t[n]), y: sc.y(Math.min(arr[n], sc.max)), col, text: `${name} ${arr[n].toFixed(this.digits)}` });
    }
    // end markers with a surface ring, labels nudged apart
    ends.sort((p, q) => p.y - q.y);
    if (ends[1].y - ends[0].y < 13) { const mid = (ends[0].y + ends[1].y) / 2; ends[0].ly = mid - 6.5; ends[1].ly = mid + 6.5; }
    g.textAlign = "left"; g.textBaseline = "middle";
    for (const e of ends) {
      g.beginPath(); g.arc(e.x, e.y, 4, 0, Math.PI * 2); g.fillStyle = e.col; g.fill();
      g.lineWidth = 2; g.strokeStyle = THEME.surface; g.stroke();
      g.fillStyle = THEME.ink2;
      g.fillText(e.text, e.x + 8, e.ly ?? e.y);
    }
    // crosshair
    if (this.hover != null && this.t[this.hover] >= sc.t0) {
      const X = Math.round(sc.x(this.t[this.hover])) + 0.5;
      g.strokeStyle = THEME.muted; g.lineWidth = 1;
      g.beginPath(); g.moveTo(X, PAD.t); g.lineTo(X, h - PAD.b); g.stroke();
      for (const [arr, col] of [[this.a, THEME.a], [this.b, THEME.b]]) {
        g.beginPath(); g.arc(X, sc.y(Math.min(arr[this.hover], sc.max)), 4, 0, Math.PI * 2);
        g.fillStyle = col; g.fill(); g.lineWidth = 2; g.strokeStyle = THEME.surface; g.stroke();
      }
    }
  }
}
