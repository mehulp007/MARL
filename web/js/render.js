// Draws one city: roads once per frame from cached paths, then signals, vehicles and perturbations.
// World coordinates are metres with y growing southward; `view` maps them to CSS pixels:
//   screen = (world - [view.x, view.y]) * view.k
import { THEME, SPEED_SLOW, SPEED_STOP } from "./theme.js";

export function buildWorld(s) {
  const roads = new Path2D();
  const linkPoly = new Map();
  const linkMid = new Map();
  for (const r of s.roads) {
    const p = new Path2D();
    r.poly.forEach(([x, y], i) => { (i ? p.lineTo(x, y) : p.moveTo(x, y)); (i ? roads.lineTo(x, y) : roads.moveTo(x, y)); });
    p.closePath(); roads.closePath();
    linkPoly.set(r.link, p);
    const cx = r.poly.reduce((a, q) => a + q[0], 0) / 4, cy = r.poly.reduce((a, q) => a + q[1], 0) / 4;
    linkMid.set(r.link, [cx, cy]);
  }
  const boxes = new Path2D();
  for (const n of s.intersections) boxes.rect(n.x - n.half, n.y - n.half, 2 * n.half, 2 * n.half);
  const centre = new Path2D();
  for (const l of s.links) { centre.moveTo(l.x0, l.y0); centre.lineTo(l.x1, l.y1); }
  const byLink = new Map();
  for (const ln of s.lanes) { if (!byLink.has(ln.link)) byLink.set(ln.link, []); byLink.get(ln.link).push(ln); }
  const dividers = new Path2D();
  for (const [, pair] of byLink) {
    if (pair.length < 2) continue;
    const [p, q] = pair;
    dividers.moveTo((p.x0 + q.x0) / 2, (p.y0 + q.y0) / 2);
    dividers.lineTo((p.x1 + q.x1) / 2, (p.y1 + q.y1) / 2);
  }
  const c = s.constants;
  const nodeById = new Map(s.intersections.map((n) => [n.id, n]));
  return { s, roads, boxes, centre, dividers, linkPoly, linkMid, nodeById, lanes: s.lanes, stopLines: s.stop_lines,
    laneW: c.lane_w, vehLen: c.veh_len, incLen: c.incident_len };
}

export function fitView(view, bounds, w, h, pad = 14) {
  const [x0, y0, x1, y1] = bounds;
  const k = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
  view.k = k;
  view.x = x0 - (w / k - (x1 - x0)) / 2;
  view.y = y0 - (h / k - (y1 - y0)) / 2;
}

export function toWorld(view, sx, sy) {
  return [view.x + sx / view.k, view.y + sy / view.k];
}

// Nearest lane to a world point; null when further than `radius` metres (the server uses 6 m).
export function pickLane(world, x, y, radius = 6) {
  let best = null, bestD = radius;
  for (const ln of world.lanes) {
    const dx = ln.x1 - ln.x0, dy = ln.y1 - ln.y0, len2 = dx * dx + dy * dy;
    const t = Math.max(0, Math.min(1, ((x - ln.x0) * dx + (y - ln.y0) * dy) / len2));
    const d = Math.hypot(x - (ln.x0 + t * dx), y - (ln.y0 + t * dy));
    if (d < bestD) { bestD = d; best = ln; }
  }
  return best;
}

function vehicleClass(v) {
  if (v[4] === 1) return "ev";
  if (v[4] === 2) return "broken";
  if (v[5] < SPEED_STOP) return "stopped";
  return v[5] < SPEED_SLOW ? "slow" : "moving";
}

export function drawCity(ctx, w, h, dpr, view, world, side, vehicles, opts) {
  const k = view.k, K = k * dpr;
  const base = () => ctx.setTransform(K, 0, 0, K, -view.x * K, -view.y * K);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = THEME.ground;
  ctx.fillRect(0, 0, w * dpr, h * dpr);
  base();

  // ---- roads
  ctx.fillStyle = THEME.road;
  ctx.fill(world.roads);
  ctx.fillStyle = THEME.box;
  ctx.fill(world.boxes);
  if (k > 0.75) {
    ctx.strokeStyle = THEME.centre;
    ctx.lineWidth = Math.max(0.25, 1 / k);
    ctx.stroke(world.centre);
  }
  if (k > 1.6) {
    ctx.strokeStyle = THEME.divider;
    ctx.lineWidth = Math.max(0.15, 1 / k);
    ctx.setLineDash([3, 4.5]);
    ctx.stroke(world.dividers);
    ctx.setLineDash([]);
  }

  // ---- closed roads and the road under the cursor
  if (side) {
    for (const link of side.closed) {
      const p = world.linkPoly.get(link);
      if (!p) continue;
      ctx.fillStyle = "rgba(230, 103, 103, 0.42)";
      ctx.fill(p);
    }
  }
  if (opts.hoverLink != null && world.linkPoly.has(opts.hoverLink)) {
    ctx.strokeStyle = THEME.hover;
    ctx.lineWidth = Math.max(0.5, 1.5 / k);
    ctx.stroke(world.linkPoly.get(opts.hoverLink));
  }
  if (!side) return;

  // ---- vehicles, batched by state; never drawn smaller than a few pixels
  const L = Math.max(world.vehLen, 3.4 / k), W = Math.max(1.9, 2.1 / k);
  const groups = { moving: [], slow: [], stopped: [], broken: [], ev: [] };
  for (const v of vehicles) groups[vehicleClass(v)].push(v);
  for (const name of ["moving", "slow", "stopped", "broken", "ev"]) {
    ctx.fillStyle = THEME[name];
    for (const v of groups[name]) {
      const c = Math.cos(v[3]), s = Math.sin(v[3]);
      ctx.setTransform(K * c, K * s, -K * s, K * c, (v[1] - view.x) * K, (v[2] - view.y) * K);
      ctx.fillRect(-L, v[6] * world.laneW - W / 2, L, W);
    }
  }
  base();

  // ---- signals: one lamp per lane, just inside the junction, drawn over the traffic.
  // Red lamps are deliberately dim so that green stands out and red queues stay readable.
  const lamps = { red: new Path2D(), yellow: new Path2D(), green: new Path2D() };
  const lampR = Math.max(1.15, 2.3 / k);
  for (const sl of world.stopLines) {
    const sig = side.sig[sl.node];
    const lit = sig && sig[0] === sl.phase;
    const col = lit && sig[1] === 0 ? "green" : lit && sig[1] === 1 ? "yellow" : "red";
    const mx = (sl.x0 + sl.x1) / 2, my = (sl.y0 + sl.y1) / 2;
    // the stop line runs across the lane; step 2.2 m into the box, perpendicular to it
    const nx = -(sl.y1 - sl.y0), ny = sl.x1 - sl.x0, nl = Math.hypot(nx, ny) || 1;
    const node = world.nodeById.get(sl.node);
    const dir = Math.sign((node.x - mx) * nx + (node.y - my) * ny) || 1;
    const cx = mx + (dir * nx / nl) * 2.2, cy = my + (dir * ny / nl) * 2.2;
    lamps[col].moveTo(cx + lampR, cy);
    lamps[col].arc(cx, cy, lampR, 0, Math.PI * 2);
  }
  for (const col of ["red", "yellow", "green"]) { ctx.fillStyle = THEME[col]; ctx.fill(lamps[col]); }

  // ---- emergency vehicles: a flashing ring that stays visible at any zoom
  if (groups.ev.length) {
    const flash = Math.floor(opts.now / 260) % 2 === 0;
    ctx.lineWidth = Math.max(0.5, 2 / k);
    ctx.strokeStyle = flash ? "#4da3ff" : "#ff5a5a";
    const r = Math.max(4.5, 8 / k);
    for (const v of groups.ev) {
      ctx.beginPath();
      ctx.arc(v[1] - Math.cos(v[3]) * world.vehLen / 2, v[2] - Math.sin(v[3]) * world.vehLen / 2, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // ---- accidents: a warning triangle
  for (const inc of side.incidents) {
    const r = Math.max(3.2, 7 / k), x = inc[1], y = inc[2];
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r * 0.95, y + r * 0.75); ctx.lineTo(x - r * 0.95, y + r * 0.75);
    ctx.closePath();
    ctx.fillStyle = THEME.incident;
    ctx.fill();
    ctx.strokeStyle = "#1b1b1b";
    ctx.lineWidth = Math.max(0.2, 1 / k);
    ctx.stroke();
    ctx.fillStyle = "#1b1b1b";
    ctx.fillRect(x - r * 0.09, y - r * 0.42, r * 0.18, r * 0.62);
    ctx.fillRect(x - r * 0.09, y + r * 0.34, r * 0.18, r * 0.18);
  }

  // ---- closed roads: a no-entry sign at the middle of the road
  for (const link of side.closed) {
    const mid = world.linkMid.get(link);
    if (!mid) continue;
    const r = Math.max(3.4, 7 / k);
    ctx.beginPath(); ctx.arc(mid[0], mid[1], r, 0, Math.PI * 2);
    ctx.fillStyle = "#d03b3b"; ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.fillRect(mid[0] - r * 0.62, mid[1] - r * 0.17, r * 1.24, r * 0.34);
  }

  // ---- labels in screen space: city edges always, entry names when zoomed in
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const [bx0, by0, bx1, by1] = world.s.bounds, mx = (bx0 + bx1) / 2, my = (by0 + by1) / 2;
  ctx.fillStyle = THEME.label;
  ctx.font = "600 12px system-ui, sans-serif";
  const edge = [["N", mx, by0 + 30], ["S", mx, by1 - 30], ["W", bx0 + 30, my], ["E", bx1 - 30, my]];
  for (const [t, x, y] of edge) ctx.fillText(t, (x - view.x) * k, (y - view.y) * k);
  if (k > 0.9) {
    ctx.font = "11px system-ui, sans-serif";
    for (const e of world.s.entries) {
      const dx = [0, 1, 0, -1][e.heading], dy = [-1, 0, 1, 0][e.heading];
      ctx.fillText(e.label, (e.x - dx * 9 - view.x) * k, (e.y - dy * 9 - view.y) * k);
    }
  }
}
