// Frames arrive as slowly as twice a second (at 1x speed), so each city keeps the last
// two frames and vehicles are drawn in between them, matched by id.
// Vehicle row: [id, x, y, angle, etype, v_norm, lat]
const TAU = Math.PI * 2;

function lerpAngle(a, b, t) {
  let d = (b - a) % TAU;
  if (d > Math.PI) d -= TAU;
  if (d < -Math.PI) d += TAU;
  return a + d * t;
}

export class SideBuffer {
  constructor() {
    this.prev = null;
    this.cur = null;
  }

  push(side, simT, wallT) {
    const map = new Map();
    for (const v of side.veh) map.set(v[0], v);
    const snap = { simT, wallT, map, side };
    if (!this.cur || simT < this.cur.simT) this.prev = null;          // first frame, or a restart
    else if (simT > this.cur.simT) this.prev = this.cur;              // normal advance
    this.cur = snap;                                                  // same time: refresh in place
  }

  get side() {
    return this.cur ? this.cur.side : null;
  }

  // Vehicles at wall-clock time `now`, drawn one frame interval behind the newest frame.
  sample(now) {
    const cur = this.cur;
    if (!cur) return [];
    const prev = this.prev;
    if (!prev) return cur.side.veh;
    const span = Math.min(1200, Math.max(16, cur.wallT - prev.wallT));
    const t = Math.min(1, Math.max(0, (now - cur.wallT) / span));
    if (t >= 1) return cur.side.veh;
    const out = [];
    for (const v of cur.side.veh) {
      const p = prev.map.get(v[0]);
      if (!p) { out.push(v); continue; }
      out.push([v[0], p[1] + (v[1] - p[1]) * t, p[2] + (v[2] - p[2]) * t, lerpAngle(p[3], v[3], t),
        v[4], p[5] + (v[5] - p[5]) * t, p[6] + (v[6] - p[6]) * t]);
    }
    return out;
  }
}
