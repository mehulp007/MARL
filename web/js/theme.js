// Colour roles used on the canvases (mirrors the custom properties in style.css).
export const THEME = {
  ground: "#101011",
  road: "#2b2d30",
  box: "#303338",
  centre: "#5d5647",
  divider: "#44474c",
  label: "#898781",
  ink: "#f0efec",
  ink2: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  axis: "#383835",
  surface: "#1a1a19",
  // vehicles: a three-step state, not a rainbow
  moving: "#cfd6dc",
  slow: "#fab219",
  stopped: "#e66767",
  ev: "#7fc8ff",
  broken: "#ec835a",
  // signals
  green: "#3ddc5a",
  yellow: "#ffd23f",
  red: "#8c3030",          // dim on purpose: see render.js
  // perturbations
  incident: "#ec835a",
  closed: "#e66767",
  hover: "#ffffff",
  // series identity in charts and tables (validated pair on the dark surface)
  a: "#3987e5",
  b: "#d95926",
};

export const SPEED_SLOW = 0.45;     // v / v0 below this is "slow"
export const SPEED_STOP = 0.05;     // ... and below this "stopped"
