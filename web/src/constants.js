// Board geometry lives here; territories, adjacency and centres come from
// /api/board so the page cannot silently disagree with the engine.
export const POS = {
  R1: { x: 130, y: 130 }, R2: { x: 250, y: 72 },
  B1: { x: 370, y: 130 }, B2: { x: 428, y: 250 },
  G1: { x: 370, y: 370 }, G2: { x: 250, y: 428 },
  Y1: { x: 130, y: 370 }, Y2: { x: 72, y: 250 },
  N1: { x: 250, y: 178 }, N2: { x: 250, y: 322 },
  C1: { x: 178, y: 250 }, C2: { x: 322, y: 250 },
};

export const COLOR = {
  Red: "#d4605a", Blue: "#5b8fd6", Green: "#5aa878", Gold: "#c9a24a",
};

export const fmt = (n, places = 2) =>
  n === null || n === undefined || Number.isNaN(n) ? "—" : Number(n).toFixed(places);

export const pct = (n) =>
  n === null || n === undefined ? "—" : `${Math.round(n * 100)}%`;
