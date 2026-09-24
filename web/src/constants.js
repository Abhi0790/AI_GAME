// The map, including draw coordinates, comes from the server per game.
export const VIEWBOX = 500;

export const COLOR = {
  Red: "#d4605a", Blue: "#5b8fd6", Green: "#5aa878", Gold: "#c9a24a",
  Purple: "#9b7fd4", Orange: "#d68a4e", Teal: "#4fa8a8", Pink: "#d478a8",
};

export const fmt = (n, places = 2) =>
  n === null || n === undefined || Number.isNaN(n) ? "—" : Number(n).toFixed(places);

export const pct = (n) =>
  n === null || n === undefined ? "—" : `${Math.round(n * 100)}%`;

// One order, short. Traces name the fields one way and /inspect another, so
// it takes both rather than making every caller normalise first.
export const orderText = (o) => {
  const u = o.unit_territory ?? o.unit;
  const type = o.order_type ?? o.type;
  if (type === "Hold") return `${u} H`;
  if (type === "Move") return `${u}→${o.target}`;
  return `${u} S ${o.supported_from ? o.supported_from + "→" : ""}${o.target}`;
};
