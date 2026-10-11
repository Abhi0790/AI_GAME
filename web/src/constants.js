// The map, including draw coordinates, comes from the server per game.
export const VIEWBOX = 500;

export const COLOR = {
  // Checked with the dataviz validator on the dark panel: every pair in seat
  // order passes; any two of seven cannot, so the board also letters each seat.
  Red: "#d4605a", Blue: "#5b8fd6", Green: "#389c76", Gold: "#b98c1a",
  Purple: "#c04fd0", Orange: "#c9783a", Teal: "#21a3b8", Pink: "#d478a8",
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
  if (type === "Move") return `${u}→${o.target}${o.coast ? "/" + o.coast : ""}`;
  if (type === "Convoy") return `${u} C ${o.supported_from}→${o.target}`;
  return `${u} S ${o.supported_from ? o.supported_from + "→" : ""}${o.target}`;
};
