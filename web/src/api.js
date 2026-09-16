// Every call the dashboard makes. Kept in one file so the shape of the API
// is visible in one place rather than scattered through components.

const json = async (res) => {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
};

const post = (path, body) =>
  fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  }).then(json);

export const getBoard = () => fetch("/api/board").then(json);
export const newGame = (cfg) => post("/api/game/new", cfg);
export const getState = (id) => fetch(`/api/game/${id}/state`).then(json);
export const getPending = (id) => fetch(`/api/game/${id}/pending`).then(json);
export const stepGame = (id, turn) => post(`/api/game/${id}/step`, turn);
export const getReplay = (id) => fetch(`/api/game/${id}/replay`).then(json);
export const saveReplay = (id) => post(`/api/game/${id}/save`);
export const inspect = (id, seat) =>
  fetch(`/api/game/${id}/inspect/${seat}`).then(json);
