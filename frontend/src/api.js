const BASE = process.env.REACT_APP_API_BASE || '';

async function request(path, options) {
  const response = await fetch(`${BASE}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `request failed with ${response.status}`);
  }
  return payload;
}

export function getStatus() {
  return request('/api/status/');
}

export function searchPlayers(term, limit = 12) {
  const query = new URLSearchParams({ q: term, limit });
  return request(`/api/players/?${query}`);
}

export function getPlayer(query) {
  return request(`/api/players/${encodeURIComponent(query)}/`);
}

export function getPartners(query, limit = 8) {
  return request(`/api/partners/${encodeURIComponent(query)}/?limit=${limit}`);
}

export function analyseTeam(players) {
  return request('/api/team/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ players }),
  });
}
