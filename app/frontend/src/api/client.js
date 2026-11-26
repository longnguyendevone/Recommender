const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function requestJson(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export function fetchHealth() {
  return requestJson("/health");
}

export function searchPois(q = "", category = "", limit = 15) {
  const params = new URLSearchParams({
    q,
    category,
    limit: String(limit),
  });
  return requestJson(`/pois/search?${params.toString()}`);
}

export function fetchCategories() {
  return requestJson("/pois/categories");
}

export function fetchSampleUsers(limit = 20) {
  return requestJson(`/users/sample?limit=${limit}`);
}

export function postRecommend(payload) {
  return requestJson("/recommend", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
