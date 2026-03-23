export function apiUrl(path) {
  return path
}

export function viewerUrl(replayPath) {
  return apiUrl(`/viewer/?replayUrl=${encodeURIComponent(replayPath)}`)
}

async function fetchJson(path, { signal } = {}) {
  const response = await fetch(apiUrl(path), { signal })
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return response.json()
}

export function fetchJsonApi(path, { signal } = {}) {
  return fetchJson(path, { signal })
}
