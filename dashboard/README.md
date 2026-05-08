# Dashboard

The dashboard now runs with a real Node backend instead of Vite-only middleware.

## Scripts

- `npm run dev`
  Starts the backend on `http://127.0.0.1:5173` and the Vite frontend on the default Vite port.
- `npm run dev:api`
  Starts only the backend server.
- `npm run dev:web`
  Starts only the Vite frontend.
- `npm run build`
  Builds the frontend assets into `dist/`.
- `npm run preview`
  Serves the built frontend plus all dashboard APIs from the backend server.

## Backend responsibilities

The backend serves:

- dashboard overview and per-round history APIs
- training session, replay, league, and log APIs
- the bundled Cambridge Battlecode visualiser under `/viewer/`
- built frontend assets during preview

## Frontend responsibilities

The frontend now:

- fetches lean overview and per-round payloads instead of bundling `ablations/history.json`
- code-splits major routes
- only mounts card contents when cards are open
- uses visibility-aware polling for live training views
