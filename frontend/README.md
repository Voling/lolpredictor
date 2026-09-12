# frontend

Next.js 15 dashboard for the compatibility model. Server components fetch the FastAPI backend
directly, so there is no client-side data layer to keep in sync.

## Running

| Command | What it does |
|---|---|
| `npm install` | Install dependencies |
| `npm run dev` | Dev server on :3000, proxying `/api/*` to `API_URL` |
| `npm run build` | Production build |
| `npm run start` | Serve the production build on :3000 |

`API_URL` defaults to `http://localhost:8000`. Under `docker compose` it is set to `http://api:8000`.

## Layout

| Path | Holds |
|---|---|
| `app/page.tsx` | Status page: corpus size, model metrics, the reliability gate |
| `app/layout.tsx` | Root layout and metadata |
| `app/globals.css` | Every style, no framework |
| `lib/api.ts` | Typed fetch helpers against the backend |
| `next.config.ts` | Rewrites `/api/*` to the backend |

## The gate

The backend returns `score: null` and `reliable: false` whenever `synergy_gain_sigma` is under 2.0,
which is currently always. The UI shows that as a notice rather than hiding it or substituting a
number, because a 0-100 score the model does not stand behind is worse than no score. Render
`reliable` before rendering `score`.
