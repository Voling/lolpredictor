# lolpredictor frontend

React dashboard for the compatibility API. `npm start` serves it on :3000 and proxies `/api` to the
Django server on :8000; `npm run build` produces the static bundle the nginx image serves. See the
repository README for the pipeline that fills the API.
