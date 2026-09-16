# One Dockerfile, several images. Build a specific one with --target:
#
#   docker build -t territory-web .                      # the dashboard (default)
#   docker build -t territory-sim    --target sim .      # headless games, no Node
#   docker build -t territory-report --target report .   # figures and the sweep
#   docker build -t territory-test   --target test .     # the test suite
#
# They share the `deps` stage, so the Python layer is built once and cached
# across all of them. `web` is the only one that needs the Node build output.

# ── Stage 1: the React bundle ───────────────────────────────────────────
FROM node:22-alpine AS ui
WORKDIR /ui
# Lockfile first so `npm ci` is cached until the dependencies actually change.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build


# ── Stage 2: Python dependencies, shared by every runtime image ─────────
FROM python:3.13-slim AS deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt


# ── Stage 3: the source, on top of the dependencies ─────────────────────
FROM deps AS source
COPY src/ ./src/
COPY scripts/ ./scripts/
# Non-root by default. Nothing here needs to write outside the volumes the
# report image declares, so an unprivileged user costs nothing.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app


# ── sim: headless games. No Node, no web assets. ────────────────────────
FROM source AS sim
USER app
ENTRYPOINT ["python"]
CMD ["scripts/run_headless.py", "--seed", "42"]


# ── report: the six figures and the parameter sweep ─────────────────────
FROM source AS report
USER app
VOLUME ["/app/figures", "/app/data"]
ENTRYPOINT ["python"]
CMD ["scripts/make_figures.py", "--games", "8", "--out", "figures"]


# ── test: the suite, including the DATC cases and property tests ────────
FROM source AS test
COPY tests/ ./tests/
USER app
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["tests/", "-q"]


# ── web: the dashboard. The default target. ─────────────────────────────
FROM source AS web
# The bundle lands next to the app module, which is where src/ui/app.py looks
# for it first -- so the image carries no reference back to the web/ source.
COPY --from=ui /ui/dist ./src/ui/dist
RUN chown -R app:app /app/src/ui/dist
USER app
EXPOSE 8000
# The API answers this without touching game state, so it is a real liveness
# check rather than a request that merely proves the port is open.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/board', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "src.ui.app:app", "--host", "0.0.0.0", "--port", "8000"]
