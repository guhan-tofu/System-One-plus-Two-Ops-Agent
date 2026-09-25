# Sentinel API image. Secrets come from the runtime environment (never baked in):
#   docker build -t sentinel .
#   docker run -p 8000:8000 --env-file .env -v sentinel-data:/data sentinel
# Behind a TLS-intercepting proxy, pass its CA bundle as an optional build secret
# (used only while downloading packages, not kept in the image):
#   docker build --secret id=extra_ca,src=/path/to/ca-bundle.crt -t sentinel .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md* PLAN.md ./
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 sentinel && mkdir /data && chown sentinel /data
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY --from=build /app/src ./src
COPY policies ./policies
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/sentinel.db
USER sentinel
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["sentinel", "serve", "--host", "0.0.0.0", "--port", "8000"]
