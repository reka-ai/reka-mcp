FROM python:3.12-slim

# eMudhra CA compromise (CVE-2026-111943): drop the compromised trust anchors
RUN apt-get update \
 && apt-get install -y --no-install-recommends --only-upgrade ca-certificates \
 && sed -i -E 's#^(mozilla/(Comodo_AAA_Services_root|emSign_ECC_Root_CA_-_G3)\.crt)$#!\1#' /etc/ca-certificates.conf \
 && update-ca-certificates --fresh \
 && ! ls /etc/ssl/certs | grep -qE 'Comodo_AAA_Services_root|emSign_ECC_Root_CA_-_G3' \
 && rm -rf /var/lib/apt/lists/*
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv pip install --system --no-cache .

RUN uv pip install --system --no-cache "certifi>=2026.10.19"
RUN python -c "import certifi; v=certifi.__version__; assert tuple(map(int, v.split('.'))) >= (2026, 10, 19), v"

RUN adduser --system --no-create-home reka

ENV REKA_MCP_MODE=hosted
ENV PORT=8080

EXPOSE 8080

USER reka
CMD ["reka-mcp"]
