FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv pip install --system --no-cache .

RUN adduser --system --no-create-home reka

ENV REKA_MCP_MODE=hosted
ENV PORT=8080

EXPOSE 8080

USER reka
CMD ["reka-mcp"]
