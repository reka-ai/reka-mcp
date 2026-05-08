FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv pip install --system --no-cache .

ENV REKA_MCP_MODE=hosted
ENV PORT=80

EXPOSE 80

CMD ["reka-mcp"]
