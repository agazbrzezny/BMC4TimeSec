FROM ubuntu:25.10

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip \
    ca-certificates tzdata \
    time z3 \
  && rm -rf /var/lib/apt/lists/*

# Copy app
COPY . /app

# Create venv + install python deps into venv (avoids PEP 668 issue)
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Ensure writable dirs + executables
RUN mkdir -p /app/var/jobs \
 && chmod -R 777 /app/var \
 && chmod -R 777 /benchmarks \
 && chmod +x /app/tools/*.py || true \
 && chmod +x /app/tools/smtreach4tis || true \
 && chmod +x /app/tools/smtreach4tiis || true
# Use venv by default
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
