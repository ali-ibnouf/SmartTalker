# SmartTalker Deployment Guide

## Quick Start (Development)

```bash
# 1. Clone and setup
git clone <repo-url> && cd SmartTalker
cp .env.example .env

# 2. Setup environment
make setup        # Linux: venv + dependencies
# or Windows:
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# 3. Download models
bash scripts/download_models.sh

# 4. Start services
make run          # docker-compose up -d
```

## Docker Deployment

### Prerequisites
- Docker 24+
- Docker Compose v2
- NVIDIA Container Toolkit
- NVIDIA GPU with 24GB+ VRAM

### Start Services

```bash
# Build and run all services (dev)
docker compose up -d --build

# Check status
docker compose ps
docker compose logs -f core
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| `core` | 8000 | FastAPI application |
| `ollama` | 11434 | Qwen 2.5 LLM |
| `redis` | 6379 | Caching and queues |
| `nginx` | 80 (dev) / 80+443 (prod) | Reverse proxy & SSL termination |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards & visualization |
| `redis-exporter` | 9121 | Redis metrics (prod only) |
| `nvidia-gpu-exporter` | 9445 | GPU metrics (prod only) |

### GPU Configuration

```yaml
# docker-compose.yml — GPU allocation
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Production Deployment

### 1. Environment Setup

```bash
# Copy and configure environment
cp .env.example .env

# Edit production values
APP_ENV=production
DEBUG=false
CORS_ORIGINS=https://yourdomain.com
API_KEY=<generate-a-strong-key>
DOMAIN_NAME=yourdomain.com
```

### 2. SSL Certificates (Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot

# Obtain certificate (stop nginx first if running on port 80)
sudo certbot certonly --standalone -d yourdomain.com

# Certificates are stored at:
#   /etc/letsencrypt/live/yourdomain.com/fullchain.pem
#   /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

After obtaining certificates, uncomment the HTTPS server block in `nginx.conf` and update `DOMAIN_NAME` in your `.env` file.

### 3. Launch Production Stack

```bash
# Validate the compose config first
docker compose -f docker-compose.yml -f docker-compose.prod.yml config

# Start production services
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Verify all services are healthy
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

## Environment Variables

See `.env.example` for all variables. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | API port |
| `API_KEY` | *(empty)* | API authentication key (required in production) |
| `LLM_MODEL_NAME` | `qwen2.5:14b` | Ollama model |
| `VIDEO_ENABLED` | `false` | Enable video generation |
| `UPSCALE_ENABLED` | `false` | Enable upscaling |
| `GPU_MEMORY_FRACTION` | `0.9` | Max GPU memory usage |
| `DOMAIN_NAME` | `localhost` | Domain for nginx / SSL |

## Monitoring

### Prometheus Targets

Prometheus scrapes the following endpoints:

| Job | Target | Interval |
|-----|--------|----------|
| `smarttalker-core` | `core:8000/metrics` | 5s |
| `redis` | `redis-exporter:9121/metrics` | 15s |
| `gpu` | `nvidia-gpu-exporter:9445/metrics` | 15s |

Prometheus config is at `prometheus.yml` with alert rules in `prometheus/alerts.yml`.

### Grafana Dashboard

Grafana is auto-provisioned with the SmartTalker dashboard on startup.

1. Open `http://localhost:3000` (default credentials: `admin` / value of `GRAFANA_ADMIN_PASSWORD` or `admin`)
2. Navigate to **Dashboards → SmartTalker Overview**
3. The dashboard includes 6 panels:
   - **Request Rate** — requests per second by method/handler
   - **Latency Percentiles** — p50/p95/p99 response times
   - **Error Rate** — 5xx error percentage (green <1%, yellow <5%, red >=5%)
   - **GPU Memory Usage** — current VRAM utilization gauge
   - **Active WebSocket Sessions** — live connection count
   - **Pipeline Latency Breakdown** — ASR/LLM/TTS/Video avg latency

### Alert Configuration

Five alerts are defined in `prometheus/alerts.yml`:

| Alert | Fires When | Severity |
|-------|-----------|----------|
| ServiceDown | Any target unreachable for 2m | critical |
| HighErrorRate | >5% 5xx responses over 5m | warning |
| GPUMemoryHigh | GPU memory >90% for 5m | warning |
| DiskSpaceLow | Disk free <10% for 5m | critical |
| RedisDown | Redis unreachable for 2m | critical |

To receive alert notifications, configure an Alertmanager instance and update the `alerting.alertmanagers` section in `prometheus.yml`.

## Security Hardening

### Firewall (UFW)

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Do NOT expose internal ports (9090, 3000, 6379, 9121, 9445)
```

### Restrict Metrics Access

In `nginx.conf`, uncomment the IP restriction block in the `/metrics` location to limit access to internal networks:

```nginx
location /metrics {
    proxy_pass http://smarttalker;
    allow 10.0.0.0/8;
    allow 172.16.0.0/12;
    deny all;
}
```

### Rate Limiting

Rate limiting is configured at two levels:
- **Nginx**: `limit_req_zone` in `nginx.conf` (30 req/s per IP with burst of 20)
- **Application**: `RATE_LIMIT_PER_MINUTE` in `.env` (default 60/min per IP)

## Backup & Maintenance

### Redis Backup

```bash
# Trigger a background save
docker exec smarttalker-redis redis-cli BGSAVE

# Backup the dump file
docker cp smarttalker-redis:/data/dump.rdb ./backups/redis-$(date +%F).rdb
```

### Volume Backup

```bash
# Back up all persistent volumes
mkdir -p ./backups

# Ollama models
docker run --rm -v smarttalker_ollama-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/ollama-data-$(date +%F).tar.gz -C /data .

# Grafana data
docker run --rm -v smarttalker_grafana-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/grafana-data-$(date +%F).tar.gz -C /data .

# Prometheus data
docker run --rm -v smarttalker_prometheus-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/prometheus-data-$(date +%F).tar.gz -C /data .
```

### SSL Auto-Renewal

```bash
# Add certbot renewal cron job
sudo crontab -e
# Add the following line (runs twice daily):
0 */12 * * * certbot renew --quiet && docker exec smarttalker-nginx nginx -s reload
```

## Production Checklist

- [ ] Set `APP_ENV=production` and `DEBUG=false`
- [ ] Configure `CORS_ORIGINS` (no wildcard)
- [ ] Set a strong `API_KEY`
- [ ] Set `WHATSAPP_*` credentials for WhatsApp integration
- [ ] Obtain and configure SSL certificates
- [ ] Set `DOMAIN_NAME` to your actual domain
- [ ] Set `GRAFANA_ADMIN_PASSWORD` to a strong password
- [ ] Configure firewall (only expose ports 80/443)
- [ ] Set `STORAGE_MAX_FILE_AGE_HOURS` for cleanup
- [ ] Set up backup cron jobs
- [ ] Run `python scripts/benchmark.py` to verify latency

## Health Monitoring

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Expected healthy response
{
  "status": "healthy",
  "gpu_available": true,
  "models_loaded": { "asr": true, "tts": true, ... }
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `GPU_MEMORY_FRACTION`, disable video/upscale |
| `Cannot connect to Ollama` | Check `docker compose ps ollama`, verify port 11434 |
| `Model not found` | Run `bash scripts/download_models.sh` |
| `Permission denied` | Ensure Docker user has GPU access: `sudo usermod -aG docker $USER` |
| `TTS silence output` | Check voice reference audio is 3-10s WAV, 16kHz+ |
| `nginx 502 Bad Gateway` | Core service not ready yet — check `docker compose logs core` |
| `SSL certificate errors` | Verify cert paths, run `certbot renew`, reload nginx |
| `Grafana no data` | Check Prometheus targets at `http://localhost:9090/targets` |
| `Prometheus alerts not firing` | Verify `alerts.yml` is mounted and rules are loaded |
