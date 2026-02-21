# Smart Gate — Dashboard & Access Logs

Visual dashboard for access history with real-time statistics, photo thumbnails and entry filtering.

## Files

```
dashboard/
├── dashboard.html     # Standalone HTML dashboard served via /local/
└── dashboard.yaml     # Lovelace view definitions
```

## Setup

### 1. Copy dashboard to HA www

```bash
cp dashboard/dashboard.html /config/www/smart_gate/dashboard.html
```

The file will be served at:
```
http://<ha-ip>:8123/local/smart_gate/dashboard.html
```

### 2. Add Lovelace views

Go to **Settings → Dashboards → your dashboard → Edit → Raw YAML editor** and paste the contents of `dashboard/dashboard.yaml`.

The YAML adds two views to your dashboard:

| View | Path | Description |
|------|------|-------------|
| Smart Gate | `/lovelace/smart-gate` | Live camera + open gate button (used by tap notification) |
| Storico Accessi | `/lovelace/smart-gate-log` | Access log dashboard via iframe |

## Dashboard Features

- **Statistics bar** — total openings, rejections, unknowns and unique plates
- **Filter buttons** — filter by status: all / opened / rejected / unknown
- **Photo thumbnails** — click any photo to open fullscreen lightbox
- **Auto-refresh** — reloads every 30 seconds automatically

## Access Log

Every gate event is saved to `/config/www/smart_gate/access_log.json` (max 200 entries, newest first).

Each entry contains:

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 datetime |
| `plate` | Plate read by OCR |
| `status` | `opened` / `rejected` / `unknown` |
| `match_type` | `exact` / `fuzzy` / `none` |
| `matched_plate` | Whitelist plate matched (if any) |
| `yolo_score` | YOLO detection confidence |
| `ocr_conf` | OCR confidence |
| `snapshot_url` | `/local/` URL of the snapshot |
| `reason` | Human readable rejection reason |

### Logged events

| Event | Status | Match type |
|-------|--------|------------|
| Exact whitelist match | `opened` | `exact` |
| Fuzzy match + person home | `opened` | `fuzzy` |
| Fuzzy match + person not home | `rejected` | `fuzzy` |
| Fuzzy match + no person_entity | `rejected` | `fuzzy` |
| YOLO score too low | `rejected` | `none` |
| OCR confidence too low | `rejected` | `none` |
| Not in whitelist | `unknown` | `none` |
