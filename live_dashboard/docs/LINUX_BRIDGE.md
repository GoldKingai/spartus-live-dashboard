# Linux Live Trading via the mt5linux Bridge

The Spartus Live Dashboard runs natively on Linux (and macOS), but
**MetaTrader 5 itself only ships a Windows binary**. To enable live
trading from a Linux host you have two options:

1. **Run the dashboard on Linux + MT5 in Wine on the same machine** (this guide)
2. **Run the dashboard on Linux + MT5 on a remote Windows host/VPS** (same setup, different `MT5_BRIDGE_HOST`)

Both work because the dashboard uses [`mt5linux`](https://github.com/lucas-campagna/mt5linux),
an RPC client that mirrors the `MetaTrader5` Python API over the network.

## Architecture

```
┌──────────────────────────────────┐         ┌─────────────────────────────┐
│  Linux host                       │  RPC    │  Wine prefix (or Windows)    │
│                                   │ ──────► │                              │
│  spartus-live-dashboard           │  18812  │  Windows Python 3.11         │
│  (native Linux Python)            │         │   ├── MetaTrader5 (Win pkg)  │
│  └── core/mt5_bridge.py           │         │   ├── mt5linux (server)      │
│       ├─ mt5linux client          │         │   └── MetaTrader 5 terminal  │
│       └─ all calls go via RPC     │         │                              │
└──────────────────────────────────┘         └─────────────────────────────┘
```

The dashboard's `MT5Bridge` auto-selects transport at import time:

| Available | Transport | Where it runs |
|---|---|---|
| Native `MetaTrader5` package | `native` | Windows host |
| `mt5linux` package + bridge daemon reachable | `mt5linux-bridge` | Linux host → Wine |
| Neither | `offline` | Linux host, UI only |

## Prerequisites

- Linux host (Ubuntu 22.04+, Debian 12+, Fedora 38+ tested)
- Wine 8+
- Internet access for MT5 installer + Windows Python installer
- A broker account ready to log into MT5 (Vantage, IC Markets, etc.)

## Step 1 — Install Wine

**Ubuntu/Debian:**
```bash
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/$(lsb_release -cs)/winehq-$(lsb_release -cs).sources
sudo apt update
sudo apt install --install-recommends winehq-stable
```

**Fedora:**
```bash
sudo dnf install wine
```

Verify: `wine --version` (should print 8.x or 9.x).

## Step 2 — Create a dedicated Wine prefix for MT5

```bash
export WINEPREFIX="$HOME/.wine_mt5"
export WINEARCH=win64
wineboot --init
```

A `~/.wine_mt5` directory will be created. Setting `WINEARCH=win64` is required — the MT5 installer is 64-bit only.

## Step 3 — Install MetaTrader 5 inside the Wine prefix

Download the MT5 installer from your broker (or generic from metaquotes.net):

```bash
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
WINEPREFIX="$HOME/.wine_mt5" wine mt5setup.exe
```

Run the installer through, log into your broker account, confirm that the MT5 terminal opens and prices update. Right-click Market Watch → "Show All" so XAUUSD is visible.

## Step 4 — Install Windows Python 3.11 inside the same Wine prefix

```bash
wget https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
WINEPREFIX="$HOME/.wine_mt5" wine python-3.11.9-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
```

Verify:
```bash
WINEPREFIX="$HOME/.wine_mt5" wine python --version
# Python 3.11.9
```

## Step 5 — Install MetaTrader5 + mt5linux inside Wine Python

```bash
WINEPREFIX="$HOME/.wine_mt5" wine python -m pip install MetaTrader5 mt5linux
```

## Step 6 — Run the bridge daemon

In one terminal (leave it running):

```bash
WINEPREFIX="$HOME/.wine_mt5" wine python -m mt5linux --host localhost --port 18812
```

The daemon will print:
```
Listening on localhost:18812
```

This is the RPC server that the Linux-side dashboard talks to.

## Step 7 — Install the Linux dashboard

```bash
git clone --branch v1.9.0 https://github.com/GoldKingai/spartus-live-dashboard.git
cd spartus-live-dashboard/live_dashboard
chmod +x install.sh launch.sh
./install.sh
```

Verify the install picked up `mt5linux`:
```bash
./venv/bin/python -c "import mt5linux; print(mt5linux.__version__)"
```

## Step 8 — Launch the dashboard

In a second terminal:

```bash
cd spartus-live-dashboard/live_dashboard
./launch.sh
```

You should see:
```
INFO: MetaTrader5 transport: mt5linux-bridge
INFO: MetaTrader5 via mt5linux bridge at localhost:18812
INFO: MT5Bridge: connecting via mt5linux bridge
INFO: MT5 terminal initialised successfully
INFO: Account currency: USD ($)
...
```

If the dashboard reports "Entering OFFLINE MODE" instead, the bridge daemon either isn't running or isn't reachable.

## Connecting to a remote Windows host (or VPS)

Run the bridge daemon on the Windows host:

```cmd
pip install mt5linux
python -m mt5linux --host 0.0.0.0 --port 18812
```

Then on the Linux host:
```bash
export MT5_BRIDGE_HOST=192.168.1.50   # Windows host IP
export MT5_BRIDGE_PORT=18812
./launch.sh
```

Open port 18812 in the Windows firewall.

## Optional — Run the bridge as a systemd service

`/etc/systemd/system/mt5-bridge.service`:

```ini
[Unit]
Description=MT5 mt5linux bridge daemon (in Wine)
After=network.target

[Service]
Type=simple
User=YOUR_USER
Environment=WINEPREFIX=/home/YOUR_USER/.wine_mt5
Environment=DISPLAY=:0
ExecStart=/usr/bin/wine python -m mt5linux --host localhost --port 18812
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable + start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mt5-bridge
sudo systemctl status mt5-bridge
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Dashboard logs `transport: offline` | mt5linux not installed in Linux venv | `./venv/bin/pip install mt5linux>=1.0.3` |
| `MT5 initialize failed` after bridge install | bridge daemon not running | Step 6 (start `python -m mt5linux`) |
| `connection refused` | wrong host/port or firewall | Check `MT5_BRIDGE_HOST` / `MT5_BRIDGE_PORT`, open port 18812 |
| MT5 terminal won't launch in Wine | missing 32-bit libs or wrong WINEARCH | `WINEARCH=win64`; `winetricks corefonts vcrun2019` |
| Account info shows USD when broker is GBP | bridge uses cached account; restart bridge | `systemctl restart mt5-bridge` (or kill + restart manual) |

## Limits / known gaps

- mt5linux RPC adds ~10-50ms latency per call. Fine for trade execution; may slow tick-by-tick streams.
- All MT5 RPC calls go over a single TCP socket — high-frequency strategies will benefit from running on Windows native.
- Wine + MT5 occasionally deadlocks if the terminal is closed unexpectedly. If the bridge stops responding, kill all wine processes and restart.
