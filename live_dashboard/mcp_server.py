"""Spartus Live Dashboard — MCP Server

Exposes live trading data as MCP tools so any AI assistant (Claude or other)
can remotely monitor one or more live trading instances across the network.

TRANSPORT: HTTP + SSE (Server-Sent Events) via FastAPI.
           Runs over the internet — each live dashboard instance has its own
           port/URL.  The AI connects to each URL independently, giving full
           multi-account visibility.

AUTHENTICATION: Bearer token (set SPARTUS_MCP_TOKEN env var or in config).

USAGE (on the trading PC):
    pip install fastapi uvicorn python-multipart
    python mcp_server.py                   # default port 7474
    python mcp_server.py --port 8080       # custom port

    Set a strong token: set SPARTUS_MCP_TOKEN=your-secret-token-here

CLAUDE DESKTOP / CLAUDE CODE INTEGRATION:
    Add to claude_desktop_config.json:
    {
        "mcpServers": {
            "spartus-live-pc1": {
                "url": "http://<PC-IP>:7474/sse",
                "headers": { "Authorization": "Bearer your-secret-token-here" }
            },
            "spartus-live-pi1": {
                "url": "http://<PI-IP>:7474/sse",
                "headers": { "Authorization": "Bearer your-secret-token-here" }
            }
        }
    }

TOOLS EXPOSED:
    get_account_status       — balance, equity, open P/L, drawdown
    get_open_positions       — all open MT5 positions right now
    get_today_summary        — today's trades, win rate, P/L, session breakdown
    get_recent_trades        — last N closed trades (default 20)
    get_performance_stats    — win rate, profit factor, avg R, streak
    get_weekly_summary       — performance for a specific date range
    get_session_breakdown    — performance split by session (London/NY/Asia)
    get_day_of_week_stats    — performance by day of week
    get_alerts               — recent system alerts and circuit breaker state
    get_system_health        — model loaded, MT5 connected, feature pipeline ok
    get_instance_info        — this instance's name, model, config summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── FastAPI + SSE transport ──────────────────────────────────────────────────
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import StreamingResponse
except ImportError:
    print("ERROR: FastAPI/uvicorn not installed.")
    print("Run:  pip install fastapi uvicorn")
    sys.exit(1)

# ── MCP SDK ─────────────────────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

log = logging.getLogger("spartus.mcp")

# ── Config ───────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_DB_PATH = _HERE / "storage" / "memory" / "spartus_live.db"  # live dashboard SQLite DB
_ALERTS_PATH = _HERE / "storage" / "alerts.jsonl"
_STATE_PATH = _HERE / "storage" / "live_state.json"
_MODEL_DIR = _HERE / "model"
_CONFIG_PATH = _HERE / "config" / "user_settings.json"

_DEFAULT_PORT = 7474
_TOKEN_ENV = "SPARTUS_MCP_TOKEN"


# ─────────────────────────────────────────────────────────────────────────────
# Data access helpers — read-only queries against the live dashboard's SQLite DB
# ─────────────────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection | None:
    """Open a read-only connection to the live trading DB."""
    if not _DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return list of dicts.  Returns [] if DB unavailable."""
    conn = _get_db()
    if conn is None:
        return []
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning("DB query failed: %s", e)
        return []
    finally:
        conn.close()


def _read_live_state() -> dict:
    """Read the live_state.json snapshot written by the dashboard."""
    if _STATE_PATH.exists():
        try:
            with open(_STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _read_user_settings() -> dict:
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_model_name() -> str:
    zips = list(_MODEL_DIR.glob("*.zip")) if _MODEL_DIR.exists() else []
    if zips:
        return sorted(zips)[-1].stem
    return "no model loaded"


def _read_recent_alerts(n: int = 20) -> list[dict]:
    if not _ALERTS_PATH.exists():
        return []
    lines = []
    try:
        with open(_ALERTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return lines[-n:]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def tool_get_account_status() -> dict:
    state = _read_live_state()
    balance = state.get("balance", 0)
    equity = state.get("equity", balance)
    peak = state.get("peak_balance", balance)
    drawdown_pct = round((peak - equity) / peak * 100, 2) if peak > 0 else 0
    return {
        "balance": round(float(balance), 2),
        "equity": round(float(equity), 2),
        "profit": round(float(equity) - round(float(balance), 2), 2),
        "margin_free": round(float(state.get("free_margin", 0)), 2),
        "peak_balance": round(float(peak), 2),
        "drawdown_pct": drawdown_pct,
        "open_positions": int(state.get("open_positions", 0)),
        "open_pnl": round(float(state.get("open_pnl", 0)), 2),
        "currency": state.get("currency", "GBP"),
        "server": state.get("server", ""),
        "as_of": state.get("last_update", datetime.now(timezone.utc).isoformat()),
    }


def tool_get_open_positions() -> list[dict]:
    state = _read_live_state()
    return state.get("open_positions_detail", [])


def tool_get_today_summary() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    rows = _query("""
        SELECT COUNT(*) as total_trades,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
               SUM(pnl) as total_pnl,
               AVG(pnl) as avg_pnl,
               MAX(pnl) as best_trade,
               MIN(pnl) as worst_trade,
               AVG(hold_bars) as avg_hold_bars
        FROM trades WHERE DATE(timestamp) = ?
    """, (today,))
    s = rows[0] if rows else {}
    total = int(s.get("total_trades") or 0)
    wins = int(s.get("wins") or 0)
    losses = int(s.get("losses") or 0)
    return {
        "date": today,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "pnl": round(float(s.get("total_pnl") or 0), 2),
        "best_trade": round(float(s.get("best_trade") or 0), 2),
        "worst_trade": round(float(s.get("worst_trade") or 0), 2),
    }


def tool_get_recent_trades(limit: int = 20) -> list[dict]:
    limit = max(1, min(limit, 200))
    return _query("""
        SELECT mt5_ticket as ticket, side as direction, lot_size as lots,
               entry_price, exit_price,
               pnl, pnl_pct, hold_bars, close_reason,
               conviction, session_at_entry as session, timestamp
        FROM trades ORDER BY timestamp DESC LIMIT ?
    """, (limit,))


def tool_get_performance_stats() -> dict:
    rows = _query("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
               SUM(CASE WHEN pnl <= 0 THEN ABS(pnl) ELSE 0 END) as gross_loss,
               AVG(hold_bars) as avg_hold_bars
        FROM trades
    """)
    s = rows[0] if rows else {}
    total = int(s.get("total") or 0)
    wins = int(s.get("wins") or 0)
    gross_profit = float(s.get("gross_profit") or 0)
    gross_loss = float(s.get("gross_loss") or 0)

    streak_rows = _query("SELECT pnl FROM trades ORDER BY timestamp DESC LIMIT 30")
    streak = 0
    if streak_rows:
        first_sign = 1 if streak_rows[0].get("pnl", 0) > 0 else -1
        for r in streak_rows:
            sign = 1 if r.get("pnl", 0) > 0 else -1
            if sign == first_sign:
                streak += 1
            else:
                break
        streak *= first_sign

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        "avg_r": 0.0,
        "streak": streak,
        "streak_type": "win" if streak > 0 else ("loss" if streak < 0 else "none"),
    }


def tool_get_weekly_summary(start_date: str = "", end_date: str = "") -> dict:
    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-01")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    rows = _query("""
        SELECT DATE(timestamp) as day,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as total_pnl
        FROM trades
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp) ORDER BY day
    """, (start_date, end_date))
    return {"start": start_date, "end": end_date, "days": rows}


def tool_get_session_breakdown() -> dict:
    rows = _query("""
        SELECT session_at_entry as session,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(AVG(pnl), 2) as avg_pnl,
               ROUND(SUM(pnl), 2) as total_pnl
        FROM trades GROUP BY session_at_entry ORDER BY total_pnl DESC
    """)
    return {"sessions": rows}


def tool_get_day_of_week_stats() -> dict:
    rows = _query("""
        SELECT CAST(strftime('%w', timestamp) AS INTEGER) as dow,
               CASE strftime('%w', timestamp)
                   WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday'
                   WHEN '2' THEN 'Tuesday' WHEN '3' THEN 'Wednesday'
                   WHEN '4' THEN 'Thursday' WHEN '5' THEN 'Friday'
                   ELSE 'Saturday' END as day_name,
               COUNT(*) as trades,
               ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate_pct,
               ROUND(SUM(pnl), 2) as total_pnl
        FROM trades GROUP BY dow ORDER BY dow
    """)
    return {"by_day": rows}


def tool_get_alerts(limit: int = 30) -> dict:
    alerts = _read_recent_alerts(limit)
    state = _read_live_state()
    return {
        "recent_alerts": alerts,
        "circuit_breaker_tripped": state.get("circuit_breaker_tripped", False),
        "emergency_stop_active": state.get("emergency_stop_active", False),
        "daily_loss_limit_hit": state.get("daily_loss_limit_hit", False),
    }


def tool_get_system_health() -> dict:
    state = _read_live_state()
    db_ok = _DB_PATH.exists()
    model_name = _get_model_name()

    db_recent = False
    if db_ok:
        rows = _query("SELECT MAX(timestamp) as last FROM trades")
        if rows and rows[0].get("last"):
            db_recent = True

    return {
        "mt5_connected": state.get("mt5_connected", False),
        "model_loaded": state.get("model_loaded", False),
        "pipeline_ok": state.get("feature_pipeline_ok", False),
        "trading_allowed": state.get("trading_enabled", False),
        "model_name": model_name,
        "db_exists": db_ok,
        "db_has_trade_data": db_recent,
        "uptime_seconds": state.get("uptime_seconds", 0),
        "last_update": state.get("last_update", "unknown"),
    }


def tool_get_instance_info() -> dict:
    settings = _read_user_settings()
    state = _read_live_state()
    model_name = _get_model_name()
    paper = state.get("paper_trading", True)
    return {
        "instance_name": settings.get("instance_name", "Spartus Live"),
        "broker": state.get("server", settings.get("broker", "unknown")),
        "account_currency": state.get("currency", "GBP"),
        "model_version": model_name,
        "mode": "paper" if paper else "live",
        "symbol": settings.get("symbol", "XAUUSD"),
        "timeframe": "M5",
        "server_version": "1.0.0",
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCP tool registry
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS = {
    "get_account_status": {
        "fn": lambda _args: tool_get_account_status(),
        "description": "Get current account balance, equity, drawdown, and open position count.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_open_positions": {
        "fn": lambda _args: tool_get_open_positions(),
        "description": "List all currently open MT5 positions with direction, lots, P/L, and conviction.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_today_summary": {
        "fn": lambda _args: tool_get_today_summary(),
        "description": "Today's trading summary: trades, win rate, P/L, best/worst trade.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_recent_trades": {
        "fn": lambda args: tool_get_recent_trades(int(args.get("limit", 20))),
        "description": "Get the last N closed trades. Default 20, max 200.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of trades to return (1-200)", "default": 20}
            },
            "required": [],
        },
    },
    "get_performance_stats": {
        "fn": lambda _args: tool_get_performance_stats(),
        "description": "Overall performance stats: win rate, profit factor, avg R, SL hit rate, current streak.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_weekly_summary": {
        "fn": lambda args: tool_get_weekly_summary(
            args.get("start_date", ""), args.get("end_date", "")
        ),
        "description": "Day-by-day P/L and win rate between two dates (YYYY-MM-DD). Defaults to current month.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
            },
            "required": [],
        },
    },
    "get_session_breakdown": {
        "fn": lambda _args: tool_get_session_breakdown(),
        "description": "Performance split by trading session: London, New York, Asian overlap.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_day_of_week_stats": {
        "fn": lambda _args: tool_get_day_of_week_stats(),
        "description": "Win rate and P/L breakdown by day of week.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_alerts": {
        "fn": lambda args: tool_get_alerts(int(args.get("limit", 30))),
        "description": "Recent system alerts and circuit breaker / emergency stop state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of alerts to return", "default": 30}
            },
            "required": [],
        },
    },
    "get_system_health": {
        "fn": lambda _args: tool_get_system_health(),
        "description": "System health: MT5 connection, model loaded, feature pipeline status, uptime.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_instance_info": {
        "fn": lambda _args: tool_get_instance_info(),
        "description": "Static info about this trading instance: name, model, broker, config summary.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app (HTTP + SSE transport, works without MCP SDK)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Spartus Live MCP Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_security = HTTPBearer(auto_error=False)


def _check_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_security)) -> None:
    token = os.environ.get(_TOKEN_ENV, "")
    if not token:
        return  # No token configured = open access (local use)
    if credentials is None or credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/health")
def health():
    """Simple health check — no auth required."""
    return {"status": "ok", "server": "Spartus Live MCP", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/tools")
def list_tools(_auth=Depends(_check_auth)):
    """List all available tools."""
    return {
        "tools": [
            {"name": name, "description": meta["description"], "inputSchema": meta["inputSchema"]}
            for name, meta in _TOOLS.items()
        ]
    }


@app.post("/call/{tool_name}")
def call_tool(tool_name: str, body: dict = None, _auth=Depends(_check_auth)):
    """Call a tool by name with optional JSON body arguments."""
    if tool_name not in _TOOLS:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
    args = body or {}
    try:
        result = _TOOLS[tool_name]["fn"](args)
        return {"tool": tool_name, "result": result}
    except Exception as e:
        log.exception("Tool %s failed", tool_name)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshot")
def snapshot(_auth=Depends(_check_auth)):
    """Return all data sources in one call — useful for a full monitoring refresh."""
    alerts_raw = tool_get_alerts(10)
    # Flatten to a plain array so the FHAE dashboard can iterate with .map()
    alerts_list = alerts_raw.get("recent_alerts", [])
    return {
        "account": tool_get_account_status(),
        "open_positions": tool_get_open_positions(),
        "today": tool_get_today_summary(),
        "performance": tool_get_performance_stats(),
        "health": tool_get_system_health(),
        "instance": tool_get_instance_info(),
        "alerts": alerts_list,
    }


@app.get("/sse")
async def sse_endpoint(request: Request, _auth=Depends(_check_auth)):
    """SSE endpoint for MCP SDK transport (Claude Desktop / Claude Code integration).

    If mcp package is installed, delegates to SseServerTransport.
    Otherwise returns a friendly error.
    """
    if not _MCP_AVAILABLE:
        return {"error": "mcp package not installed. Use /call/<tool> REST endpoint instead.",
                "install": "pip install mcp"}

    # Build MCP server on demand
    mcp_server = Server("spartus-live")

    @mcp_server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(name=name, description=meta["description"],
                           inputSchema=meta["inputSchema"])
            for name, meta in _TOOLS.items()
        ]

    @mcp_server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        if name not in _TOOLS:
            return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]
        result = _TOOLS[name]["fn"](arguments or {})
        return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    transport = SseServerTransport("/messages")
    async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Spartus Live MCP Server")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT,
                        help=f"Port to listen on (default: {_DEFAULT_PORT})")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 = all interfaces)")
    args = parser.parse_args()

    token = os.environ.get(_TOKEN_ENV, "")
    if not token:
        log.warning("No auth token set. Set %s env var for security.", _TOKEN_ENV)
        log.warning("Example:  set %s=your-strong-secret-token", _TOKEN_ENV)
    else:
        log.info("Auth token configured (length=%d)", len(token))

    log.info("Spartus Live MCP Server starting on %s:%d", args.host, args.port)
    log.info("REST endpoint: http://<IP>:%d/call/<tool_name>", args.port)
    log.info("Snapshot endpoint: http://<IP>:%d/snapshot", args.port)
    if _MCP_AVAILABLE:
        log.info("SSE/MCP endpoint: http://<IP>:%d/sse", args.port)
    else:
        log.info("SSE/MCP: install 'mcp' package to enable Claude Desktop integration")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
