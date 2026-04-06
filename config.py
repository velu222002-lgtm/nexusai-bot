# ══════════════════════════════════════════════════
#  NEXUSAI TRADING BOT — CONFIG FILE
#  Edit this file with your keys and settings
# ══════════════════════════════════════════════════

# ── Binance API Keys ───────────────────────────────
API_KEY    = "XqZ4C82qO5lUJvZS3VeiFmFai8Plp5ntsmU0DQAf2M5J9Ksm9VTAz0S64RdwXMK9"
API_SECRET = "NTgqGcG4U3qIxPEqhhVADTLFngZ8GqjsGOTSS3rGmk8cenCJWfEcZtKlo0TxtWsq"

# ── Mode ───────────────────────────────────────────
# "testnet" = fake money (safe practice)
# "live"    = real money (be careful!)
MODE = "testnet"

# ── Coins to Track ─────────────────────────────────
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LTCUSDT"
]

# ── Timeframe ──────────────────────────────────────
INTERVAL = "1h"

# ── Trade Settings ─────────────────────────────────
TRADE_USDT  = 20    # How much USDT per trade
STOP_LOSS   = 2.0   # Stop loss percentage
TAKE_PROFIT = 4.0   # Take profit percentage

# ── Telegram Alerts ────────────────────────────────
# Get token from @BotFather on Telegram
# Get chat_id from @userinfobot on Telegram
TELEGRAM_TOKEN   = "8679767093:AAGm61FjhzHJUBg2iwiIclgAR1UaD0qedYA"
TELEGRAM_CHAT_ID = "1720777283"

# ── Alert Settings ─────────────────────────────────
MIN_CONFIDENCE = 50  # Only alert if confidence >= this %