from flask import Flask, render_template_string, jsonify
from binance.client import Client
import joblib, ta, threading, time
import pandas as pd
from datetime import datetime
import config

try:
    from telegram_bot import send_startup, send_signal, send_trade
    TELEGRAM_ENABLED = bool(config.TELEGRAM_TOKEN)
except:
    TELEGRAM_ENABLED = False

app = Flask(__name__)

print("Connecting to Binance Testnet...")
client = Client(config.API_KEY, config.API_SECRET)
try:
    st = client.get_server_time()
    client.timestamp_offset = st['serverTime'] - int(time.time() * 1000)
    print(f"Clock synced!")
except:
    pass
print("Connected!")
model = joblib.load("models/trading_model.pkl")
print("AI Model loaded!")

FEATURES = [
    "rsi","macd","macd_sig","ema_9","ema_20","ema_50",
    "bb_high","bb_low","bb_mid","atr","adx","cci",
    "stoch","williams","momentum","volume"
]

trade_history = []
latest_data   = {"signals":[],"history":[],"usdt":0,"time":"--:--:--","stats":{}}
last_signals  = {}

# P&L Tracking
stats = {
    "total_trades": 0,
    "wins": 0,
    "losses": 0,
    "total_profit": 0.0,
    "total_loss": 0.0,
    "best_trade": 0.0,
    "worst_trade": 0.0,
    "starting_balance": 10000.0,
    "current_balance": 10000.0,
}

def get_signal(symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval="1h", limit=50)
        df = pd.DataFrame(klines, columns=["time","open","high","low","close","volume","ct","qv","tr","tb","tq","ig"])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["rsi"]      = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["macd"]     = ta.trend.MACD(df["close"]).macd()
        df["macd_sig"] = ta.trend.MACD(df["close"]).macd_signal()
        df["ema_9"]    = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["ema_20"]   = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["ema_50"]   = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["bb_high"]  = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
        df["bb_low"]   = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
        df["bb_mid"]   = ta.volatility.BollingerBands(df["close"]).bollinger_mavg()
        df["atr"]      = ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"]).average_true_range()
        df["adx"]      = ta.trend.ADXIndicator(df["high"],df["low"],df["close"]).adx()
        df["cci"]      = ta.trend.CCIIndicator(df["high"],df["low"],df["close"]).cci()
        df["stoch"]    = ta.momentum.StochasticOscillator(df["high"],df["low"],df["close"]).stoch()
        df["williams"] = ta.momentum.WilliamsRIndicator(df["high"],df["low"],df["close"]).williams_r()
        df["momentum"] = df["close"].pct_change(5)
        df.dropna(inplace=True)
        latest     = df[FEATURES].iloc[-1:]
        proba      = model.predict_proba(latest)[0]
        pred       = model.predict(latest)[0]
        confidence = round(max(proba)*100, 1)
        signal     = "BUY" if pred == 1 else "SELL"
        return {
            "symbol":     symbol,
            "price":      round(df["close"].iloc[-1], 2),
            "rsi":        round(df["rsi"].iloc[-1], 2),
            "ema20":      round(df["ema_20"].iloc[-1], 2),
            "ema50":      round(df["ema_50"].iloc[-1], 2),
            "signal":     signal,
            "confidence": confidence,
            "prices":     [round(x,2) for x in df["close"].tail(20).tolist()],
            "rsis":       [round(x,2) for x in df["rsi"].tail(20).tolist()],
            "vols":       [round(x,2) for x in df["volume"].tail(20).tolist()]
        }
    except Exception as e:
        print(f"Error {symbol}: {e}")
        return None

def fetch_loop():
    global latest_data, last_signals, stats
    startup_sent = False
    while True:
        try:
            print(f"\n{datetime.now().strftime('%H:%M:%S')} Fetching {len(config.SYMBOLS)} coins...")
            signals = []
            for s in config.SYMBOLS:
                r = get_signal(s)
                if r:
                    signals.append(r)
                    print(f"  {s}: {r['signal']} ${r['price']} RSI:{r['rsi']} Conf:{r['confidence']}%")
                    if TELEGRAM_ENABLED:
                        prev = last_signals.get(s)
                        if r["confidence"] >= config.MIN_CONFIDENCE:
                            if prev != r["signal"]:
                                send_signal(s, r["signal"], r["price"], r["rsi"], r["confidence"])
                                last_signals[s] = r["signal"]

            try:
                st2 = client.get_server_time()
                client.timestamp_offset = st2['serverTime'] - int(time.time() * 1000)
                usdt = client.get_asset_balance("USDT")
                bal  = round(float(usdt["free"]), 2) if usdt else 0
                stats["current_balance"] = bal
            except Exception as e:
                print(f"Balance error: {e}")
                bal = 0

            # Calculate net P&L
            net_pnl = round(stats["total_profit"] - abs(stats["total_loss"]), 2)

            latest_data = {
                "signals": signals,
                "history": trade_history,
                "usdt":    bal,
                "time":    datetime.now().strftime("%H:%M:%S"),
                "stats": {
                    "total_trades": stats["total_trades"],
                    "wins":         stats["wins"],
                    "losses":       stats["losses"],
                    "total_profit": round(stats["total_profit"], 2),
                    "total_loss":   round(stats["total_loss"], 2),
                    "net_pnl":      net_pnl,
                    "win_rate":     round((stats["wins"]/stats["total_trades"]*100) if stats["total_trades"] > 0 else 0, 1),
                    "best_trade":   stats["best_trade"],
                    "worst_trade":  stats["worst_trade"],
                }
            }
            print(f"Data ready! {len(signals)} coins | Balance: ${bal}")

            if not startup_sent and TELEGRAM_ENABLED:
                send_startup()
                startup_sent = True

        except Exception as e:
            print(f"Error: {e}")
        time.sleep(30)

@app.route("/api/data")
def api_data():
    return jsonify(latest_data)

@app.route("/api/add_trade", methods=["POST"])
def add_trade():
    from flask import request
    data = request.json
    trade_history.append(data)
    # Update stats
    if data.get("pnl") is not None:
        pnl = float(data["pnl"])
        stats["total_trades"] += 1
        if pnl > 0:
            stats["wins"] += 1
            stats["total_profit"] += pnl
            if pnl > stats["best_trade"]:
                stats["best_trade"] = pnl
        else:
            stats["losses"] += 1
            stats["total_loss"] += pnl
            if pnl < stats["worst_trade"]:
                stats["worst_trade"] = pnl
    return jsonify({"ok": True})

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>NexusAI Trading</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#020510;color:#e0e8ff;font-family:Arial,sans-serif;overflow-x:hidden;}
canvas#bg{position:fixed;top:0;left:0;z-index:0;opacity:0.25;}
.wrap{position:relative;z-index:1;padding:12px;}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:10px 20px;background:rgba(0,200,255,0.04);border:0.5px solid rgba(0,200,255,0.2);border-radius:12px;margin-bottom:10px;}
.logo{font-size:18px;font-weight:700;color:#00c8ff;letter-spacing:2px;}.logo span{color:#ff3e6c;}
.srow{display:flex;gap:10px;align-items:center;}
.dot{width:8px;height:8px;border-radius:50%;background:#00ff88;animation:p 1.5s infinite;}
@keyframes p{0%,100%{opacity:1;}50%{opacity:0.3;}}
.pill{font-size:11px;color:#8899bb;padding:3px 10px;border:0.5px solid rgba(0,200,255,0.2);border-radius:20px;}
.pill b{color:#00c8ff;}
.ticker{overflow:hidden;background:rgba(0,0,0,0.4);border-top:0.5px solid rgba(0,200,255,0.1);border-bottom:0.5px solid rgba(0,200,255,0.1);padding:5px 0;margin-bottom:10px;}
.ticker-inner{display:flex;gap:40px;animation:tk 30s linear infinite;white-space:nowrap;}
@keyframes tk{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.ti{font-size:11px;color:#8899bb;font-family:monospace;}
.ti .up{color:#00ff88;}.ti .dn{color:#ff3e6c;}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:10px;}
.m{background:rgba(0,200,255,0.03);border:0.5px solid rgba(0,200,255,0.12);border-radius:12px;padding:12px;position:relative;overflow:hidden;}
.m::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--c);}
.ml{font-size:10px;color:#6677aa;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;}
.mv{font-size:22px;font-weight:700;color:var(--c);}
.ms{font-size:10px;color:#6677aa;margin-top:3px;}

/* PNL SECTION */
.pnl-section{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px;}
.pnl-card{border-radius:12px;padding:14px;position:relative;overflow:hidden;}
.pnl-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--c);}
.pnl-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;color:#6677aa;}
.pnl-value{font-size:24px;font-weight:700;}
.pnl-sub{font-size:11px;margin-top:4px;color:#6677aa;}
.profit-card{background:rgba(0,255,136,0.05);border:0.5px solid rgba(0,255,136,0.2);--c:#00ff88;}
.loss-card{background:rgba(255,62,108,0.05);border:0.5px solid rgba(255,62,108,0.2);--c:#ff3e6c;}
.netpnl-card{background:rgba(0,200,255,0.05);border:0.5px solid rgba(0,200,255,0.2);--c:#00c8ff;}
.winrate-card{background:rgba(176,96,255,0.05);border:0.5px solid rgba(176,96,255,0.2);--c:#b060ff;}

.coins-row{display:grid;gap:10px;margin-bottom:10px;}
.coin-card{background:rgba(0,10,30,0.9);border:0.5px solid rgba(0,200,255,0.15);border-radius:12px;padding:14px;cursor:pointer;transition:all 0.3s;}
.coin-card:hover,.coin-card.sel{border-color:#00c8ff;box-shadow:0 0 20px rgba(0,200,255,0.1);}
.chead{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;}
.cname{font-size:16px;font-weight:700;}
.cprice{font-size:11px;color:#8899bb;margin-top:2px;font-family:monospace;}
.sig-buy{padding:4px 12px;border-radius:6px;font-size:12px;font-weight:700;background:rgba(0,255,136,0.1);color:#00ff88;border:0.5px solid rgba(0,255,136,0.4);}
.sig-sell{padding:4px 12px;border-radius:6px;font-size:12px;font-weight:700;background:rgba(255,62,108,0.1);color:#ff3e6c;border:0.5px solid rgba(255,62,108,0.4);}
.cstats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;}
.cst{background:rgba(0,0,0,0.3);border-radius:6px;padding:6px;text-align:center;}
.csl{font-size:9px;color:#445577;text-transform:uppercase;letter-spacing:1px;}
.csv{font-size:13px;font-weight:600;margin-top:2px;}
.conf-track{height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;margin-top:8px;}
.conf-fill{height:100%;border-radius:2px;}
.hint{text-align:center;font-size:10px;color:#445577;margin-top:8px;}
.detail{background:rgba(0,10,30,0.95);border:0.5px solid rgba(0,200,255,0.25);border-radius:12px;padding:16px;margin-bottom:10px;display:none;}
.detail.show{display:block;}
.dhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
.dtitle{font-size:14px;font-weight:700;color:#00c8ff;letter-spacing:2px;}
.cbtn{font-size:14px;cursor:pointer;background:rgba(255,62,108,0.1);border:0.5px solid rgba(255,62,108,0.3);border-radius:6px;padding:4px 12px;color:#ff3e6c;}
.dstats{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:14px;}
.ds{background:rgba(0,200,255,0.04);border:0.5px solid rgba(0,200,255,0.1);border-radius:8px;padding:10px;text-align:center;}
.dsl{font-size:9px;color:#6677aa;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;}
.dsv{font-size:16px;font-weight:700;}
.charts3{display:grid;grid-template-columns:2fr 1fr 1fr;gap:10px;margin-bottom:14px;}
.ch{background:rgba(0,0,0,0.3);border:0.5px solid rgba(0,200,255,0.08);border-radius:8px;padding:10px;}
.cht{font-size:9px;color:#445577;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.tsec{margin-top:10px;}
.tst{font-size:11px;color:#00c8ff;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;}
table{width:100%;border-collapse:collapse;}
th{font-size:10px;color:#445577;text-transform:uppercase;padding:7px 10px;border-bottom:0.5px solid rgba(0,200,255,0.08);text-align:left;}
td{font-size:12px;padding:8px 10px;border-bottom:0.5px solid rgba(255,255,255,0.03);color:#aabbcc;font-family:monospace;}
tr:hover td{background:rgba(0,200,255,0.03);}
.buy{color:#00ff88;font-weight:700;}.sell{color:#ff3e6c;font-weight:700;}
.win{color:#00ff88;}.loss{color:#ff3e6c;}
.rsi-ob{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;background:rgba(255,62,108,0.12);color:#ff3e6c;}
.rsi-ok{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;background:rgba(0,255,136,0.1);color:#00ff88;}
.rsi-os{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;background:rgba(0,160,255,0.12);color:#40a0ff;}
.bw{background:rgba(255,170,0,0.1);color:#ffaa00;padding:2px 8px;border-radius:4px;font-size:10px;}
.bwin{background:rgba(0,255,136,0.1);color:#00ff88;padding:2px 8px;border-radius:4px;font-size:10px;}
.bloss{background:rgba(255,62,108,0.1);color:#ff3e6c;padding:2px 8px;border-radius:4px;font-size:10px;}
.bgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;}
.panel{background:rgba(0,10,30,0.9);border:0.5px solid rgba(0,200,255,0.15);border-radius:12px;padding:14px;}
.ptitle{font-size:11px;font-weight:600;color:#00c8ff;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;}
.footer{text-align:center;font-size:10px;color:#445577;padding:12px;}
.nodata{text-align:center;color:#445577;padding:20px;font-size:13px;}
.pnl-chart-section{background:rgba(0,10,30,0.9);border:0.5px solid rgba(0,200,255,0.15);border-radius:12px;padding:14px;margin-bottom:10px;}
</style>
</head>
<body>
<canvas id="bg"></canvas>
<div class="wrap">

<div class="ticker"><div class="ticker-inner" id="tkr"><span class="ti">Loading...</span></div></div>

<div class="topbar">
  <div class="logo">&#9654; NEXUS<span>AI</span> TRADING SYSTEM</div>
  <div class="srow">
    <div class="dot"></div>
    <div class="pill">MODE: <b>TESTNET</b></div>
    <div class="pill">AI: <b>ONLINE</b></div>
    <div class="pill">BALANCE: <b id="t-bal">...</b></div>
    <div class="pill">TIME: <b id="t-time">--:--:--</b></div>
  </div>
</div>

<div class="metrics">
  <div class="m" style="--c:#00ff88;"><div class="ml">USDT Balance</div><div class="mv" id="usdt">Loading...</div><div class="ms">Available</div></div>
  <div class="m" style="--c:#00c8ff;"><div class="ml">Coins Tracked</div><div class="mv" id="cnt" style="color:#00c8ff;">0</div><div class="ms">Live signals</div></div>
  <div class="m" style="--c:#b060ff;"><div class="ml">AI Accuracy</div><div class="mv" style="color:#b060ff;">54.5%</div><div class="ms">RandomForest</div></div>
  <div class="m" style="--c:#ffaa00;"><div class="ml">Total Trades</div><div class="mv" id="tcnt" style="color:#ffaa00;">0</div><div class="ms">Session</div></div>
  <div class="m" style="--c:#ff3e6c;"><div class="ml">Risk Mode</div><div class="mv" style="color:#ff3e6c;">LOW</div><div class="ms">SL 2% / TP 4%</div></div>
</div>

<!-- PROFIT & LOSS SECTION -->
<div class="pnl-section">
  <div class="pnl-card profit-card">
    <div class="pnl-label">Total Profit</div>
    <div class="pnl-value" id="total-profit" style="color:#00ff88;">$0.00</div>
    <div class="pnl-sub" id="wins-count">0 winning trades</div>
  </div>
  <div class="pnl-card loss-card">
    <div class="pnl-label">Total Loss</div>
    <div class="pnl-value" id="total-loss" style="color:#ff3e6c;">$0.00</div>
    <div class="pnl-sub" id="losses-count">0 losing trades</div>
  </div>
  <div class="pnl-card netpnl-card">
    <div class="pnl-label">Net P&L</div>
    <div class="pnl-value" id="net-pnl" style="color:#00c8ff;">$0.00</div>
    <div class="pnl-sub" id="best-trade">Best: $0.00</div>
  </div>
  <div class="pnl-card winrate-card">
    <div class="pnl-label">Win Rate</div>
    <div class="pnl-value" id="win-rate" style="color:#b060ff;">0%</div>
    <div class="pnl-sub" id="worst-trade">Worst: $0.00</div>
  </div>
</div>

<!-- P&L CHART -->
<div class="pnl-chart-section">
  <div class="ptitle">Profit & Loss Chart</div>
  <div style="position:relative;height:150px;"><canvas id="pnlChart"></canvas></div>
</div>

<div class="coins-row" id="coins-row"></div>

<div class="detail" id="detail">
  <div class="dhead">
    <div class="dtitle" id="dt-title">Coin Analysis</div>
    <button class="cbtn" id="close-btn">X Close</button>
  </div>
  <div class="dstats">
    <div class="ds"><div class="dsl">Price</div><div class="dsv" id="dp" style="color:#00c8ff;">-</div></div>
    <div class="ds"><div class="dsl">Signal</div><div class="dsv" id="dsi">-</div></div>
    <div class="ds"><div class="dsl">RSI</div><div class="dsv" id="dr" style="color:#ffaa00;">-</div></div>
    <div class="ds"><div class="dsl">EMA 20</div><div class="dsv" id="de20" style="color:#b060ff;">-</div></div>
    <div class="ds"><div class="dsl">EMA 50</div><div class="dsv" id="de50" style="color:#8899bb;">-</div></div>
    <div class="ds"><div class="dsl">Confidence</div><div class="dsv" id="dc" style="color:#00ff88;">-</div></div>
  </div>
  <div class="charts3">
    <div class="ch"><div class="cht">Price History + Buy/Sell Points</div><div style="position:relative;height:200px;"><canvas id="pc"></canvas></div></div>
    <div class="ch"><div class="cht">RSI (OB=70 / OS=30)</div><div style="position:relative;height:200px;"><canvas id="rc"></canvas></div></div>
    <div class="ch"><div class="cht">Volume</div><div style="position:relative;height:200px;"><canvas id="vc"></canvas></div></div>
  </div>
  <div class="tsec">
    <div class="tst">Trade History for this Coin</div>
    <table><tr><th>#</th><th>Side</th><th>Entry Price</th><th>Exit Price</th><th>P&L ($)</th><th>P&L (%)</th><th>Time</th><th>Status</th></tr>
    <tbody id="dt"></tbody></table>
  </div>
</div>

<div class="bgrid">
  <div class="panel">
    <div class="ptitle">Market Analysis</div>
    <table><tr><th>Symbol</th><th>Price</th><th>RSI</th><th>EMA 20</th><th>Signal</th><th>Conf</th></tr>
    <tbody id="mkt"></tbody></table>
  </div>
  <div class="panel">
    <div class="ptitle">All Trade History</div>
    <table><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Price</th><th>P&L</th></tr>
    <tbody id="hist"><tr><td colspan="5" class="nodata">No trades yet</td></tr></tbody></table>
  </div>
</div>

<div class="footer">NEXUSAI TRADING SYSTEM &bull; BINANCE TESTNET &bull; CLICK COIN CARD FOR CHARTS &bull; AUTO REFRESH 5S</div>
</div>

<script>
var cv=document.getElementById('bg'),cx=cv.getContext('2d');
cv.width=window.innerWidth;cv.height=window.innerHeight;
var pp=[];
for(var i=0;i<50;i++){pp.push({x:Math.random()*cv.width,y:Math.random()*cv.height,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()+.5,a:Math.random()*.3+.1});}
function drawBG(){cx.clearRect(0,0,cv.width,cv.height);pp.forEach(function(p){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>cv.width)p.vx*=-1;if(p.y<0||p.y>cv.height)p.vy*=-1;cx.beginPath();cx.arc(p.x,p.y,p.r,0,Math.PI*2);cx.fillStyle='rgba(0,200,255,'+p.a+')';cx.fill();});requestAnimationFrame(drawBG);}
drawBG();

var COLORS=['#00c8ff','#b060ff','#ffaa00','#00ff88','#ff3e6c','#ff9900','#40e0ff','#ff60a0'];
var allData={signals:[],history:[],usdt:0,stats:{}};
var miniC={};
var PC=null,RC=null,VC=null;
var selSym=null;
var pnlHistory=[];

// P&L Chart
var pnlChart=new Chart(document.getElementById('pnlChart'),{
  type:'bar',
  data:{labels:[],datasets:[
    {label:'Profit',data:[],backgroundColor:'rgba(0,255,136,0.5)',borderRadius:4},
    {label:'Loss',data:[],backgroundColor:'rgba(255,62,108,0.5)',borderRadius:4}
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:true,labels:{color:'#8899bb',font:{size:10},boxWidth:10}}},
    scales:{
      x:{grid:{color:'rgba(0,200,255,0.04)'},ticks:{color:'#445577',font:{size:9}}},
      y:{grid:{color:'rgba(0,200,255,0.04)'},ticks:{color:'#445577',font:{size:9},callback:function(v){return '$'+v;}}}
    }
  }
});

document.getElementById('close-btn').onclick=function(){document.getElementById('detail').className='detail';selSym=null;};

function showDetail(idx){
  var s=allData.signals[idx];
  if(!s)return;
  selSym=s.symbol;
  document.getElementById('detail').className='detail show';
  document.getElementById('dt-title').textContent=s.symbol+' - Trade Analysis';
  document.getElementById('dp').textContent='$'+s.price.toLocaleString();
  var se=document.getElementById('dsi');
  se.textContent=s.signal;se.style.color=s.signal==='BUY'?'#00ff88':'#ff3e6c';
  document.getElementById('dr').textContent=s.rsi;
  document.getElementById('de20').textContent=s.ema20;
  document.getElementById('de50').textContent=s.ema50;
  document.getElementById('dc').textContent=s.confidence+'%';

  var prices=s.prices||[s.price];
  var rsis=s.rsis||[s.rsi];
  var vols=s.vols||[1000000];
  var labs=[];
  for(var i=0;i<prices.length;i++){labs.push(i===prices.length-1?'Now':(prices.length-1-i)+'h');}

  var coinTrades=(allData.history||[]).filter(function(t){return t.symbol===s.symbol;});
  var buyPts=new Array(prices.length).fill(null);
  var sellPts=new Array(prices.length).fill(null);
  coinTrades.forEach(function(t){var li=prices.length-1;if(t.side==='BUY')buyPts[li]=t.price;else sellPts[li]=t.price;});

  if(PC)PC.destroy();
  PC=new Chart(document.getElementById('pc'),{type:'line',data:{labels:labs,datasets:[{label:'Price',data:prices,borderColor:'#00c8ff',backgroundColor:'rgba(0,200,255,0.08)',borderWidth:2,fill:true,tension:0.4,pointRadius:2,order:2},{label:'BUY',data:buyPts,borderColor:'transparent',backgroundColor:'#00ff88',pointRadius:12,pointStyle:'triangle',showLine:false,order:1},{label:'SELL',data:sellPts,borderColor:'transparent',backgroundColor:'#ff3e6c',pointRadius:12,pointStyle:'triangle',showLine:false,rotation:180,order:1}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,labels:{color:'#8899bb',font:{size:10},boxWidth:10}},tooltip:{backgroundColor:'rgba(0,10,30,0.95)',borderColor:'rgba(0,200,255,0.3)',borderWidth:1,titleColor:'#00c8ff',bodyColor:'#e0e8ff',callbacks:{label:function(c){if(c.dataset.label==='BUY'&&c.raw!==null)return 'BUY @ $'+c.raw.toLocaleString();if(c.dataset.label==='SELL'&&c.raw!==null)return 'SELL @ $'+c.raw.toLocaleString();return '$'+c.raw.toLocaleString();}}}},scales:{x:{grid:{color:'rgba(0,200,255,0.04)'},ticks:{color:'#445577',font:{size:9}}},y:{grid:{color:'rgba(0,200,255,0.04)'},ticks:{color:'#445577',font:{size:9},callback:function(v){return '$'+v.toLocaleString();}}}}}});

  if(RC)RC.destroy();
  RC=new Chart(document.getElementById('rc'),{type:'line',data:{labels:labs,datasets:[{label:'RSI',data:rsis,borderColor:'#ffaa00',backgroundColor:'rgba(255,170,0,0.06)',borderWidth:2,fill:true,tension:0.4,pointRadius:2},{label:'OB',data:new Array(labs.length).fill(70),borderColor:'rgba(255,62,108,0.5)',borderDash:[4,4],pointRadius:0,fill:false,borderWidth:1},{label:'OS',data:new Array(labs.length).fill(30),borderColor:'rgba(0,200,255,0.5)',borderDash:[4,4],pointRadius:0,fill:false,borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(0,200,255,0.04)'},ticks:{color:'#445577',font:{size:9}}},y:{min:0,max:100,grid:{color:'rgba(0,200,255,0.04)'},ticks:{color:'#445577',font:{size:9}}}}}});

  if(VC)VC.destroy();
  VC=new Chart(document.getElementById('vc'),{type:'bar',data:{labels:labs,datasets:[{data:vols,backgroundColor:prices.map(function(p,i){return i>0&&prices[i]>=prices[i-1]?'rgba(0,255,136,0.5)':'rgba(255,62,108,0.5)';}),borderRadius:2}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(0,200,255,0.03)'},ticks:{color:'#445577',font:{size:8}}},y:{grid:{color:'rgba(0,200,255,0.03)'},ticks:{color:'#445577',font:{size:8},callback:function(v){return (v/1000).toFixed(0)+'K';}}}}}});

  var th='';
  if(coinTrades.length===0){
    th='<tr><td colspan="8" class="nodata">No trades for '+s.symbol+' yet</td></tr>';
  } else {
    coinTrades.forEach(function(t,i){
      var ep=t.exitPrice;
      var pnlDollar='-',pnlPct='-',pc2='',badge='<span class="bw">OPEN</span>';
      if(ep&&t.pnl!==undefined){
        var pv=t.pnl;
        var pvDollar=((ep-t.price)/t.price*20).toFixed(2);
        pnlPct=(pv>0?'+':'')+pv+'%';
        pnlDollar=(pvDollar>0?'+$':'$')+Math.abs(pvDollar);
        pc2=pv>0?'win':'loss';
        badge=pv>0?'<span class="bwin">WIN</span>':'<span class="bloss">LOSS</span>';
      }
      th+='<tr><td style="color:#8899bb;">'+(i+1)+'</td>';
      th+='<td class="'+(t.side==='BUY'?'buy':'sell')+'">'+t.side+'</td>';
      th+='<td>$'+t.price.toLocaleString()+'</td>';
      th+='<td>'+(ep?'$'+ep.toLocaleString():'-')+'</td>';
      th+='<td class="'+pc2+'">'+pnlDollar+'</td>';
      th+='<td class="'+pc2+'">'+pnlPct+'</td>';
      th+='<td>'+t.time+'</td>';
      th+='<td>'+badge+'</td></tr>';
    });
  }
  document.getElementById('dt').innerHTML=th;
  document.getElementById('detail').scrollIntoView({behavior:'smooth',block:'start'});
}

function updatePnlCards(stats){
  if(!stats)return;
  var profit=stats.total_profit||0;
  var loss=stats.total_loss||0;
  var net=stats.net_pnl||0;
  var wr=stats.win_rate||0;

  document.getElementById('total-profit').textContent='+$'+profit.toFixed(2);
  document.getElementById('wins-count').textContent=(stats.wins||0)+' winning trades';
  document.getElementById('total-loss').textContent='-$'+Math.abs(loss).toFixed(2);
  document.getElementById('losses-count').textContent=(stats.losses||0)+' losing trades';

  var netEl=document.getElementById('net-pnl');
  netEl.textContent=(net>=0?'+$':'-$')+Math.abs(net).toFixed(2);
  netEl.style.color=net>=0?'#00ff88':'#ff3e6c';

  document.getElementById('win-rate').textContent=wr+'%';
  document.getElementById('best-trade').textContent='Best: +$'+(stats.best_trade||0).toFixed(2);
  document.getElementById('worst-trade').textContent='Worst: $'+(stats.worst_trade||0).toFixed(2);

  // Update P&L chart
  if(stats.total_trades>0){
    pnlChart.data.labels=['Profit','Loss','Net'];
    pnlChart.data.datasets[0].data=[profit,0,net>=0?net:0];
    pnlChart.data.datasets[1].data=[0,Math.abs(loss),net<0?Math.abs(net):0];
    pnlChart.update('none');
  }
}

function render(data){
  allData=data;
  var sigs=data.signals||[];
  document.getElementById('usdt').textContent='$'+data.usdt;
  document.getElementById('t-bal').textContent='$'+data.usdt;
  document.getElementById('cnt').textContent=sigs.length;
  document.getElementById('tcnt').textContent=(data.history||[]).length;

  updatePnlCards(data.stats);

  var cols=sigs.length<=3?'repeat('+sigs.length+',1fr)':'repeat(3,1fr)';
  document.getElementById('coins-row').style.gridTemplateColumns=cols;

  var ch='';
  sigs.forEach(function(s,i){
    var c=COLORS[i%COLORS.length];var isBuy=s.signal==='BUY';
    var rc2=s.rsi>65?'rsi-ob':s.rsi<35?'rsi-os':'rsi-ok';
    var sel=selSym===s.symbol?' sel':'';
    ch+='<div class="coin-card'+sel+'" id="card-'+i+'" style="border-top:2px solid '+c+';">';
    ch+='<div class="chead"><div><div class="cname" style="color:'+c+';">'+s.symbol.replace('USDT','')+'<span style="font-size:11px;color:#6677aa;">/USDT</span></div><div class="cprice">$'+s.price.toLocaleString()+'</div></div><span class="sig-'+(isBuy?'buy':'sell')+'">'+s.signal+'</span></div>';
    ch+='<div class="cstats"><div class="cst"><div class="csl">RSI</div><div class="csv"><span class="'+rc2+'">'+s.rsi+'</span></div></div><div class="cst"><div class="csl">EMA 20</div><div class="csv" style="color:#b060ff;font-size:11px;">'+s.ema20+'</div></div><div class="cst"><div class="csl">Confidence</div><div class="csv" style="color:'+(isBuy?'#00ff88':'#ff3e6c')+';">'+s.confidence+'%</div></div></div>';
    ch+='<div style="position:relative;height:70px;"><canvas id="cc-'+i+'"></canvas></div>';
    ch+='<div class="conf-track"><div class="conf-fill" style="width:'+s.confidence+'%;background:'+(isBuy?'#00ff88':'#ff3e6c')+';transition:width 1s;"></div></div>';
    ch+='<div class="hint">Click to see price chart and trade history</div></div>';
  });
  document.getElementById('coins-row').innerHTML=ch;

  sigs.forEach(function(s,i){
    var el=document.getElementById('cc-'+i);if(!el)return;
    if(miniC[i])miniC[i].destroy();
    var prices=s.prices||[s.price];var c=COLORS[i%COLORS.length];
    miniC[i]=new Chart(el,{type:'line',data:{labels:prices.map(function(_,j){return j;}),datasets:[{data:prices,borderColor:c,backgroundColor:c+'33',borderWidth:1.5,fill:true,tension:0.4,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{enabled:false}},scales:{x:{display:false},y:{display:false}},animation:{duration:300}}});
    document.getElementById('card-'+i).onclick=(function(idx){return function(){showDetail(idx);};})(i);
  });

  var tw='';for(var r=0;r<2;r++){sigs.forEach(function(s){var isBuy=s.signal==='BUY';tw+='<span class="ti">'+s.symbol+' <span class="'+(isBuy?'up':'dn')+'">$'+s.price.toLocaleString()+'</span> &bull; <span class="'+(isBuy?'up':'dn')+'">'+s.signal+' '+s.confidence+'%</span>&nbsp;&nbsp;&nbsp;</span>';});}
  document.getElementById('tkr').innerHTML=tw||'<span class="ti">Loading...</span>';

  var mt='';sigs.forEach(function(s,i){var rc2=s.rsi>65?'rsi-ob':s.rsi<35?'rsi-os':'rsi-ok';mt+='<tr style="cursor:pointer;" id="mr-'+i+'"><td style="color:#e0e8ff;font-weight:700;">'+s.symbol+'</td><td>$'+s.price.toLocaleString()+'</td><td><span class="'+rc2+'">'+s.rsi+'</span></td><td>'+s.ema20+'</td><td class="'+(s.signal==='BUY'?'buy':'sell')+'">'+s.signal+'</td><td>'+s.confidence+'%</td></tr>';});
  document.getElementById('mkt').innerHTML=mt||'<tr><td colspan="6" class="nodata">Loading...</td></tr>';
  sigs.forEach(function(s,i){var el=document.getElementById('mr-'+i);if(el)el.onclick=(function(idx){return function(){showDetail(idx);};})(i);});

  var hh='';
  if(data.history&&data.history.length>0){
    data.history.slice().reverse().slice(0,20).forEach(function(t){
      var pnlStr='-';var pnlCls='';
      if(t.pnl!==undefined){pnlStr=(t.pnl>0?'+':'')+t.pnl+'%';pnlCls=t.pnl>0?'win':'loss';}
      hh+='<tr><td>'+t.time+'</td><td style="color:#e0e8ff;">'+t.symbol+'</td><td class="'+(t.side==='BUY'?'buy':'sell')+'">'+t.side+'</td><td>$'+t.price+'</td><td class="'+pnlCls+'">'+pnlStr+'</td></tr>';
    });
  } else {hh='<tr><td colspan="5" class="nodata">No trades yet</td></tr>';}
  document.getElementById('hist').innerHTML=hh;

  if(selSym){var idx=sigs.findIndex(function(s){return s.symbol===selSym;});if(idx>=0)showDetail(idx);}
}

function fetchData(){fetch('/api/data').then(function(r){return r.json();}).then(function(d){render(d);}).catch(function(e){console.log('err',e);});}
fetchData();
setInterval(fetchData,5000);
setInterval(function(){document.getElementById('t-time').textContent=new Date().toTimeString().slice(0,8);},1000);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    print("\n"+"="*50)
    print("   NEXUSAI TRADING BOT — WITH P&L TRACKING")
    print("="*50)
    print(f"Tracking {len(config.SYMBOLS)} coins")
    print("Open Chrome: http://localhost:5000")
    print("="*50+"\n")
    t = threading.Thread(target=fetch_loop)
    t.daemon = True
    t.start()
    app.run(debug=False, port=5000)
