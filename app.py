import os
import requests
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update, dash_table
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import griddata
import yfinance as yf
import pandas as pd
import datetime
try:
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1. SETTINGS & DEFAULTS
# -----------------------------------------------------------------------------
DEFAULT_TICKER = "SPY"
DEFAULT_SPOT = 400.00
DEFAULT_STRIKE = 400.00
DEFAULT_TIME = 1.0
DEFAULT_VOL = 0.2
DEFAULT_RATE = 0.04
DEFAULT_SPREAD_A = "SPY"
DEFAULT_SPREAD_B = "GLD"
DEFAULT_SCANNER_TICKERS = "SPY, QQQ, GLD, SLV, TLT, USO"
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
ORATS_API_KEY   = os.environ.get('ORATS_API_KEY', '')

# --- Historical vol-surface date slider: weekly steps for the past ~2 years ---

colors = {
    'background': '#000000', 'text': '#ffffff', 'card_bg': '#0d0d0d',
    'input_bg': '#1a1a1a', 'call_text': '#00c800', 'put_text': '#ff3300',
    'accent': '#ff6600', 'success': '#00c800', 'danger': '#ff3300',
    'muted': '#666',
    'card_border': '#222',
}

layout_settings = dict(
    template='plotly_dark',
    paper_bgcolor=colors['card_bg'],
    plot_bgcolor=colors['card_bg'],
    font=dict(color=colors['text']),
    hovermode="x unified",
    autosize=True
)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_initial_price(ticker_symbol):
    try:
        init_ticker = yf.Ticker(ticker_symbol)
        init_hist = init_ticker.history(period="1d")
        if not init_hist.empty:
            return round(float(init_hist['Close'].iloc[-1]), 2)
    except:
        pass
    return 400.00

initial_spot = get_initial_price(DEFAULT_TICKER)
initial_strike = initial_spot



# CHANGE: Helper to build a stat row (label: value) used in sidebar panels
def make_stat_row(label, value, value_color=None):
    """Creates a label-value row for stats panels."""
    return html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '8px'}, children=[
        html.Span(label, style={'color': colors['muted'], 'fontSize': '0.9em'}),
        html.Span(str(value), style={'fontWeight': 'bold', 'color': value_color or colors['text'], 'fontSize': '0.95em'})
    ])


def generate_vol_signals(current_iv, current_hv, rvr, skew_ratio, term_slope):
    """Generate trading signals from volatility smile data."""
    signals = []
    if current_iv and current_hv:
        r = current_iv / current_hv
        if r > 1.25:
            signals.append({'label': 'VRP', 'signal': 'Sell Vol', 'detail': f'IV/HV={r:.2f}x — options overpriced vs realized', 'badge': 'badge-red'})
        elif r < 0.80:
            signals.append({'label': 'VRP', 'signal': 'Buy Vol', 'detail': f'IV/HV={r:.2f}x — options cheap vs realized', 'badge': 'badge-green'})
        else:
            signals.append({'label': 'VRP', 'signal': 'Neutral', 'detail': f'IV/HV={r:.2f}x — fairly priced', 'badge': 'badge-orange'})
    if skew_ratio is not None:
        if skew_ratio > 1.35:
            signals.append({'label': 'Skew', 'signal': 'Bearish Fear', 'detail': f'OTM puts {skew_ratio:.2f}x ATM — market pricing tail risk', 'badge': 'badge-red'})
        elif skew_ratio < 1.05:
            signals.append({'label': 'Skew', 'signal': 'Complacent', 'detail': 'Flat skew — little downside demand', 'badge': 'badge-orange'})
        else:
            signals.append({'label': 'Skew', 'signal': 'Normal Skew', 'detail': f'OTM puts {skew_ratio:.2f}x ATM — healthy premium', 'badge': 'badge-green'})
    if rvr > 80:
        signals.append({'label': 'HV Rank', 'signal': 'Vol Elevated', 'detail': f'{rvr:.0f}th pct — mean reversion likely, sell realized vol', 'badge': 'badge-red'})
    elif rvr < 20:
        signals.append({'label': 'HV Rank', 'signal': 'Vol Suppressed', 'detail': f'{rvr:.0f}th pct — expansion likely, buy vol', 'badge': 'badge-green'})
    else:
        signals.append({'label': 'HV Rank', 'signal': 'Vol Normal', 'detail': f'{rvr:.0f}th pct — within historical range', 'badge': 'badge-orange'})
    if term_slope is not None:
        if term_slope < -2:
            signals.append({'label': 'Term Struct', 'signal': 'Backwardation', 'detail': 'Near-term IV > far-term — stress or event risk priced in', 'badge': 'badge-red'})
        elif term_slope > 2:
            signals.append({'label': 'Term Struct', 'signal': 'Steep Contango', 'detail': 'Far-term IV >> near-term — calm now, risk priced ahead', 'badge': 'badge-orange'})
        else:
            signals.append({'label': 'Term Struct', 'signal': 'Normal Curve', 'detail': 'Balanced term structure', 'badge': 'badge-green'})
    return signals




def fetch_orats_surface(ticker_symbol, contract_type, moneyness_pct):
    """Pull a live options chain from ORATS /live/strikes and return (data_dict, error_string)."""
    if not ORATS_API_KEY:
        return None, "ORATS_API_KEY not set."
    try:
        resp = requests.get(
            'https://api.orats.io/datav2/live/strikes',
            params={'token': ORATS_API_KEY, 'ticker': ticker_symbol.upper()},
            timeout=20
        )
        if resp.status_code in (401, 403):
            return None, "ORATS key unauthorized — check your subscription."
        resp.raise_for_status()

        rows = resp.json().get('data', [])
        if not rows:
            return None, f"No ORATS strike data for {ticker_symbol}"

        today = datetime.date.today()
        spot  = rows[0].get('stockPrice') or get_initial_price(ticker_symbol) or None

        strikes, dtes, ivs, prices, deltas, volumes, exps, ctypes = [], [], [], [], [], [], [], []
        expirations_seen = set()

        for row in rows:
            exp_str = row.get('expirDate')
            strike  = row.get('strike')
            if not exp_str or strike is None:
                continue
            exp_date = datetime.date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if dte <= 0:
                continue
            if spot and not (spot * (1 - moneyness_pct) <= strike <= spot * (1 + moneyness_pct)):
                continue

            for ctype, iv_key, price_key, delta_key, vol_key in [
                ('call', 'callMidIv',  'callMidPrice', 'callDelta', 'callVolume'),
                ('put',  'putMidIv',   'putMidPrice',  'putDelta',  'putVolume'),
            ]:
                if contract_type not in (ctype, 'both'):
                    continue
                iv = row.get(iv_key)
                if not iv or iv < 0.005:
                    continue
                strikes.append(strike)
                dtes.append(dte)
                ivs.append(iv)
                prices.append(row.get(price_key))
                deltas.append(abs(row.get(delta_key) or 0) or None)
                volumes.append(row.get(vol_key) or 0)
                exps.append(exp_str)
                ctypes.append(ctype)
                expirations_seen.add(exp_str)

        if len(strikes) < 5:
            return None, "Too few liquid contracts. Try widening the moneyness range."

        return {
            "strikes": strikes, "dtes": dtes, "ivs": ivs, "prices": prices,
            "deltas": deltas, "volumes": volumes, "exps": exps, "ctypes": ctypes,
            "spot": spot,
            "contract_count": len(strikes),
            "exp_count": len(expirations_seen),
        }, None

    except requests.exceptions.Timeout:
        return None, "ORATS request timed out"
    except Exception as e:
        return None, f"ORATS error: {e}"


def scan_ticker(ticker_symbol, target_dte=30):
    """Fetch vol metrics for one ticker using Polygon.io and return a table row dict."""
    if not POLYGON_AVAILABLE or not POLYGON_API_KEY:
        return None
    try:
        client = PolygonClient(api_key=POLYGON_API_KEY)
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=365)

        # --- Historical daily closes for HV ---
        aggs = list(client.list_aggs(
            ticker_symbol.upper(), 1, 'day',
            from_date.isoformat(), today.isoformat(),
            adjusted=True, sort='asc', limit=365
        ))
        if len(aggs) < 31:
            return None

        closes = np.array([a.close for a in aggs])
        log_rets = np.log(closes[1:] / closes[:-1])
        hv_series = [np.std(log_rets[i - 29:i + 1], ddof=1) * np.sqrt(252) * 100
                     for i in range(29, len(log_rets))]
        current_hv = hv_series[-1]
        hv_min, hv_max = min(hv_series), max(hv_series)
        hv_rank = ((current_hv - hv_min) / (hv_max - hv_min) * 100) if hv_max > hv_min else 50.0
        spot = closes[-1]

        # get_previous_close_agg gives the actual last trading day's close;
        # list_aggs can return data several days stale on this plan.
        try:
            prev = client.get_previous_close_agg(ticker_symbol.upper())
            if prev and prev[0].close and prev[0].close > 0:
                spot = prev[0].close
        except Exception:
            pass

        # --- Options chain snapshot ---
        current_iv, skew_ratio, pc_vol_ratio = None, None, None
        try:
            contracts = list(client.list_snapshot_options_chain(
                ticker_symbol.upper(), params={'limit': 250}
            ))
        except Exception:
            contracts = []

        if contracts:
            # Find nearest expiry >= 7 days out
            valid_exps = sorted({
                c.details.expiration_date for c in contracts
                if c.details and c.details.expiration_date
                and (datetime.date.fromisoformat(c.details.expiration_date) - today).days >= 7
            })
            if valid_exps:
                nearest_exp = min(valid_exps, key=lambda e: abs((datetime.date.fromisoformat(e) - today).days - target_dte))

                # Split into calls/puts for nearest expiry with valid IV
                near = [c for c in contracts
                        if c.details and c.details.expiration_date == nearest_exp
                        and c.implied_volatility and c.implied_volatility > 0.005]
                calls = [c for c in near if c.details.contract_type == 'call']
                puts  = [c for c in near if c.details.contract_type == 'put']

                # ATM call IV
                if calls:
                    atm_call = min(calls, key=lambda c: abs(c.details.strike_price - spot))
                    current_iv = atm_call.implied_volatility * 100

                # Put skew: 90% moneyness put IV vs ATM call IV
                if puts and current_iv:
                    atm_put = min(puts, key=lambda c: abs(c.details.strike_price - spot * 0.90))
                    skew_ratio = atm_put.implied_volatility * 100 / current_iv

                # Put/Call volume ratio (all near-term contracts)
                call_vol = sum(c.day.volume for c in calls if c.day and c.day.volume)
                put_vol  = sum(c.day.volume for c in puts  if c.day and c.day.volume)
                if call_vol > 0:
                    pc_vol_ratio = put_vol / call_vol

        vrp_ratio = (current_iv / current_hv) if current_iv and current_hv else None
        verdict = ('Market Closed' if current_iv is not None and current_iv == 0
                   else 'Expensive' if vrp_ratio and vrp_ratio > 1.25
                   else 'Cheap' if vrp_ratio and vrp_ratio < 0.80
                   else 'Neutral' if vrp_ratio else 'N/A')

        skew_bearish = skew_ratio is not None and skew_ratio > 1.15
        skew_bullish = skew_ratio is not None and skew_ratio < 0.88
        pc_bearish   = pc_vol_ratio is not None and pc_vol_ratio > 1.10
        pc_bullish   = pc_vol_ratio is not None and pc_vol_ratio < 0.75
        bear_votes = int(skew_bearish) + int(pc_bearish)
        bull_votes = int(skew_bullish) + int(pc_bullish)
        bias = "Bearish" if bear_votes > bull_votes else "Bullish" if bull_votes > bear_votes else "Neutral"

        iv_cheap     = vrp_ratio is not None and vrp_ratio < 0.85
        iv_expensive = vrp_ratio is not None and vrp_ratio > 1.25

        if bias == "Bullish" and iv_cheap:
            call_rec = "Buy"
        elif bias == "Bearish" and iv_expensive:
            call_rec = "Sell"
        else:
            call_rec = "Hold"

        if bias == "Bearish" and iv_cheap:
            put_rec = "Buy"
        elif bias == "Bullish" and iv_expensive:
            put_rec = "Sell"
        else:
            put_rec = "Hold"

        return {
            'Ticker':   ticker_symbol,
            'Price':    f"${spot:.2f}",
            'IV %':     f"{current_iv:.1f}" if current_iv else 'N/A',
            'HV %':     f"{current_hv:.1f}",
            'IV/HV':    f"{vrp_ratio:.2f}x" if vrp_ratio else 'N/A',
            'Skew':     f"{skew_ratio:.2f}x" if skew_ratio else 'N/A',
            'P/C Vol':  f"{pc_vol_ratio:.2f}" if pc_vol_ratio else 'N/A',
            'HV Rank':  f"{hv_rank:.0f}%",
            'Bias':     bias,
            'Call Rec': call_rec,
            'Put Rec':  put_rec,
            'Verdict':  verdict,
        }
    except Exception as e:
        return {'Ticker': ticker_symbol, '_error': str(e)}


def scan_ticker_orats(ticker_symbol, target_dte=30):
    """Fetch vol metrics for one ticker using ORATS summaries API."""
    if not ORATS_API_KEY:
        return {'Ticker': ticker_symbol, '_error': 'ORATS_API_KEY not set. Add it in Render → Environment.'}
    try:
        resp = requests.get(
            'https://api.orats.io/datav2/live/summaries',
            params={'token': ORATS_API_KEY, 'ticker': ticker_symbol.upper()},
            timeout=10
        )
        if resp.status_code in (401, 403):
            return {'Ticker': ticker_symbol, '_error': 'ORATS key unauthorized — check your subscription.'}
        resp.raise_for_status()

        data = resp.json().get('data', [])
        if not data:
            return {'Ticker': ticker_symbol, '_error': f'No ORATS data for {ticker_symbol}'}

        d = data[0]
        spot   = d.get('stockPrice')
        atm_iv = d.get('iv30d')   # 30-day ATM implied vol (decimal)
        hv20   = d.get('rVol30')  # 30-day realized vol (decimal)

        if not spot or not atm_iv or not hv20:
            missing = [k for k, v in [('stockPrice', spot), ('iv30d', atm_iv), ('rVol30', hv20)] if not v]
            return {'Ticker': ticker_symbol, '_error': f'Missing fields: {missing}'}

        current_iv = atm_iv * 100
        current_hv = hv20   * 100
        vrp_ratio  = current_iv / current_hv if current_hv else None

        # Skew: 25-delta put IV / 75-delta call IV (both already in summaries)
        put_iv_d25  = d.get('dlt25Iv30d')
        call_iv_d75 = d.get('dlt75Iv30d')
        skew_ratio  = (put_iv_d25 / call_iv_d75) if (put_iv_d25 and call_iv_d75 and call_iv_d75 > 0) else None

        verdict = ('Expensive' if vrp_ratio and vrp_ratio > 1.25
                   else 'Cheap' if vrp_ratio and vrp_ratio < 0.80
                   else 'Neutral' if vrp_ratio else 'N/A')

        # Supplement with yfinance: P/C vol from options chain + HV rank from price history
        pc_vol_ratio = None
        hv_rank_val  = None
        try:
            import yfinance as yf
            tkr  = yf.Ticker(ticker_symbol)
            exps = tkr.options
            if exps:
                chain     = tkr.option_chain(exps[0])
                c_vol     = chain.calls['volume'].sum()
                p_vol     = chain.puts['volume'].sum()
                pc_vol_ratio = p_vol / c_vol if c_vol > 0 else None
            hist = tkr.history(period='1y')
            if len(hist) > 25:
                log_ret    = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                roll_hv    = log_ret.rolling(20).std() * np.sqrt(252) * 100
                roll_hv    = roll_hv.dropna()
                cur_hv     = roll_hv.iloc[-1]
                hv_rank_val = (roll_hv < cur_hv).mean() * 100
        except Exception:
            pass

        skew_bearish = skew_ratio is not None and skew_ratio > 1.15
        skew_bullish = skew_ratio is not None and skew_ratio < 0.88
        pc_bearish   = pc_vol_ratio is not None and pc_vol_ratio > 1.10
        pc_bullish   = pc_vol_ratio is not None and pc_vol_ratio < 0.75
        bear_votes   = int(skew_bearish) + int(pc_bearish)
        bull_votes   = int(skew_bullish) + int(pc_bullish)
        bias         = "Bearish" if bear_votes > bull_votes else "Bullish" if bull_votes > bear_votes else "Neutral"

        iv_cheap     = vrp_ratio is not None and vrp_ratio < 0.85
        iv_expensive = vrp_ratio is not None and vrp_ratio > 1.25
        call_rec = ("Buy"  if bias == "Bullish" and iv_cheap
                    else "Sell" if bias == "Bearish" and iv_expensive else "Hold")
        put_rec  = ("Buy"  if bias == "Bearish" and iv_cheap
                    else "Sell" if bias == "Bullish" and iv_expensive else "Hold")

        return {
            'Ticker':   ticker_symbol,
            'Price':    f"${spot:.2f}",
            'IV %':     f"{current_iv:.1f}",
            'HV %':     f"{current_hv:.1f}",
            'IV/HV':    f"{vrp_ratio:.2f}x" if vrp_ratio else 'N/A',
            'Skew':     f"{skew_ratio:.2f}x" if skew_ratio else 'N/A',
            'P/C Vol':  f"{pc_vol_ratio:.2f}" if pc_vol_ratio is not None else 'N/A',
            'HV Rank':  f"{hv_rank_val:.0f}%" if hv_rank_val is not None else 'N/A',
            'Bias':     bias,
            'Call Rec': call_rec,
            'Put Rec':  put_rec,
            'Verdict':  verdict,
        }
    except requests.exceptions.Timeout:
        return {'Ticker': ticker_symbol, '_error': 'ORATS request timed out'}
    except Exception as e:
        return {'Ticker': ticker_symbol, '_error': f'ORATS error: {e}'}


def fetch_orats_hist_iv(ticker_symbol):
    """Fetch full historical daily IV/HV from ORATS /hist/summaries. Returns (DataFrame, error_string)."""
    if not ORATS_API_KEY:
        return None, "ORATS_API_KEY not set."
    try:
        params = {'token': ORATS_API_KEY, 'ticker': ticker_symbol.upper()}
        resp = requests.get(
            'https://api.orats.io/datav2/hist/summaries',
            params=params,
            timeout=20
        )
        if resp.status_code in (401, 403):
            return None, "ORATS key unauthorized."
        resp.raise_for_status()
        rows = resp.json().get('data', [])
        if not rows:
            return None, f"No ORATS historical data for {ticker_symbol}"
        df = pd.DataFrame(rows)
        df['tradeDate'] = pd.to_datetime(df['tradeDate'])
        df = df.sort_values('tradeDate').reset_index(drop=True)
        return df, None
    except requests.exceptions.Timeout:
        return None, "ORATS request timed out"
    except Exception as e:
        return None, f"ORATS error: {e}"


# -----------------------------------------------------------------------------
# 3. APP LAYOUT & STYLES (Mobile Optimized)
# -----------------------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, title='Equity Research Dashboard',
                update_title=None,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

SIDEBAR_STYLE = {
    'flex': '1 1 350px',
    'backgroundColor': colors['card_bg'], 'padding': '20px',
    'borderRadius': '10px', 'boxSizing': 'border-box',
    'display': 'flex', 'flexDirection': 'column',
    'marginBottom': '20px',
    'border': f"1px solid {colors['card_border']}",  # CHANGE: Added subtle border for depth
}

CONTENT_STYLE = {
    'flex': '3 1 500px',
    'backgroundColor': colors['card_bg'], 'padding': '10px',
    'borderRadius': '10px', 'minHeight': '400px', 'boxSizing': 'border-box',
    'overflow': 'hidden',
    'border': f"1px solid {colors['card_border']}",  # CHANGE: Added subtle border for depth
}

FLEX_WRAPPER_STYLE = {
    'display': 'flex',
    'flexWrap': 'wrap',
    'gap': '20px',
}

# CHANGE: Standardized input style to reduce repetition and ensure consistency
INPUT_STYLE = {
    'width': '100%', 'boxSizing': 'border-box', 'padding': '10px',
    'backgroundColor': colors['input_bg'], 'color': 'white',
    'border': '1px solid #333', 'borderRadius': '6px', 'fontSize': '0.95em'
}

# CHANGE: Standardized button style
BUTTON_STYLE = {
    'width': '100%', 'boxSizing': 'border-box', 'padding': '12px',
    'backgroundColor': colors['accent'], 'border': 'none', 'borderRadius': '6px',
    'fontWeight': 'bold', 'cursor': 'pointer', 'fontSize': '0.95em',
    'color': '#000000', 'letterSpacing': '0.3px'
}

def make_control_row(label, id_prefix, min_val, max_val, step, default_val, helper=None):
    """CHANGE: Added optional helper text parameter to explain each input to the user."""
    children = [
        html.Label(label, style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
    ]
    # CHANGE: Add helper text below label if provided
    if helper:
        children.append(html.Div(helper, className='helper-text'))
    children.append(
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
            dcc.Input(id=f'{id_prefix}-input', type='number', value=default_val, step=step,
                      style={'width': '80px', 'padding': '6px', 'backgroundColor': colors['input_bg'], 'color': 'white', 'border': '1px solid #333', 'borderRadius': '6px', 'fontSize': '0.9em'}),
            html.Div(style={'flex': '1'}, children=[
                dcc.Slider(id=f'{id_prefix}-slider', min=min_val, max=max_val, step=step, value=default_val, marks=None, tooltip={"placement": "bottom", "always_visible": True})
            ])
        ])
    )
    return html.Div(style={'marginBottom': '18px'}, children=children)


# CHANGE: Empty state placeholder for charts before data loads
def make_empty_chart(message="Enter a ticker and click Analyze to get started."):
    """Creates a styled empty figure with a helpful message."""
    fig = go.Figure()
    fig.update_layout(
        **layout_settings,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message,
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=colors['muted'])
        )]
    )
    return fig



# --- 3. SPREAD ANALYSIS TAB LAYOUT ---
PERIOD_MAP = {0: '1mo', 1: '3mo', 2: '6mo', 3: '1y', 4: '3y', 5: '5y', 6: '10y', 7: 'max'}
CAL_PERIOD_MAP = {0: '1y', 1: '2y', 2: '3y', 3: '5y', 4: '10y', 5: 'max'}

SPREAD_LINE_COLORS = [
    '#FF3333',  # red          (0°)
    '#3366FF',  # blue         (225°)
    '#33CC55',  # green        (135°)
    '#FF9900',  # orange       (36°)
    '#CC33FF',  # purple       (275°)
    '#00CCDD',  # cyan         (185°)
    '#FFD700',  # yellow       (51°)
    '#FF44AA',  # hot pink     (325°)
    '#00BB88',  # teal         (163°)
    '#FF6633',  # red-orange   (15°)
    '#55AAFF',  # sky blue     (210°)
    '#99DD00',  # lime         (88°)
    '#CC33BB',  # violet       (308°)
    '#00DDBB',  # turquoise    (172°)
    '#FFAA33',  # amber        (38°)
    '#4455DD',  # indigo       (236°)
    '#FF8899',  # salmon       (352°)
    '#44CC88',  # mint         (150°)
    '#DD6622',  # burnt orange (22°)
    '#88AAFF',  # periwinkle   (228°)
]

spread_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Spread Analysis", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Normalize multiple tickers to 100 and compare relative performance.", className='helper-text', style={'marginTop': '0'}),

            html.Label("Tickers", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Div("Comma-separated. Add as many as you like.", className='helper-text'),
            dcc.Textarea(id='spread-tickers-input', value=f"{DEFAULT_SPREAD_A}, {DEFAULT_SPREAD_B}",
                         placeholder="e.g. SPY, QQQ, DIA",
                         style={**INPUT_STYLE, 'height': '70px', 'resize': 'vertical', 'fontFamily': 'monospace'}),

            html.Label("Date Range", style={'color': colors['text'], 'fontWeight': 'bold', 'display': 'block', 'fontSize': '0.9em', 'marginTop': '12px', 'marginBottom': '4px'}),
            html.Div("Drag both handles to set start and end.", className='helper-text'),
            html.Div(style={'padding': '0 10px 28px 10px'}, children=[
                dcc.RangeSlider(
                    id='spread-period-slider',
                    min=0, max=12, step=1,
                    value=[4, 12],
                    marks={
                        0:  {'label': 'MAX', 'style': {'color': '#888', 'fontSize': '0.75em'}},
                        2:  {'label': '15Y', 'style': {'color': '#888', 'fontSize': '0.75em'}},
                        4:  {'label': '10Y', 'style': {'color': '#888', 'fontSize': '0.75em'}},
                        6:  {'label': '5Y',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                        7:  {'label': '3Y',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                        8:  {'label': '2Y',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                        9:  {'label': '1Y',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                        10: {'label': '6M',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                        11: {'label': '3M',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                        12: {'label': 'Now', 'style': {'color': '#888', 'fontSize': '0.75em'}},
                    },
                    tooltip=None,
                )
            ]),

            html.Button('Analyze', id='spread-analyze-btn', n_clicks=0, style=BUTTON_STYLE),
            html.Hr(className='section-divider'),

            html.Div(id='spread-stats-display',
                children=html.Div([
                    html.P("Returns will appear here after analysis.", style={'color': colors['muted'], 'fontStyle': 'italic'})
                ])
            )
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(dcc.Graph(id='spread-norm-chart', style={'height': '65vh', 'minHeight': '400px'}), type='circle'),
        ])
    ])
])

# --- 4. VOLATILITY SURFACE TAB LAYOUT ---
vol_surface_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Vol Surface", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Live implied volatility surface powered by ORATS.", className='helper-text', style={'marginTop': '0'}),

            html.Label("Ticker Symbol", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.Input(id='vol-ticker-input', type='text', value='SLV', placeholder="e.g. SPY, AAPL, TSLA",
                      style={**INPUT_STYLE, 'marginBottom': '12px'}),

            html.Label("Contract Type", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em', 'display': 'block'}),
            dcc.RadioItems(
                id='vol-contract-type',
                options=[
                    {'label': ' Calls',    'value': 'call'},
                    {'label': ' Puts',     'value': 'put'},
                    {'label': ' Both',     'value': 'both'},
                ],
                value='call',
                labelStyle={'display': 'inline-block', 'color': colors['text'], 'marginRight': '14px', 'cursor': 'pointer'},
                style={'marginBottom': '10px'},
            ),

            html.Label("Moneyness Range", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em', 'display': 'block'}),
            html.Div("Only show strikes within ± % of spot price", className='helper-text'),
            html.Div(style={'padding': '0 10px 16px 10px'}, children=[
                dcc.Slider(id='vol-moneyness-slider', min=10, max=40, step=5, value=25,
                           marks={10: '±10%', 20: '±20%', 30: '±30%', 40: '±40%'},
                           tooltip=None),
            ]),

            html.Label("DTE Range", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em', 'display': 'block'}),
            html.Div("Filter expirations shown on the surface", className='helper-text'),
            html.Div(style={'padding': '0 10px 16px 10px'}, children=[
                dcc.RangeSlider(id='vol-dte-range', min=1, max=365, step=1, value=[7, 180],
                                marks={7: '7d', 30: '30d', 60: '60d', 90: '90d', 180: '180d', 365: '1y'},
                                tooltip=None),
            ]),

            html.Label("Z-Axis", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em', 'display': 'block'}),
            dcc.RadioItems(
                id='vol-z-axis',
                options=[
                    {'label': ' Implied Vol (%)',   'value': 'iv'},
                    {'label': ' Option Price ($)',  'value': 'price'},
                ],
                value='iv',
                labelStyle={'display': 'inline-block', 'color': colors['text'], 'marginRight': '14px', 'cursor': 'pointer'},
                style={'marginBottom': '10px'},
            ),

            html.Label("Plot Type", style={'color': colors['text'], 'fontWeight': 'bold', 'display': 'block', 'fontSize': '0.9em'}),
            dcc.RadioItems(
                id='vol-plot-type',
                options=[
                    {'label': ' Surface (interpolated)', 'value': 'surface'},
                    {'label': ' Scatter (raw quotes)',    'value': 'scatter'},
                ],
                value='surface',
                labelStyle={'display': 'block', 'color': colors['text'], 'marginBottom': '5px', 'cursor': 'pointer'},
                style={'marginBottom': '10px'},
            ),

            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '6px', 'marginTop': '6px'}, children=[
                html.Button('Fetch (ORATS)', id='vol-submit-btn', n_clicks=0, style=BUTTON_STYLE),
            ]),
            html.Hr(className='section-divider'),
            html.Div(id='vol-info-display', children=html.Div([
                html.P("Surface data will appear here after fetch.", style={'color': colors['muted'], 'fontStyle': 'italic'})
            ]))
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Store(id='vol-data-store'),
            dcc.Tabs(id='vol-view-tabs', value='tab-surface', style={'marginBottom': '10px'}, children=[
                dcc.Tab(label='3D Surface', value='tab-surface',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666'},
                        selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"},
                        children=[
                            dcc.Loading(dcc.Graph(id='vol-surface-chart', style={'height': '65vh', 'minHeight': '450px'}), type='circle'),
                        ]),
                dcc.Tab(label='Smile Slice', value='tab-smile',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666'},
                        selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"},
                        children=[
                            html.Div(style={'padding': '10px 12px 24px 12px'}, children=[
                                html.Label("Expiry", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em', 'display': 'block', 'marginBottom': '2px'}),
                                html.Div("Fetch surface first, then drag to select an expiration.", className='helper-text', style={'marginBottom': '14px'}),
                                dcc.Slider(id='vol-smile-expiry', min=0, max=0, step=1, value=0, marks={},
                                           tooltip=None),
                            ]),
                            dcc.Graph(id='vol-smile-chart', style={'height': '58vh', 'minHeight': '400px'}),
                        ]),
            ]),
        ])
    ])
])

# --- 6. SCANNER TAB LAYOUT ---
scanner_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Stock Scanner", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Scan multiple tickers to find cheap or expensive options based on IV vs realized vol.", className='helper-text', style={'marginTop': '0'}),
            html.Label("Tickers to Scan", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Div("Comma-separated. Edit or add your own.", className='helper-text'),
            dcc.Textarea(id='scanner-tickers-input', value=DEFAULT_SCANNER_TICKERS,
                         style={**INPUT_STYLE, 'height': '80px', 'resize': 'vertical', 'fontFamily': 'monospace'}),

            html.Hr(className='section-divider'),

            html.Button('Scan (ORATS)', id='scanner-polygon-btn', n_clicks=0,
                        style={**BUTTON_STYLE, 'marginTop': '10px'}),
            html.Div(id='scanner-status', style={'color': colors['muted'], 'fontSize': '0.8em', 'fontStyle': 'italic', 'marginTop': '4px'}),

        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(html.Div(id='scanner-results-table'), type='circle'),
        ])
    ])
])

# --- CALENDAR RETURNS LAYOUT ---
cal_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Calendar Returns", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Monthly return heatmap — green positive, red negative.", className='helper-text', style={'marginTop': '0'}),
            html.Label("Ticker", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.Input(id='cal-ticker-input', type='text', placeholder='e.g. SPY', debounce=False,
                      style={**INPUT_STYLE, 'marginBottom': '14px'}),
            html.Label("Lookback Period", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Div("How many years of monthly data to show.", className='helper-text'),
            dcc.Slider(id='cal-period-slider', min=0, max=5, step=1, value=2,
                       marks={0: '1Y', 1: '2Y', 2: '3Y', 3: '5Y', 4: '10Y', 5: 'MAX'}),
            html.Button('Load Calendar', id='cal-submit-btn', n_clicks=0,
                        style={**BUTTON_STYLE, 'marginTop': '14px'}),
            html.Hr(className='section-divider'),
            html.Div(id='cal-stats-display',
                     children=html.Div("Stats will appear here after loading.",
                                       style={'color': colors['muted'], 'fontStyle': 'italic'})),
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(html.Div(id='cal-heatmap-output'), type='circle')
        ])
    ])
])

# --- 4b. VOL ANALYSIS TAB LAYOUT ---
vol_analysis_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Vol Analysis", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Historical implied & realized volatility powered by ORATS.", className='helper-text', style={'marginTop': '0'}),

            html.Label("Ticker Symbol", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.Input(id='vola-ticker-input', type='text', value='SPY', placeholder='e.g. SPY, AAPL',
                      style={**INPUT_STYLE, 'marginBottom': '12px'}),

            html.Label("Date Range", style={'color': colors['text'], 'fontWeight': 'bold', 'display': 'block', 'fontSize': '0.9em', 'marginTop': '4px', 'marginBottom': '4px'}),
            html.Div("Drag both handles to set start and end.", className='helper-text'),
            html.Div(style={'padding': '0 10px 28px 10px'}, children=[
                dcc.RangeSlider(
                    id='vola-period-slider',
                    min=0, max=11, step=1,
                    value=[6, 11],
                    marks={
                        0:  {'label': 'MAX', 'style': {'color': '#888', 'fontSize': '0.75em'}},
                        1:  {'label': '10Y',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                        2:  {'label': '7Y',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        3:  {'label': '5Y',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        4:  {'label': '3Y',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        5:  {'label': '2Y',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        6:  {'label': '1Y',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        7:  {'label': '6M',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        8:  {'label': '3M',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        9:  {'label': '1M',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        10: {'label': '2W',   'style': {'color': '#888', 'fontSize': '0.75em'}},
                        11: {'label': 'Now',  'style': {'color': '#888', 'fontSize': '0.75em'}},
                    },
                    tooltip=None,
                )
            ]),

            html.Button('Analyze', id='vola-analyze-btn', n_clicks=0, style=BUTTON_STYLE),
            html.Hr(className='section-divider'),
            html.Div(id='vola-stats-display', children=html.Div([
                html.P("Stats will appear here after analysis.", style={'color': colors['muted'], 'fontStyle': 'italic'})
            ]))
        ]),

        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(dcc.Graph(id='vola-chart', style={'height': '75vh', 'minHeight': '500px'}), type='circle'),
        ])
    ])
])

# --- APP LAYOUT ---
# CHANGE: Revamped header with subtitle, added footer
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '10px 20px 0 20px', 'fontFamily': "'Segoe UI', Arial, sans-serif"}, children=[

    # CHANGE: Revamped header with subtitle and visual separator
    html.Div(style={'textAlign': 'center', 'padding': '16px 0 8px 0'}, children=[
        html.H1("Equity Research Dashboard", style={
            'color': colors['text'], 'fontSize': '1.6rem', 'margin': '0', 'fontWeight': '700', 'letterSpacing': '0.5px'
        }),
        # CHANGE: Added subtitle so users immediately understand the app's purpose
        html.P("Options pricing, volatility analysis & relative value tools", style={
            'color': colors['muted'], 'fontSize': '0.85em', 'margin': '6px 0 0 0'
        }),
    ]),

    # Navigation Tabs
    dcc.Tabs(id='main-tabs', value='tab-scanner',
             style={'marginTop': '16px', 'marginBottom': '20px'},
             children=[
                dcc.Tab(label='Scanner', value='tab-scanner',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Spread Analysis', value='tab-spread',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Calendar', value='tab-cal',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Vol Surface', value='tab-vol',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Vol Analysis', value='tab-vola',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
    ]),

    html.Div(id='spread-content-wrapper', children=spread_layout, style={'display': 'none'}),
    html.Div(id='cal-content-wrapper', children=cal_layout, style={'display': 'none'}),
    html.Div(id='vol-content-wrapper', children=vol_surface_layout, style={'display': 'none'}),
    html.Div(id='scanner-content-wrapper', children=scanner_layout, style={'display': 'block'}),
    html.Div(id='vola-content-wrapper', children=vol_analysis_layout, style={'display': 'none'}),

    # CHANGE: Added footer with context so the app feels polished
    html.Div(style={
        'textAlign': 'center', 'padding': '20px 0', 'marginTop': '30px',
        'borderTop': f"1px solid {colors['card_border']}",
    }, children=[
        html.P("Market data provided by Yahoo Finance. Options priced using the Black-Scholes model.",
               style={'color': colors['muted'], 'fontSize': '0.75em', 'margin': '0'}),
        html.P("For educational purposes only. Not financial advice.",
               style={'color': '#666', 'fontSize': '0.7em', 'margin': '4px 0 0 0'})
    ])
])

# -----------------------------------------------------------------------------
# 5. CALLBACKS
# -----------------------------------------------------------------------------

# Tab Visibility Toggle
@app.callback(
    [Output('spread-content-wrapper', 'style'), Output('cal-content-wrapper', 'style'),
     Output('vol-content-wrapper', 'style'), Output('scanner-content-wrapper', 'style'),
     Output('vola-content-wrapper', 'style')],
    [Input('main-tabs', 'value')]
)
def toggle_tabs(tab_value):
    spread_style, cal_style, vol_style, scanner_style, vola_style = [{'display': 'none'}] * 5
    if tab_value == 'tab-spread':   spread_style  = {'display': 'block'}
    elif tab_value == 'tab-cal':    cal_style     = {'display': 'block'}
    elif tab_value == 'tab-vol':    vol_style     = {'display': 'block'}
    elif tab_value == 'tab-scanner': scanner_style = {'display': 'block'}
    elif tab_value == 'tab-vola':   vola_style    = {'display': 'block'}
    return spread_style, cal_style, vol_style, scanner_style, vola_style

# --- SPREAD ANALYSIS CALLBACK ---
@app.callback(
    [Output('spread-norm-chart', 'figure'), Output('spread-stats-display', 'children')],
    [Input('spread-analyze-btn', 'n_clicks')],
    [State('spread-tickers-input', 'value'), State('spread-period-slider', 'value')]
)
def update_spread_analysis(n_clicks, tickers_raw, slider_val):
    if not tickers_raw:
        return go.Figure(layout=layout_settings), html.Div()

    tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
    if not tickers:
        return go.Figure(layout=layout_settings), html.Div()

    # Map slider index → date offset from today
    today = datetime.date.today()
    _date_map = {
        0:  None,                                       # MAX
        1:  today - datetime.timedelta(days=365 * 20),
        2:  today - datetime.timedelta(days=365 * 15),
        3:  today - datetime.timedelta(days=365 * 12),
        4:  today - datetime.timedelta(days=365 * 10),
        5:  today - datetime.timedelta(days=365 * 7),
        6:  today - datetime.timedelta(days=365 * 5),
        7:  today - datetime.timedelta(days=365 * 3),
        8:  today - datetime.timedelta(days=365 * 2),
        9:  today - datetime.timedelta(days=365),
        10: today - datetime.timedelta(days=182),
        11: today - datetime.timedelta(days=91),
        12: today,
    }
    start_idx, end_idx = (slider_val if isinstance(slider_val, list) else [slider_val, 12])
    start_date = _date_map.get(start_idx)
    end_date   = _date_map.get(end_idx, today)

    selected_period = (
        f"{start_date.isoformat()} → {end_date.isoformat()}" if start_date
        else f"MAX → {end_date.isoformat()}"
    )

    try:
        hist_kwargs = ({'start': start_date.isoformat(), 'end': end_date.isoformat()}
                       if start_date else {'period': 'max'})
        series = {}
        for t in tickers:
            s = yf.Ticker(t).history(**hist_kwargs)['Close']
            if not s.empty:
                series[t] = s

        if not series:
            return go.Figure(layout=layout_settings), html.Div("No data found for any ticker.", style={'color': colors['danger']})

        df = pd.DataFrame(series).ffill().dropna()
        if df.empty:
            return go.Figure(layout=layout_settings), html.Div("No overlapping dates across tickers.", style={'color': colors['danger']})

        fig_norm = go.Figure()
        for i, col in enumerate(df.columns):
            color = SPREAD_LINE_COLORS[i % len(SPREAD_LINE_COLORS)]
            norm = (df[col] / df[col].iloc[0]) * 100
            fig_norm.add_trace(go.Scatter(x=df.index, y=norm, mode='lines', name=col,
                                          line=dict(color=color, width=2)))

        fig_norm.update_layout(
            title=f"Normalized Performance — {selected_period.upper()}",
            yaxis_title="Normalized Price (100 = Start)",
            margin=dict(l=20, r=20, t=40, b=20),
            uirevision=f"{'-'.join(df.columns)}-{selected_period}",
            transition=dict(duration=200, easing='cubic-in-out'),
            **layout_settings,
        )

        daily_ret = df.pct_change().dropna()

        stat_rows = [
            html.H4("Returns", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),
            make_stat_row("Period", selected_period.upper()),
            html.Hr(className='section-divider'),
        ]
        for col in df.columns:
            ret = (df[col].iloc[-1] / df[col].iloc[0] - 1) * 100
            ret_color = colors['call_text'] if ret >= 0 else colors['put_text']
            stat_rows.append(make_stat_row(col, f"{ret:+.1f}%", ret_color))

        stat_rows += [
            html.Hr(className='section-divider'),
            html.H4("Sharpe Ratio", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),
        ]
        for col in df.columns:
            std = daily_ret[col].std()
            sharpe = (daily_ret[col].mean() / std * np.sqrt(252)) if std > 0 else 0
            sharpe_color = (colors['call_text'] if sharpe > 1
                            else colors['put_text'] if sharpe < 0
                            else colors['text'])
            stat_rows.append(make_stat_row(col, f"{sharpe:.2f}", sharpe_color))

        if len(df.columns) >= 2:
            corr = df.pct_change().corr()
            pairs = [(a, b) for i, a in enumerate(df.columns) for b in list(df.columns)[i+1:]]
            stat_rows += [
                html.Hr(className='section-divider'),
                html.H4("Correlation", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),
            ]
            for a, b in pairs:
                val = corr.loc[a, b]
                cor_color = (colors['call_text'] if val > 0.7
                             else colors['put_text'] if val < 0.3
                             else colors['text'])
                stat_rows.append(make_stat_row(f"{a}/{b}", f"{val:.2f}", cor_color))

        return fig_norm, html.Div(stat_rows)

    except Exception as e:
        return go.Figure(layout=layout_settings), html.Div([
            html.Div("Could not fetch data", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(f"{e}", style={'color': colors['muted'], 'fontSize': '0.85em'})
        ])


# Results are cached per (ticker, date, contract_type, moneyness) so re-visiting
# a previously fetched date is instant with zero additional API calls.
def _build_surface_figure(data, sym, contract_type, z_axis, plot_type, moneyness_pct, dte_range=None):
    """Render a surface/scatter figure from a cached data dict. Returns (fig, info_html, smile_marks, smile_max)."""
    dte_min = dte_range[0] if dte_range else 1
    dte_max = dte_range[1] if dte_range else 9999

    mask       = [dte_min <= d <= dte_max for d in data['dtes']]
    strikes    = [v for v, m in zip(data['strikes'],    mask) if m]
    dtes       = [v for v, m in zip(data['dtes'],       mask) if m]
    ivs        = [v for v, m in zip(data['ivs'],        mask) if m]
    raw_prices = [v for v, m in zip(data['prices'],     mask) if m]
    spot       = data['spot']

    if len(strikes) < 5:
        return None, "not_enough_price", None, None

    min_strike, max_strike = min(strikes), max(strikes)
    min_dte,    max_dte    = min(dtes),    max(dtes)

    use_price = (z_axis == 'price')
    if use_price:
        valid = [(s, d, p) for s, d, p in zip(strikes, dtes, raw_prices) if p is not None]
        if len(valid) < 5:
            return None, "not_enough_price", None, None
        z_strikes, z_dtes, z_vals = zip(*valid)
        z_strikes, z_dtes, z_vals = list(z_strikes), list(z_dtes), list(z_vals)
        z_label, z_tickprefix, z_ticksuffix, hover_z = "Price ($)", "$", "", "$%{z:.2f}"
    else:
        z_strikes, z_dtes = strikes, dtes
        z_vals = [v * 100 for v in ivs]
        z_label, z_tickprefix, z_ticksuffix, hover_z = "IV (%)", "", "%", "%{z:.1f}%"

    ct_label   = {'call': 'Call', 'put': 'Put', 'both': 'Call + Put'}.get(contract_type, '')
    z_display  = "Option Price" if use_price else "Implied Vol"
    spot_label = f"  Spot: ${spot:.2f}" if spot else ""

    # Build DTE → expiry date tick mapping; only label the first expiry per month
    dte_to_exp = {}
    for d, e in zip(data['dtes'], data['exps']):
        dte_to_exp[d] = e
    sorted_dte_exp = sorted(dte_to_exp.items())
    y_tickvals, y_ticktext = [], []
    seen_months = set()
    for d, e in sorted_dte_exp:
        try:
            dt = datetime.date.fromisoformat(e)
            month_key = (dt.year, dt.month)
            label = dt.strftime("%b %d") if month_key not in seen_months else ""
            seen_months.add(month_key)
        except:
            label = e
        y_tickvals.append(d)
        y_ticktext.append(label)

    fig = go.Figure()
    if plot_type == 'surface':
        sg = np.linspace(min(z_strikes), max(z_strikes), 50)
        dg = np.linspace(min(z_dtes),    max(z_dtes),    50)
        X, Y = np.meshgrid(sg, dg)
        Z = griddata((z_strikes, z_dtes), z_vals, (X, Y), method='cubic')
        if np.isnan(Z).all():
            Z = griddata((z_strikes, z_dtes), z_vals, (X, Y), method='linear')
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y, colorscale='Jet',
            colorbar=dict(title=z_label, tickprefix=z_tickprefix, ticksuffix=z_ticksuffix),
            hovertemplate=f"Strike: $%{{x:.0f}}<br>DTE: %{{y:.0f}}d<br>{z_display}: {hover_z}<extra></extra>",
        ))
        if spot:
            si = int(np.argmin(np.abs(np.array(sg) - spot)))
            z_col = Z[:, si]
            mask = ~np.isnan(z_col)
            if mask.any():
                fig.add_trace(go.Scatter3d(
                    x=np.full(mask.sum(), sg[si]),
                    y=dg[mask],
                    z=z_col[mask],
                    mode='lines',
                    name=f'Spot ${spot:.2f}',
                    line=dict(color='white', width=5),
                    showlegend=True,
                ))
    else:
        fig.add_trace(go.Scatter3d(
            x=z_strikes, y=z_dtes, z=z_vals, mode='markers',
            marker=dict(size=3, color=z_vals, colorscale='Jet', opacity=0.85,
                        colorbar=dict(title=z_label, tickprefix=z_tickprefix, ticksuffix=z_ticksuffix)),
            hovertemplate=f"Strike: $%{{x:.0f}}<br>DTE: %{{y:.0f}}d<br>{z_display}: {hover_z}<extra></extra>",
            name=z_display,
        ))
        if spot:
            fig.add_trace(go.Scatter3d(
                x=[spot, spot],
                y=[min(z_dtes), min(z_dtes)],
                z=[min(z_vals), max(z_vals)],
                mode='lines',
                name=f'Spot ${spot:.2f}',
                line=dict(color='white', width=4),
                showlegend=True,
            ))

    today_str = datetime.date.today().isoformat()
    fig.update_layout(
        title=f"{sym} {ct_label} {z_display} Surface — Live{spot_label}",
        scene=dict(
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=0), eye=dict(x=-1.8,y=-1.2,z=1.0)),
            xaxis_title='Strike ($)', zaxis_title=z_label,
            xaxis=dict(backgroundcolor=colors['card_bg'], gridcolor='#333', showbackground=True),
            yaxis=dict(backgroundcolor=colors['card_bg'], gridcolor='#333', showbackground=True,
                       title='Expiry', tickvals=y_tickvals, ticktext=y_ticktext),
            zaxis=dict(backgroundcolor=colors['card_bg'], gridcolor='#333', showbackground=True),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision=sym,
        **layout_settings
    )

    iv_pcts    = [v * 100 for v in ivs]
    price_vals = [p for p in raw_prices if p is not None]
    info_html = html.Div([
        html.H4("Surface Data · ORATS", style={'color': colors['text'], 'marginBottom': '10px', 'fontSize': '1em'}),
        make_stat_row("Source",       "ORATS — Live"),
        make_stat_row("Date",         today_str),
        make_stat_row("Spot Price",   f"${spot:.2f}" if spot else "N/A", colors['accent']),
        make_stat_row("Contracts",    f"{data['contract_count']:,}"),
        make_stat_row("Expirations",  f"{data['exp_count']}"),
        make_stat_row("Strike Range", f"${min_strike:.0f} – ${max_strike:.0f}"),
        make_stat_row("DTE Range",    f"{min_dte} – {max_dte} days"),
        make_stat_row("IV Range",     f"{min(iv_pcts):.1f}% – {max(iv_pcts):.1f}%"),
        *([ make_stat_row("Price Range", f"${min(price_vals):.2f} – ${max(price_vals):.2f}") ] if price_vals else []),
        make_stat_row("Moneyness",    f"±{int(moneyness_pct*100)}% of spot"),
    ])

    sorted_exps = sorted(set(data['exps']))
    data['sorted_exps'] = sorted_exps

    def fmt_exp(e):
        try:    return datetime.date.fromisoformat(e).strftime("%b %d")
        except: return e

    smile_marks = {i: {'label': fmt_exp(e), 'style': {'color': '#aaa', 'fontSize': '10px'}}
                   for i, e in enumerate(sorted_exps)}
    return fig, info_html, smile_marks, max(0, len(sorted_exps) - 1)


@app.callback(
    [Output('vol-surface-chart', 'figure'), Output('vol-info-display', 'children'),
     Output('vol-data-store', 'data'), Output('vol-smile-expiry', 'marks'),
     Output('vol-smile-expiry', 'max'), Output('vol-smile-expiry', 'value')],
    [Input('vol-submit-btn', 'n_clicks'),
     Input('vol-plot-type', 'value'),
     Input('vol-z-axis', 'value'),
     Input('vol-dte-range', 'value')],
    [State('vol-ticker-input', 'value'),
     State('vol-contract-type', 'value'),
     State('vol-moneyness-slider', 'value'),
     State('vol-data-store', 'data')],
    prevent_initial_call=True
)
def update_vol_surface(_n_poly, plot_type, z_axis, dte_range, ticker_symbol, contract_type, moneyness_slider, stored_data):
    empty_fig = go.Figure(layout=layout_settings)
    triggered  = ctx.triggered_id

    # --- Re-render from cache (no fetch) when only display toggles changed ---
    if triggered in ('vol-plot-type', 'vol-z-axis', 'vol-dte-range'):
        if not stored_data:
            return no_update, no_update, no_update, no_update, no_update, no_update
        sym           = (ticker_symbol or '').upper().strip()
        moneyness_pct = (moneyness_slider or 25) / 100
        fig, _, _, _ = _build_surface_figure(stored_data, sym, contract_type, z_axis, plot_type, moneyness_pct, dte_range)
        if fig is None:
            return no_update, no_update, no_update, no_update, no_update, no_update
        return fig, no_update, no_update, no_update, no_update, no_update

    # --- Fetch fresh data ---
    if not ticker_symbol:
        return empty_fig, html.Div(), no_update, no_update, no_update, no_update

    sym           = ticker_symbol.upper().strip()
    moneyness_pct = (moneyness_slider or 25) / 100

    data, err    = fetch_orats_surface(sym, contract_type, moneyness_pct)
    source_label = "ORATS"

    if err:
        info = html.Div([
            html.Div(f"{source_label} fetch failed", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(err, style={'color': colors['muted'], 'fontSize': '0.85em'}),
        ])
        return empty_fig, info, no_update, no_update, no_update, no_update

    fig, info_html, smile_marks, smile_max = _build_surface_figure(
        data, sym, contract_type, z_axis, plot_type, moneyness_pct, dte_range)
    if fig is None:
        return empty_fig, html.Div("Not enough price quotes. Switch to IV (%).",
                                   style={'color': colors['danger']}), \
               no_update, no_update, no_update, no_update
    return fig, info_html, data, smile_marks, smile_max, 0


# --- VOL SMILE SLICE CALLBACK ---
@app.callback(
    Output('vol-smile-chart', 'figure'),
    [Input('vol-smile-expiry', 'drag_value'), Input('vol-smile-expiry', 'value'), Input('vol-data-store', 'data')],
    State('vol-z-axis', 'value'),
)
def update_vol_smile(drag_idx, value_idx, store_data, z_axis):
    selected_idx = drag_idx if drag_idx is not None else value_idx
    empty_fig = go.Figure(layout=layout_settings)
    if not store_data or selected_idx is None or 'sorted_exps' not in store_data:
        empty_fig.update_layout(annotations=[dict(
            text="Fetch the surface first, then drag the expiry slider.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=colors['muted'])
        )])
        return empty_fig

    sorted_exps = store_data['sorted_exps']
    if selected_idx >= len(sorted_exps):
        return empty_fig
    selected_exp = sorted_exps[selected_idx]

    use_price = (z_axis == 'price')
    y_label   = "Option Price ($)" if use_price else "Implied Volatility (%)"
    y_prefix  = "$" if use_price else ""
    y_suffix  = ""  if use_price else "%"

    spot    = store_data.get('spot')
    strikes = store_data['strikes']
    ivs     = store_data['ivs']
    prices  = store_data['prices']
    exps    = store_data['exps']
    ctypes  = store_data['ctypes']

    # Compute global axis ranges across ALL expiries so axes stay fixed while scrubbing
    all_y = []
    for iv, p in zip(ivs, prices):
        v = p if use_price else iv * 100
        if v is not None:
            all_y.append(v)

    x_pad = (max(strikes) - min(strikes)) * 0.04 if strikes else 1
    y_pad = (max(all_y)   - min(all_y))   * 0.06 if all_y   else 1
    x_range = [min(strikes) - x_pad, max(strikes) + x_pad]
    y_range = [max(0, min(all_y) - y_pad), max(all_y) + y_pad]

    # Filter to selected expiry
    call_strikes, call_y = [], []
    put_strikes,  put_y  = [], []

    for s, iv, p, exp, ct in zip(strikes, ivs, prices, exps, ctypes):
        if exp != selected_exp:
            continue
        y_val = p if use_price else iv * 100
        if y_val is None:
            continue
        if ct == 'call':
            call_strikes.append(s); call_y.append(y_val)
        elif ct == 'put':
            put_strikes.append(s);  put_y.append(y_val)
        else:
            call_strikes.append(s); call_y.append(y_val)

    fig = go.Figure()
    if call_strikes:
        pairs = sorted(zip(call_strikes, call_y))
        sc, yc = zip(*pairs)
        fig.add_trace(go.Scatter(x=list(sc), y=list(yc), mode='lines+markers',
                                 name='Calls', line=dict(color=colors['call_text'], width=2),
                                 marker=dict(size=5)))
    if put_strikes:
        pairs = sorted(zip(put_strikes, put_y))
        sp, yp = zip(*pairs)
        fig.add_trace(go.Scatter(x=list(sp), y=list(yp), mode='lines+markers',
                                 name='Puts', line=dict(color=colors['put_text'], width=2),
                                 marker=dict(size=5)))

    if spot:
        fig.add_vline(x=spot, line_width=1, line_dash='dash', line_color='#666',
                      annotation_text=f'Spot ${spot:.2f}', annotation_font_color='#aaa')

    fig.update_layout(
        title=f"Volatility Smile — Expiry {selected_exp}",
        xaxis=dict(title="Strike ($)", range=x_range),
        yaxis=dict(title=y_label, range=y_range, tickprefix=y_prefix, ticksuffix=y_suffix),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=60, b=40),
        uirevision='smile',
        transition=dict(duration=100, easing='cubic-in-out'),
        **layout_settings,
    )
    return fig


# --- CALENDAR RETURNS CALLBACK ---
@app.callback(
    [Output('cal-heatmap-output', 'children'), Output('cal-stats-display', 'children')],
    [Input('cal-submit-btn', 'n_clicks')],
    [State('cal-ticker-input', 'value'), State('cal-period-slider', 'value')],
    prevent_initial_call=True
)
def run_calendar(_n_clicks, ticker_raw, slider_val):
    if not ticker_raw:
        return html.Div("Enter a ticker and click Load.", style={'color': colors['muted'], 'fontStyle': 'italic', 'padding': '20px'}), html.Div()

    ticker_sym = ticker_raw.strip().upper()
    period = CAL_PERIOD_MAP.get(slider_val, '5y')

    try:
        hist = yf.Ticker(ticker_sym).history(period=period)
        if hist.empty:
            return html.Div("No data found.", style={'color': colors['danger']}), html.Div()

        monthly = hist['Close'].resample('ME').last()
        monthly_ret = monthly.pct_change().dropna() * 100

        df_cal = pd.DataFrame({
            'Year': monthly_ret.index.year,
            'Month': monthly_ret.index.month,
            'Return': monthly_ret.values
        })
        pivot = df_cal.pivot(index='Year', columns='Month', values='Return')
        pivot = pivot.sort_index(ascending=False)

        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        col_labels = [month_labels[m - 1] for m in pivot.columns]

        text_matrix = []
        for yr in pivot.index:
            row_text = []
            for m in pivot.columns:
                val = pivot.loc[yr, m]
                row_text.append(f"{val:+.1f}%" if not pd.isna(val) else "")
            text_matrix.append(row_text)

        fig = go.Figure(go.Heatmap(
            z=pivot.values.tolist(),
            x=col_labels,
            y=[str(y) for y in pivot.index],
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, color='black'),
            colorscale='RdYlGn',
            zmid=0,
            showscale=True,
            colorbar=dict(
                title=dict(text='Return %', font=dict(color=colors['muted'])),
                tickfont=dict(color=colors['muted']),
                bgcolor=colors['card_bg'],
            ),
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=f"{ticker_sym} — Monthly Returns ({period.upper()})", font=dict(color=colors['text'])),
            paper_bgcolor=colors['card_bg'],
            plot_bgcolor=colors['card_bg'],
            font=dict(color=colors['text']),
            xaxis=dict(side='top', tickfont=dict(color=colors['text']), gridcolor=colors['card_border']),
            yaxis=dict(tickfont=dict(color=colors['text']), gridcolor=colors['card_border']),
            margin=dict(l=60, r=20, t=80, b=20),
            height=max(350, len(pivot.index) * 42 + 140),
        )

        annual = monthly_ret.groupby(monthly_ret.index.year).apply(
            lambda x: ((1 + x / 100).prod() - 1) * 100
        )
        pos_months = int((monthly_ret > 0).sum())
        total_months = len(monthly_ret)

        stat_rows = [
            html.H4("Summary", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),
            make_stat_row("Ticker", ticker_sym),
            make_stat_row("Period", period.upper()),
            make_stat_row("Hit Rate", f"{pos_months}/{total_months} ({pos_months / total_months * 100:.0f}%)"),
            make_stat_row("Best Month", f"{monthly_ret.max():+.1f}%", colors['call_text']),
            make_stat_row("Worst Month", f"{monthly_ret.min():+.1f}%", colors['put_text']),
            html.Hr(className='section-divider'),
            html.H4("Annual Returns", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),
        ]
        for yr in sorted(annual.index, reverse=True):
            ret = annual[yr]
            ret_color = colors['call_text'] if ret >= 0 else colors['put_text']
            stat_rows.append(make_stat_row(str(yr), f"{ret:+.1f}%", ret_color))

        return dcc.Graph(figure=fig, config={'displayModeBar': False}), html.Div(stat_rows)

    except Exception as e:
        return html.Div(f"Error: {e}", style={'color': colors['danger']}), html.Div()


# --- SCANNER CALLBACK ---

@app.callback(
    [Output('scanner-results-table', 'children'), Output('scanner-status', 'children')],
    Input('scanner-polygon-btn', 'n_clicks'),
    State('scanner-tickers-input', 'value'),
    prevent_initial_call=True
)
def run_scanner(_n, tickers_raw):
    if not tickers_raw:
        return html.Div("Enter tickers and click Scan.", style={'color': colors['muted'], 'fontStyle': 'italic', 'padding': '20px'}), ""

    tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
    results = [scan_ticker_orats(t) for t in tickers]
    source_label = 'ORATS'

    errors = [r for r in results if r and '_error' in r]
    rows   = [r for r in results if r and '_error' not in r]

    if not rows:
        err_msg = errors[0]['_error'] if errors else "all tickers returned no data"
        return html.Div([
            html.Div("Scanner failed — no data returned.", style={'color': colors['danger'], 'fontWeight': 'bold', 'padding': '20px 20px 4px'}),
            html.Div(f"First error: {err_msg}", style={'color': colors['muted'], 'padding': '0 20px 20px', 'fontSize': '0.85em'}),
        ]), ""

    col_order = ['Ticker', 'Price', 'IV %', 'HV %', 'IV/HV', 'Skew', 'P/C Vol', 'HV Rank', 'Bias', 'Call Rec', 'Put Rec']

    # Per-row heatmap styles for IV/HV and Bias
    bias_t = {'Bullish': 1.0, 'Neutral': 0.5, 'Bearish': 0.0}
    cell_styles = []
    for i, row in enumerate(rows):
        ivhv_str = row.get('IV/HV', 'N/A')
        if ivhv_str != 'N/A':
            try:
                val = float(ivhv_str.replace('x', ''))
                t = 1.0 - max(0.0, min(1.0, (val - 0.5) / 1.0))
                bg = pc.sample_colorscale('RdYlGn', [t])[0]
                cell_styles.append({'if': {'row_index': i, 'column_id': 'IV/HV'}, 'backgroundColor': bg, 'color': '#111'})
            except Exception:
                pass
        skew_str = row.get('Skew', 'N/A')
        if skew_str != 'N/A':
            try:
                val = float(skew_str.replace('x', ''))
                t = 1.0 - max(0.0, min(1.0, (val - 0.75) / 0.5))
                bg = pc.sample_colorscale('RdYlGn', [t])[0]
                cell_styles.append({'if': {'row_index': i, 'column_id': 'Skew'}, 'backgroundColor': bg, 'color': '#111'})
            except Exception:
                pass
        pc_str = row.get('P/C Vol', 'N/A')
        if pc_str != 'N/A':
            try:
                val = float(pc_str)
                t = 1.0 - max(0.0, min(1.0, (val - 0.5) / 1.0))
                bg = pc.sample_colorscale('RdYlGn', [t])[0]
                cell_styles.append({'if': {'row_index': i, 'column_id': 'P/C Vol'}, 'backgroundColor': bg, 'color': '#111'})
            except Exception:
                pass
        t_bias = bias_t.get(row.get('Bias', 'Neutral'), 0.5)
        bg_bias = pc.sample_colorscale('RdYlGn', [t_bias])[0]
        cell_styles.append({'if': {'row_index': i, 'column_id': 'Bias'}, 'backgroundColor': bg_bias, 'color': '#111', 'fontWeight': 'bold'})

    table = dash_table.DataTable(
        data=rows,
        columns=[{'name': c, 'id': c} for c in col_order],
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': '#1a1a1a', 'color': colors['accent'],
            'fontWeight': 'bold', 'border': f"1px solid {colors['card_border']}", 'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': colors['card_bg'], 'color': colors['text'],
            'border': f"1px solid {colors['card_border']}", 'textAlign': 'center',
            'padding': '10px', 'fontFamily': "'Segoe UI', Arial, sans-serif"
        },
        style_data_conditional=[
            # Call Rec coloring
            {'if': {'filter_query': '{Call Rec} = "Buy"',  'column_id': 'Call Rec'}, 'color': colors['call_text'], 'fontWeight': 'bold'},
            {'if': {'filter_query': '{Call Rec} = "Sell"', 'column_id': 'Call Rec'}, 'color': colors['put_text'],  'fontWeight': 'bold'},
            {'if': {'filter_query': '{Call Rec} = "Hold"', 'column_id': 'Call Rec'}, 'color': colors['muted']},
            # Put Rec coloring
            {'if': {'filter_query': '{Put Rec} = "Buy"',   'column_id': 'Put Rec'}, 'color': colors['call_text'], 'fontWeight': 'bold'},
            {'if': {'filter_query': '{Put Rec} = "Sell"',  'column_id': 'Put Rec'}, 'color': colors['put_text'],  'fontWeight': 'bold'},
            {'if': {'filter_query': '{Put Rec} = "Hold"',  'column_id': 'Put Rec'}, 'color': colors['muted']},
            # Ticker highlight
            {'if': {'column_id': 'Ticker'}, 'fontWeight': 'bold', 'color': colors['accent']},
            *cell_styles,
        ],
        sort_action='native',
        tooltip_header={
            'Skew':    'OTM 90% put IV ÷ ATM call IV. >1.15 = bearish skew, <0.88 = bullish',
            'P/C Vol': 'Total put volume ÷ total call volume. >1.10 = bearish, <0.75 = bullish',
            'Bias':    'Directional vote from Skew + P/C Vol. Both must agree for Bullish/Bearish.',
            'Call Rec':'Buy = bullish bias + cheap IV. Sell = bearish bias + expensive IV.',
            'Put Rec': 'Buy = bearish bias + cheap IV. Sell = bullish bias + expensive IV.',
        },
        tooltip_delay=0,
        tooltip_duration=None,
    )
    now_et   = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=4)
    ts       = now_et.strftime('%H:%M:%S')
    status   = f"Scanned {len(rows)}/{len(tickers)} via {source_label} · {ts} ET"
    return table, status



# --- VOL ANALYSIS CALLBACK ---
@app.callback(
    [Output('vola-chart', 'figure'), Output('vola-stats-display', 'children')],
    [Input('vola-analyze-btn', 'n_clicks')],
    [State('vola-ticker-input', 'value'), State('vola-period-slider', 'value')],
    prevent_initial_call=True
)
def update_vol_analysis(_n, ticker_symbol, slider_val):
    empty_fig = go.Figure(layout=layout_settings)
    if not ticker_symbol:
        return empty_fig, html.Div()

    sym   = ticker_symbol.strip().upper()
    today = datetime.date.today()

    _vola_date_map = {
        0:  None,
        1:  today - datetime.timedelta(days=365 * 10),
        2:  today - datetime.timedelta(days=365 * 7),
        3:  today - datetime.timedelta(days=365 * 5),
        4:  today - datetime.timedelta(days=365 * 3),
        5:  today - datetime.timedelta(days=365 * 2),
        6:  today - datetime.timedelta(days=365),
        7:  today - datetime.timedelta(days=182),
        8:  today - datetime.timedelta(days=91),
        9:  today - datetime.timedelta(days=30),
        10: today - datetime.timedelta(days=14),
        11: today,
    }

    start_idx, end_idx = (slider_val if isinstance(slider_val, list) else [6, 11])
    start_date = _vola_date_map.get(start_idx)
    end_date   = _vola_date_map.get(end_idx, today)

    # Fetch ORATS historical IV and filter by date range in Python
    df, err = fetch_orats_hist_iv(sym)
    if err:
        return empty_fig, html.Div(err, style={'color': colors['danger'], 'padding': '10px'})
    if start_date:
        df = df[df['tradeDate'] >= pd.Timestamp(start_date)]
    if end_date and end_date != today:
        df = df[df['tradeDate'] <= pd.Timestamp(end_date)]
    df = df.reset_index(drop=True)

    # Fetch stock price history from yfinance
    hist_kwargs = ({'start': start_date.isoformat(), 'end': (end_date + datetime.timedelta(days=1)).isoformat()}
                   if start_date else {'period': 'max'})
    price_series = yf.Ticker(sym).history(**hist_kwargs)['Close']

    # Build 2-row subplot: price on top, IV + HV below
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(f'{sym} Price', 'Implied & Realized Volatility (30d)'),
        row_heights=[0.4, 0.6],
    )

    # Row 1: stock price
    if not price_series.empty:
        fig.add_trace(go.Scatter(
            x=price_series.index, y=price_series.values,
            mode='lines', name='Price',
            line=dict(color='#ffffff', width=1.5),
            hovertemplate='$%{y:.2f}<extra>Price</extra>',
        ), row=1, col=1)

    # Row 2: IV30d and RVol30
    if 'iv30d' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['tradeDate'], y=df['iv30d'] * 100,
            mode='lines', name='IV 30d',
            line=dict(color=colors['accent'], width=1.5),
            hovertemplate='%{y:.1f}%<extra>IV 30d</extra>',
        ), row=2, col=1)

    if 'rVol30' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['tradeDate'], y=df['rVol30'] * 100,
            mode='lines', name='HV 30d',
            line=dict(color='#00c8ff', width=1.5, dash='dot'),
            hovertemplate='%{y:.1f}%<extra>HV 30d</extra>',
        ), row=2, col=1)

    fig.update_layout(
        **layout_settings,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text='Price ($)', tickprefix='$', row=1, col=1)
    fig.update_yaxes(title_text='Vol (%)', ticksuffix='%', row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='#222', row=2, col=1)

    # Stats panel
    iv_latest  = df['iv30d'].iloc[-1]  * 100 if 'iv30d'  in df.columns and len(df) else None
    hv_latest  = df['rVol30'].iloc[-1] * 100 if 'rVol30' in df.columns and len(df) else None
    iv_max     = df['iv30d'].max()     * 100 if 'iv30d'  in df.columns else None
    iv_min     = df['iv30d'].min()     * 100 if 'iv30d'  in df.columns else None
    iv_rank    = ((iv_latest - iv_min) / (iv_max - iv_min) * 100
                  if iv_latest and iv_max and iv_min and iv_max != iv_min else None)
    vrp        = (iv_latest / hv_latest if iv_latest and hv_latest else None)

    stats = html.Div([
        html.H4(f"{sym} Vol Summary", style={'color': colors['text'], 'marginBottom': '10px', 'fontSize': '1em'}),
        make_stat_row("IV 30d (latest)",  f"{iv_latest:.1f}%"        if iv_latest  else "N/A"),
        make_stat_row("HV 30d (latest)",  f"{hv_latest:.1f}%"        if hv_latest  else "N/A"),
        make_stat_row("IV/HV (VRP)",      f"{vrp:.2f}x"              if vrp        else "N/A"),
        make_stat_row("IV Rank (period)", f"{iv_rank:.0f}%"          if iv_rank    else "N/A"),
        make_stat_row("IV Range",         f"{iv_min:.1f}–{iv_max:.1f}%" if iv_min and iv_max else "N/A"),
        make_stat_row("Data points",      str(len(df))),
    ])

    return fig, stats


if __name__ == '__main__':
    app.run(debug=True)
