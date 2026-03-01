import os
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm
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
DEFAULT_SCANNER_TICKERS = "SPY, AAPL, TSLA, NVDA, AMD, MSFT, AMZN, META, GOOGL, GLD, SLV, TLT"
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'qvG5Nf6OFdw8Od7oMVeUo7B0q3lB0zbo')

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

def black_scholes(S, K, T, r, sigma, option_type='call'):
    try:
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    except: return 0.0
    if T <= 0 or sigma <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    try:
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    except: return 0, 0, 0, 0, 0
    if T <= 0 or sigma <= 0: return 0, 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d2 = norm.cdf(-d2)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100
    if option_type == 'call':
        delta = cdf_d1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2) / 365
    else:
        delta = cdf_d1 - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * cdf_neg_d2) / 365
    return delta, gamma, theta, vega

def get_bs_charts(S, K, T, r, sigma):
    lower_bound = max(0.01, S * 0.5)
    spot_range = np.linspace(lower_bound, S * 1.5, 100)
    call_prices = [black_scholes(s, K, T, r, sigma, 'call') for s in spot_range]
    put_prices = [black_scholes(s, K, T, r, sigma, 'put') for s in spot_range]

    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(x=spot_range, y=call_prices, mode='lines', name='Call Value', line=dict(color=colors['call_text'])))
    fig_spot.add_trace(go.Scatter(x=spot_range, y=put_prices, mode='lines', name='Put Value', line=dict(color=colors['put_text'])))
    fig_spot.add_vline(x=S, line_width=1, line_dash="dash", line_color="#666", annotation_text="Spot")

    fig_spot.update_layout(title='Option Value vs. Spot Price', xaxis_title='Spot Price ($)', yaxis_title='Value ($)',
                           margin=dict(l=20, r=20, t=40, b=20), uirevision='bs-spot',
                           transition=dict(duration=150, easing='cubic-in-out'), **layout_settings)

    greeks_call = [calculate_greeks(s, K, T, r, sigma, 'call') for s in spot_range]
    delta_c, gamma_c, theta_c, vega_c = zip(*greeks_call)

    fig_greeks = make_subplots(rows=2, cols=2, subplot_titles=("Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)"))
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=delta_c, name='Call Delta', line=dict(color=colors['call_text']), showlegend=False), 1, 1)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=gamma_c, name='Gamma', line=dict(color=colors['accent']), showlegend=False), 1, 2)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=theta_c, name='Call Theta', line=dict(color=colors['call_text']), showlegend=False), 2, 1)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=vega_c, name='Vega', line=dict(color=colors['accent']), showlegend=False), 2, 2)

    fig_greeks.update_layout(title="Greeks Sensitivity", margin=dict(l=20, r=20, t=40, b=20), uirevision='bs-greeks',
                             transition=dict(duration=150, easing='cubic-in-out'), **layout_settings)
    return fig_spot, fig_greeks



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




def fetch_polygon_surface(ticker_symbol, contract_type, moneyness_pct, api_key):
    """Pull a live options chain snapshot from Polygon.io and return (data_dict, error_string)."""
    if not POLYGON_AVAILABLE:
        return None, "polygon-api-client not installed. Run: pip install polygon-api-client"
    if not api_key or not api_key.strip():
        return None, "No API key. Enter your Polygon.io key in the sidebar."

    try:
        client = PolygonClient(api_key=api_key.strip())
        today = datetime.date.today()

        params = {"limit": 250}
        if contract_type in ("call", "put"):
            params["contract_type"] = contract_type

        contracts = list(client.list_snapshot_options_chain(
            ticker_symbol.upper(), params=params
        ))

        if not contracts:
            return None, (
                f"No options data returned for {ticker_symbol}. "
                "Check your API key or ensure your plan includes options data."
            )

        spot = None
        for c in contracts:
            if c.underlying_asset and c.underlying_asset.price:
                spot = c.underlying_asset.price
                break

        ref_date = today

        strikes, dtes, ivs, prices, deltas, volumes, exps, ctypes = [], [], [], [], [], [], [], []
        expirations_seen = set()

        for c in contracts:
            iv = c.implied_volatility
            if not iv or iv < 0.005:
                continue
            if not c.details:
                continue

            strike = c.details.strike_price
            exp_str = c.details.expiration_date   # "YYYY-MM-DD"

            # Moneyness filter (skip deep ITM / OTM)
            if spot and not (spot * (1 - moneyness_pct) <= strike <= spot * (1 + moneyness_pct)):
                continue

            exp_date = datetime.date.fromisoformat(exp_str)
            dte = (exp_date - ref_date).days
            if dte <= 0:
                continue

            # Option price: prefer bid-ask midpoint, fall back to last trade, then day close
            opt_price = None
            if c.last_quote and c.last_quote.midpoint:
                opt_price = c.last_quote.midpoint
            elif c.last_trade and c.last_trade.price:
                opt_price = c.last_trade.price
            elif c.day and c.day.close:
                opt_price = c.day.close

            strikes.append(strike)
            dtes.append(dte)
            ivs.append(iv)           # Polygon returns as decimal (0.25 = 25%)
            prices.append(opt_price)
            expirations_seen.add(exp_str)
            exps.append(exp_str)
            ctypes.append(c.details.contract_type if c.details.contract_type else 'unknown')

            deltas.append(abs(c.greeks.delta) if c.greeks and c.greeks.delta is not None else None)
            volumes.append(c.day.volume if c.day and c.day.volume else 0)

        if len(strikes) < 5:
            return None, "Too few liquid contracts returned. Try widening the moneyness range."

        return {
            "strikes": strikes, "dtes": dtes, "ivs": ivs, "prices": prices,
            "deltas": deltas, "volumes": volumes, "exps": exps, "ctypes": ctypes,
            "spot": spot,
            "contract_count": len(strikes),
            "exp_count": len(expirations_seen),
        }, None

    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            return None, "API key rejected or plan doesn't include options. Check polygon.io account."
        return None, f"Polygon error: {msg}"


def scan_ticker(ticker_symbol):
    """Fetch vol metrics for one ticker and return a table row dict."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            return None
        hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['HV'] = hist['Log_Ret'].rolling(window=30).std() * np.sqrt(252) * 100
        hist = hist.dropna()
        current_hv = hist['HV'].iloc[-1]
        hv_rank = ((current_hv - hist['HV'].min()) / (hist['HV'].max() - hist['HV'].min())) * 100

        current_iv, skew_ratio, pc_vol_ratio = None, None, None
        options = ticker.options
        if options:
            today = datetime.datetime.now().date()
            valid = [e for e in options if (datetime.datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 7]
            exp = valid[0] if valid else options[0]
            chain = ticker.option_chain(exp)
            spot = hist['Close'].iloc[-1]

            # ATM IV (calls)
            atm = chain.calls.iloc[(chain.calls['strike'] - spot).abs().argsort()[:1]]
            if not atm.empty:
                current_iv = atm['impliedVolatility'].values[0] * 100

            # Put skew: 90% strike vs ATM call IV
            closest_put = chain.puts.iloc[(chain.puts['strike'] - spot * 0.90).abs().argsort()[:1]]
            if not closest_put.empty and current_iv:
                skew_ratio = closest_put['impliedVolatility'].values[0] * 100 / current_iv

            # Put/Call volume ratio across whole chain
            total_call_vol = chain.calls['volume'].fillna(0).sum()
            total_put_vol = chain.puts['volume'].fillna(0).sum()
            if total_call_vol > 0:
                pc_vol_ratio = total_put_vol / total_call_vol

        vrp = (current_iv - current_hv) if current_iv else None
        vrp_ratio = (current_iv / current_hv) if current_iv and current_hv else None
        verdict = ('Expensive' if vrp_ratio and vrp_ratio > 1.25
                   else 'Cheap' if vrp_ratio and vrp_ratio < 0.80
                   else 'Neutral' if vrp_ratio else 'N/A')

        # --- Directional bias: skew + P/C volume each cast a vote ---
        skew_bearish = skew_ratio is not None and skew_ratio > 1.15
        skew_bullish = skew_ratio is not None and skew_ratio < 0.88
        pc_bearish   = pc_vol_ratio is not None and pc_vol_ratio > 1.10
        pc_bullish   = pc_vol_ratio is not None and pc_vol_ratio < 0.75
        bear_votes = int(skew_bearish) + int(pc_bearish)
        bull_votes = int(skew_bullish) + int(pc_bullish)
        bias = "Bearish" if bear_votes > bull_votes else "Bullish" if bull_votes > bear_votes else "Neutral"

        # --- Recommendations: direction + VRP tells you whether to buy or sell premium ---
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
            'IV %':     f"{current_iv:.1f}" if current_iv else 'N/A',
            'HV %':     f"{current_hv:.1f}",
            'VRP':      f"{vrp:+.1f}" if vrp is not None else 'N/A',
            'IV/HV':    f"{vrp_ratio:.2f}x" if vrp_ratio else 'N/A',
            'Skew':     f"{skew_ratio:.2f}x" if skew_ratio else 'N/A',
            'P/C Vol':  f"{pc_vol_ratio:.2f}" if pc_vol_ratio else 'N/A',
            'HV Rank':  f"{hv_rank:.0f}%",
            'Bias':     bias,
            'Call Rec': call_rec,
            'Put Rec':  put_rec,
            'Verdict':  verdict,
        }
    except:
        return None



# -----------------------------------------------------------------------------
# 3. APP LAYOUT & STYLES (Mobile Optimized)
# -----------------------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, title='Equity Research Dashboard',
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



# --- 2. BLACK-SCHOLES TAB LAYOUT ---
# CHANGE: Added helper text to each input explaining what the parameter means
bs_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Option Inputs", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Adjust parameters to price European options using the Black-Scholes model.", className='helper-text', style={'marginTop': '0'}),

            # Ticker lookup — auto-fills Spot & Strike from live price
            html.Label("Load from Ticker", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Div("Fetches live price to pre-fill Spot & Strike", className='helper-text'),
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '4px'}, children=[
                dcc.Input(id='bs-ticker-input', type='text', placeholder="e.g. AAPL, NVDA, SPY",
                          style={**INPUT_STYLE, 'flex': 1}),
                html.Button('Load', id='bs-load-btn', n_clicks=0, style={
                    'padding': '10px 14px', 'backgroundColor': '#1a1a1a',
                    'border': f"1px solid {colors['accent']}", 'borderRadius': '6px',
                    'color': colors['accent'], 'fontWeight': 'bold', 'cursor': 'pointer',
                    'whiteSpace': 'nowrap', 'fontSize': '0.9em'
                }),
            ]),
            html.Div(id='bs-ticker-status', style={'minHeight': '18px', 'fontSize': '0.82em', 'marginBottom': '4px'}),
            html.Hr(className='section-divider'),

            make_control_row("Spot Price ($)", "spot", 0, 800, 0.01, initial_spot,
                             helper="Current market price of the underlying asset"),
            make_control_row("Strike Price ($)", "strike", 0, 800, 0.01, initial_strike,
                             helper="Price at which the option can be exercised"),
            make_control_row("Time (Years)", "time", 0.01, 5, 0.01, DEFAULT_TIME,
                             helper="Time remaining until option expiration"),
            make_control_row("Volatility (σ)", "vol", 0.01, 1.5, 0.01, DEFAULT_VOL,
                             helper="Annualized standard deviation of returns"),
            make_control_row("Risk-Free Rate (r)", "rate", 0.0, 0.2, 0.001, DEFAULT_RATE,
                             helper="Annualized risk-free interest rate (e.g. T-bill)"),
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            # CHANGE: Redesigned price cards with more visual weight and moneyness indicator
            html.Div(style={'display': 'flex', 'gap': '12px', 'marginBottom': '16px'}, children=[
                html.Div(style={
                    'flex': 1, 'backgroundColor': '#001500', 'padding': '16px', 'borderRadius': '10px',
                    'textAlign': 'center', 'border': f"1px solid {colors['call_text']}"
                }, children=[
                    html.Div("CALL", style={'margin': '0', 'fontSize': '0.75em', 'color': colors['call_text'], 'letterSpacing': '1px', 'fontWeight': 'bold'}),
                    html.H2(id='call-price-display', style={'margin': '8px 0 0 0', 'color': colors['call_text'], 'fontSize': '1.8em'})
                ]),
                html.Div(style={
                    'flex': 1, 'backgroundColor': '#150000', 'padding': '16px', 'borderRadius': '10px',
                    'textAlign': 'center', 'border': f"1px solid {colors['put_text']}"
                }, children=[
                    html.Div("PUT", style={'margin': '0', 'fontSize': '0.75em', 'color': colors['put_text'], 'letterSpacing': '1px', 'fontWeight': 'bold'}),
                    html.H2(id='put-price-display', style={'margin': '8px 0 0 0', 'color': colors['put_text'], 'fontSize': '1.8em'})
                ]),
            ]),
            # CHANGE: Added Greeks summary row so users see key greeks at a glance
            html.Div(style={
                'display': 'flex', 'gap': '8px', 'marginBottom': '16px', 'flexWrap': 'wrap'
            }, children=[
                html.Div(id='greeks-delta-card', className='metric-card', style={'flex': '1 1 80px', 'textAlign': 'center', 'minWidth': '80px'}, children=[
                    html.Div("Delta", style={'color': colors['muted'], 'fontSize': '0.7em', 'textTransform': 'uppercase'}),
                    html.Div("--", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '1em'})
                ]),
                html.Div(id='greeks-gamma-card', className='metric-card', style={'flex': '1 1 80px', 'textAlign': 'center', 'minWidth': '80px'}, children=[
                    html.Div("Gamma", style={'color': colors['muted'], 'fontSize': '0.7em', 'textTransform': 'uppercase'}),
                    html.Div("--", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '1em'})
                ]),
                html.Div(id='greeks-theta-card', className='metric-card', style={'flex': '1 1 80px', 'textAlign': 'center', 'minWidth': '80px'}, children=[
                    html.Div("Theta", style={'color': colors['muted'], 'fontSize': '0.7em', 'textTransform': 'uppercase'}),
                    html.Div("--", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '1em'})
                ]),
                html.Div(id='greeks-vega-card', className='metric-card', style={'flex': '1 1 80px', 'textAlign': 'center', 'minWidth': '80px'}, children=[
                    html.Div("Vega", style={'color': colors['muted'], 'fontSize': '0.7em', 'textTransform': 'uppercase'}),
                    html.Div("--", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '1em'})
                ]),
            ]),
            html.Div(style={'backgroundColor': colors['card_bg'], 'borderRadius': '10px'}, children=[
                dcc.Tabs(style={'color': colors['text']}, children=[
                    dcc.Tab(label='Payoff', style={'backgroundColor': colors['card_bg'], 'color': '#666'}, selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
                        dcc.Graph(id='payoff-graph', style={'height': '50vh', 'minHeight': '350px'})  # CHANGE: Reduced height to accommodate Greeks cards
                    ]),
                    dcc.Tab(label='Greeks', style={'backgroundColor': colors['card_bg'], 'color': '#666'}, selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
                        dcc.Graph(id='greeks-graph', style={'height': '50vh', 'minHeight': '350px'})
                    ]),
                ])
            ])
        ])
    ])
])

# --- 3. SPREAD ANALYSIS TAB LAYOUT ---
PERIOD_MAP = {0: '1mo', 1: '3mo', 2: '6mo', 3: '1y', 4: '2y', 5: '5y', 6: 'max'}

SPREAD_LINE_COLORS = [
    '#ff6600',  # orange (accent)
    '#ef4444',  # red
    '#22c55e',  # green
    '#3b82f6',  # blue
    '#a855f7',  # purple
    '#eab308',  # yellow
    '#06b6d4',  # cyan
    '#f472b6',  # pink
    '#f97316',  # deep orange
    '#84cc16',  # lime
    '#6366f1',  # indigo
    '#14b8a6',  # teal
    '#fb923c',  # amber-orange
    '#e879f9',  # fuchsia
    '#4ade80',  # light green
    '#60a5fa',  # light blue
    '#c084fc',  # lavender
    '#fbbf24',  # gold
    '#34d399',  # emerald
    '#f87171',  # light red
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

            html.Label("Lookback Period", style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block', 'fontSize': '0.9em', 'marginTop': '12px'}),
            html.Div(style={'padding': '0 10px 20px 10px'}, children=[
                dcc.Slider(
                    id='spread-period-slider',
                    min=0, max=6, step=1,
                    value=2,
                    marks={0: '1M', 1: '3M', 2: '6M', 3: '1Y', 4: '2Y', 5: '5Y', 6: 'MAX'},
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
            html.P("Live implied volatility surface powered by Polygon.io.", className='helper-text', style={'marginTop': '0'}),

            html.Label("Ticker Symbol", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.Input(id='vol-ticker-input', type='text', value=DEFAULT_TICKER, placeholder="e.g. SPY, AAPL, TSLA",
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
                           tooltip={"placement": "bottom", "always_visible": False}),
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

            html.Button('Fetch Surface', id='vol-submit-btn', n_clicks=0, style={**BUTTON_STYLE, 'marginTop': '6px'}),
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
                                           tooltip={"placement": "bottom", "always_visible": False}),
                            ]),
                            dcc.Graph(id='vol-smile-chart', style={'height': '58vh', 'minHeight': '400px'}),
                        ]),
            ]),
        ])
    ])
])

# --- 5. VOLATILITY ANALYTICS TAB LAYOUT ---
vol_analytics_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Vol Analytics", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Compare realized volatility against implied volatility and track the live skew.", className='helper-text', style={'marginTop': '0'}),

            html.Label("Ticker Symbol", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.Input(id='va-ticker-input', type='text', value=DEFAULT_TICKER, placeholder="e.g. SPY",
                      style={**INPUT_STYLE, 'marginBottom': '15px'}),

            html.Label("HV Lookback Window (Days)", style={'color': colors['text'], 'fontWeight': 'bold', 'display': 'block', 'fontSize': '0.9em'}),
            html.Div("Number of trading days to compute rolling historical volatility.", className='helper-text'),
            dcc.Slider(id='va-window-slider', min=10, max=90, step=10, value=30, marks={10: '10d', 30: '30d', 60: '60d', 90: '90d'}),

            html.Button('Analyze Volatility', id='va-submit-btn', n_clicks=0, style={**BUTTON_STYLE, 'marginTop': '20px'}),
            html.Hr(className='section-divider'),

            html.Div(id='va-stats-display',
                children=html.Div([
                    html.P("Volatility metrics will appear here.", style={'color': colors['muted'], 'fontStyle': 'italic'})
                ])
            )
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(dcc.Graph(id='va-hv-chart', style={'height': '45vh', 'minHeight': '300px'}), type='circle'),
            dcc.Loading(dcc.Graph(id='va-skew-chart', style={'height': '35vh', 'minHeight': '250px', 'marginTop': '10px'}), type='circle')
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

            html.Button('Scan Options', id='scanner-submit-btn', n_clicks=0, style={**BUTTON_STYLE, 'marginTop': '10px', 'width': '100%'}),
            html.Div(id='scanner-status', style={'color': colors['muted'], 'fontSize': '0.85em', 'fontStyle': 'italic', 'marginTop': '6px'}),

            html.Hr(className='section-divider'),

            html.Div([
                html.P("How to read the table:", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.85em', 'marginBottom': '6px'}),
                make_stat_row("IV %", "ATM implied vol (front-month)"),
                make_stat_row("HV %", "30-day realized vol"),
                make_stat_row("VRP", "IV minus HV (+ = options expensive)"),
                make_stat_row("IV/HV", "> 1.25x = Expensive, < 0.80x = Cheap"),
                make_stat_row("Skew", "OTM put IV / ATM IV ratio"),
                make_stat_row("HV Rank", "Realized vol percentile (1yr)"),
            ])
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(html.Div(id='scanner-results-table'), type='circle'),
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
                dcc.Tab(label='Vol Surface', value='tab-vol',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Vol Analytics', value='tab-va',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Black-Scholes', value='tab-bs',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
    ]),

    html.Div(id='bs-content-wrapper', children=bs_layout, style={'display': 'none'}),
    html.Div(id='spread-content-wrapper', children=spread_layout, style={'display': 'none'}),
    html.Div(id='vol-content-wrapper', children=vol_surface_layout, style={'display': 'none'}),
    html.Div(id='va-content-wrapper', children=vol_analytics_layout, style={'display': 'none'}),
    html.Div(id='scanner-content-wrapper', children=scanner_layout, style={'display': 'block'}),

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
    [Output('bs-content-wrapper', 'style'),
     Output('spread-content-wrapper', 'style'), Output('vol-content-wrapper', 'style'),
     Output('va-content-wrapper', 'style'), Output('scanner-content-wrapper', 'style')],
    [Input('main-tabs', 'value')]
)
def toggle_tabs(tab_value):
    bs_style, spread_style, vol_style, va_style, scanner_style = [{'display': 'none'}] * 5
    if tab_value == 'tab-bs': bs_style = {'display': 'block'}
    elif tab_value == 'tab-spread': spread_style = {'display': 'block'}
    elif tab_value == 'tab-vol': vol_style = {'display': 'block'}
    elif tab_value == 'tab-va': va_style = {'display': 'block'}
    elif tab_value == 'tab-scanner': scanner_style = {'display': 'block'}
    return bs_style, spread_style, vol_style, va_style, scanner_style

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

    selected_period = PERIOD_MAP.get(slider_val, '6mo')

    try:
        series = {}
        for t in tickers:
            s = yf.Ticker(t).history(period=selected_period)['Close']
            if not s.empty:
                series[t] = s

        if not series:
            return go.Figure(layout=layout_settings), html.Div("No data found for any ticker.", style={'color': colors['danger']})

        df = pd.DataFrame(series).dropna()
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

        stat_rows = [
            html.H4("Returns", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),
            make_stat_row("Period", selected_period.upper()),
            html.Hr(className='section-divider'),
        ]
        for col in df.columns:
            ret = (df[col].iloc[-1] / df[col].iloc[0] - 1) * 100
            ret_color = colors['call_text'] if ret >= 0 else colors['put_text']
            stat_rows.append(make_stat_row(col, f"{ret:+.1f}%", ret_color))

        return fig_norm, html.Div(stat_rows)

    except Exception as e:
        return go.Figure(layout=layout_settings), html.Div([
            html.Div("Could not fetch data", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(f"{e}", style={'color': colors['muted'], 'fontSize': '0.85em'})
        ])

# --- VOLATILITY SURFACE CALLBACK (Polygon.io) ---
# Slider is a State — only the Fetch button fires an API call.
# Results are cached per (ticker, date, contract_type, moneyness) so re-visiting
# a previously fetched date is instant with zero additional API calls.
def _build_surface_figure(data, sym, contract_type, z_axis, plot_type, moneyness_pct):
    """Render a surface/scatter figure from a cached data dict. Returns (fig, info_html, smile_marks, smile_max)."""
    strikes    = data['strikes']
    dtes       = data['dtes']
    ivs        = data['ivs']
    raw_prices = data['prices']
    spot       = data['spot']

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
    else:
        fig.add_trace(go.Scatter3d(
            x=z_strikes, y=z_dtes, z=z_vals, mode='markers',
            marker=dict(size=3, color=z_vals, colorscale='Jet', opacity=0.85,
                        colorbar=dict(title=z_label, tickprefix=z_tickprefix, ticksuffix=z_ticksuffix)),
            hovertemplate=f"Strike: $%{{x:.0f}}<br>DTE: %{{y:.0f}}d<br>{z_display}: {hover_z}<extra></extra>",
            name=z_display,
        ))

    today_str = datetime.date.today().isoformat()
    fig.update_layout(
        title=f"{sym} {ct_label} {z_display} Surface — Live{spot_label}",
        scene=dict(
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=0), eye=dict(x=-1.8,y=-1.2,z=1.0)),
            xaxis_title='Strike ($)', yaxis_title='DTE (days)', zaxis_title=z_label,
            xaxis=dict(backgroundcolor=colors['card_bg'], gridcolor='#333', showbackground=True),
            yaxis=dict(backgroundcolor=colors['card_bg'], gridcolor='#333', showbackground=True),
            zaxis=dict(backgroundcolor=colors['card_bg'], gridcolor='#333', showbackground=True),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision=sym,
        **layout_settings
    )

    iv_pcts    = [v * 100 for v in ivs]
    price_vals = [p for p in raw_prices if p is not None]
    info_html = html.Div([
        html.H4("Surface Data · Polygon.io", style={'color': colors['text'], 'marginBottom': '10px', 'fontSize': '1em'}),
        make_stat_row("Source",       "Polygon.io — Live"),
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
    [Input('vol-submit-btn', 'n_clicks')],
    [State('vol-ticker-input', 'value'),
     State('vol-contract-type', 'value'),
     State('vol-moneyness-slider', 'value'),
     State('vol-z-axis', 'value'),
     State('vol-plot-type', 'value')],
    prevent_initial_call=True
)
def update_vol_surface(_n, ticker_symbol, contract_type, moneyness_slider, z_axis, plot_type):
    empty_fig = go.Figure(layout=layout_settings)

    if not ticker_symbol:
        return empty_fig, html.Div(), no_update, no_update, no_update, no_update

    sym           = ticker_symbol.upper().strip()
    moneyness_pct = (moneyness_slider or 25) / 100

    data, err = fetch_polygon_surface(sym, contract_type, moneyness_pct, POLYGON_API_KEY)
    if err:
        info = html.Div([
            html.Div("Data fetch failed", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(err, style={'color': colors['muted'], 'fontSize': '0.85em'}),
        ])
        return empty_fig, info, no_update, no_update, no_update, no_update

    fig, info_html, smile_marks, smile_max = _build_surface_figure(
        data, sym, contract_type, z_axis, plot_type, moneyness_pct)
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


# --- VOLATILITY ANALYTICS CALLBACK ---
# CHANGE: Improved stats panel with visual indicators and better organization
@app.callback(
    [Output('va-hv-chart', 'figure'), Output('va-skew-chart', 'figure'), Output('va-stats-display', 'children')],
    [Input('va-submit-btn', 'n_clicks')],
    [State('va-ticker-input', 'value'), State('va-window-slider', 'value')]
)
def update_vol_analytics(n_clicks, ticker_symbol, window):
    if not ticker_symbol:
        return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div()

    ticker_symbol = ticker_symbol.upper().strip()

    try:
        ticker = yf.Ticker(ticker_symbol)

        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < window:
            return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div(f"Not enough historical data for {ticker_symbol}.", style={'color': colors['danger']})

        hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['HV'] = hist['Log_Ret'].rolling(window=window).std() * np.sqrt(252) * 100
        hist = hist.dropna()

        current_hv = hist['HV'].iloc[-1]
        hv_min = hist['HV'].min()
        hv_max = hist['HV'].max()
        rvr = ((current_hv - hv_min) / (hv_max - hv_min)) * 100

        options = ticker.options
        current_iv = None
        fig_skew = go.Figure()
        skew_ratio_text = "N/A"

        if options:
            today = datetime.datetime.now().date()
            valid_expiries = [exp for exp in options if (datetime.datetime.strptime(exp, "%Y-%m-%d").date() - today).days >= 7]
            target_exp = valid_expiries[0] if valid_expiries else options[0]

            chain = ticker.option_chain(target_exp)
            calls = chain.calls
            puts = chain.puts
            spot_price = hist['Close'].iloc[-1]

            atm_call = calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]]
            if not atm_call.empty:
                current_iv = atm_call['impliedVolatility'].values[0] * 100

            lower_bound, upper_bound = spot_price * 0.8, spot_price * 1.2
            calls_skew = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound) & (calls['volume'] > 0)]
            puts_skew = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound) & (puts['volume'] > 0)]

            fig_skew.add_trace(go.Scatter(x=puts_skew['strike'], y=puts_skew['impliedVolatility']*100, mode='lines+markers', name='Puts IV', line=dict(color=colors['put_text'])))
            fig_skew.add_trace(go.Scatter(x=calls_skew['strike'], y=calls_skew['impliedVolatility']*100, mode='lines+markers', name='Calls IV', line=dict(color=colors['call_text'])))
            fig_skew.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="#666", annotation_text="Spot")

            fig_skew.update_layout(title=f"Live Volatility Skew (Expiry: {target_exp})", xaxis_title="Strike Price ($)", yaxis_title="Implied Volatility (%)", margin=dict(l=20, r=20, t=40, b=20),
                                   uirevision=ticker_symbol, transition=dict(duration=200, easing='cubic-in-out'), **layout_settings)

            otm_put_target = spot_price * 0.90
            closest_put = puts.iloc[(puts['strike'] - otm_put_target).abs().argsort()[:1]]
            skew_ratio_num = None
            if not closest_put.empty and current_iv:
                otm_put_iv = closest_put['impliedVolatility'].values[0] * 100
                skew_ratio_num = otm_put_iv / current_iv
                skew_ratio_text = f"{skew_ratio_num:.2f}x"

            # Term structure: compare front-month vs back-month ATM IV
            back_month_iv = None
            back_valid = [exp for exp in options if (datetime.datetime.strptime(exp, "%Y-%m-%d").date() - today).days >= 45]
            if back_valid and current_iv:
                try:
                    back_chain = ticker.option_chain(back_valid[0])
                    back_calls = back_chain.calls
                    atm_back = back_calls.iloc[(back_calls['strike'] - spot_price).abs().argsort()[:1]]
                    if not atm_back.empty:
                        back_month_iv = atm_back['impliedVolatility'].values[0] * 100
                except:
                    pass
            term_slope = (back_month_iv - current_iv) if back_month_iv else None

        else:
            skew_ratio_num = None
            term_slope = None
            fig_skew.update_layout(title="No Options Data Available for Skew", uirevision=ticker_symbol, **layout_settings)

        fig_hv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            row_heights=[0.6, 0.4], subplot_titles=(f"{ticker_symbol} Price", f"{window}-Day Rolling Historical Volatility (HV)"))
        fig_hv.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price', line=dict(color=colors['accent'], width=2)), row=1, col=1)
        fig_hv.add_trace(go.Scatter(x=hist.index, y=hist['HV'], mode='lines', name=f'{window}d HV', line=dict(color=colors['put_text'], width=2)), row=2, col=1)

        fig_hv.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
                             uirevision=ticker_symbol, transition=dict(duration=200, easing='cubic-in-out'), **layout_settings)
        fig_hv.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_hv.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        # CHANGE: Redesigned stats panel with sections, visual HV rank bar, and clearer layout
        vrp_text = "N/A"
        vrp_color = colors['text']
        if current_iv:
            vrp = current_iv - current_hv
            vrp_text = f"{vrp:+.2f}%"
            vrp_color = colors['success'] if vrp > 0 else colors['danger']

        stats_html = html.Div([
            html.H4("Volatility Metrics", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),

            make_stat_row("Realized HV", f"{current_hv:.2f}%"),
            make_stat_row("ATM Implied Vol", f"{current_iv:.2f}%" if current_iv else "N/A", colors['accent']),
            make_stat_row("Vol Risk Premium", vrp_text, vrp_color),

            # CHANGE: VRP badge indicating if options are "rich" or "cheap"
            html.Div(style={'marginBottom': '12px', 'marginTop': '4px'}, children=[
                html.Span(
                    "Options Expensive" if current_iv and (current_iv - current_hv) > 0 else "Options Cheap",
                    className=f"badge {'badge-green' if current_iv and (current_iv - current_hv) > 0 else 'badge-red'}"
                )
            ]) if current_iv else html.Div(),

            html.Hr(className='section-divider'),

            make_stat_row("OTM Put Skew", skew_ratio_text),

            html.Hr(className='section-divider'),

            # CHANGE: HV Rank with a visual progress bar
            html.Div(style={'marginBottom': '8px'}, children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '6px'}, children=[
                    html.Span("HV Rank (Percentile)", style={'color': colors['muted'], 'fontSize': '0.9em'}),
                    html.Span(f"{rvr:.1f}%", style={'fontWeight': 'bold', 'color': colors['call_text'] if rvr < 50 else colors['put_text']})
                ]),
                # CHANGE: Visual progress bar for HV rank
                html.Div(style={'height': '8px', 'backgroundColor': '#1a1a1a', 'borderRadius': '4px', 'overflow': 'hidden'}, children=[
                    html.Div(style={
                        'height': '100%', 'borderRadius': '4px',
                        'width': f"{min(rvr, 100)}%",
                        'backgroundColor': colors['call_text'] if rvr < 50 else colors['put_text'],
                        'transition': 'width 0.3s ease'
                    })
                ]),
                # CHANGE: Label under the bar for quick interpretation
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '4px'}, children=[
                    html.Span("Low", style={'color': '#666', 'fontSize': '0.7em'}),
                    html.Span("High", style={'color': '#666', 'fontSize': '0.7em'})
                ])
            ]),

            html.Hr(className='section-divider'),
            html.H4("Trading Signals", style={'color': colors['text'], 'marginBottom': '10px', 'fontSize': '1em'}),
            *[html.Div(style={'marginBottom': '10px'}, children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '3px'}, children=[
                    html.Span(s['label'], style={'color': colors['muted'], 'fontSize': '0.8em', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                    html.Span(s['signal'], className=f"badge {s['badge']}")
                ]),
                html.Div(s['detail'], style={'color': colors['muted'], 'fontSize': '0.75em', 'fontStyle': 'italic'})
            ]) for s in generate_vol_signals(current_iv, current_hv, rvr, skew_ratio_num, term_slope)]
        ])

        return fig_hv, fig_skew, stats_html

    except Exception as e:
        return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div([
            html.Div("Could not fetch data", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(f"{e}", style={'color': colors['muted'], 'fontSize': '0.85em'})
        ])

# --- BS TICKER LOOKUP ---
@app.callback(
    [Output('spot-input', 'value', allow_duplicate=True),
     Output('spot-slider', 'value', allow_duplicate=True),
     Output('strike-input', 'value', allow_duplicate=True),
     Output('strike-slider', 'value', allow_duplicate=True),
     Output('bs-ticker-status', 'children')],
    [Input('bs-load-btn', 'n_clicks')],
    [State('bs-ticker-input', 'value')],
    prevent_initial_call=True
)
def load_bs_ticker_price(n_clicks, ticker_symbol):
    if not n_clicks or not ticker_symbol:
        return (no_update,) * 5
    sym = ticker_symbol.strip().upper()
    try:
        hist = yf.Ticker(sym).history(period='5d')
        if hist.empty:
            return no_update, no_update, no_update, no_update, \
                   html.Span(f'"{sym}" not found.', style={'color': colors['danger']})
        price = round(hist['Close'].iloc[-1], 2)
        status = html.Span(f"{sym} @ ${price:.2f}", style={'color': colors['call_text']})
        return price, price, price, price, status
    except Exception as e:
        return no_update, no_update, no_update, no_update, \
               html.Span(f"Error: {e}", style={'color': colors['danger']})

# --- BLACK-SCHOLES SYNC ---
def sync_input(slider_val, input_val):
    trigger_id = ctx.triggered_id
    if trigger_id and 'slider' in trigger_id: return slider_val, no_update
    return no_update, input_val

@app.callback([Output('spot-input', 'value', allow_duplicate=True), Output('spot-slider', 'value', allow_duplicate=True)],
              [Input('spot-slider', 'value'), Input('spot-input', 'value')], prevent_initial_call=True)
def sync_spot_ui(s, b): return sync_input(s, b)

@app.callback([Output('strike-input', 'value', allow_duplicate=True), Output('strike-slider', 'value', allow_duplicate=True)],
              [Input('strike-slider', 'value'), Input('strike-input', 'value')], prevent_initial_call=True)
def sync_strike_ui(s, b): return sync_input(s, b)

@app.callback([Output('time-input', 'value'), Output('time-slider', 'value')], [Input('time-slider', 'value'), Input('time-input', 'value')])
def sync_time(s, b): return sync_input(s, b)
@app.callback([Output('vol-input', 'value'), Output('vol-slider', 'value')], [Input('vol-slider', 'value'), Input('vol-input', 'value')])
def sync_vol(s, b): return sync_input(s, b)
@app.callback([Output('rate-input', 'value'), Output('rate-slider', 'value')], [Input('rate-slider', 'value'), Input('rate-input', 'value')])
def sync_rate(s, b): return sync_input(s, b)

# CHANGE: Added Greeks summary cards output alongside call/put prices
@app.callback(
    [Output('call-price-display', 'children'), Output('put-price-display', 'children'),
     Output('payoff-graph', 'figure'), Output('greeks-graph', 'figure'),
     Output('greeks-delta-card', 'children'), Output('greeks-gamma-card', 'children'),
     Output('greeks-theta-card', 'children'), Output('greeks-vega-card', 'children')],
    [Input('spot-input', 'value'), Input('strike-input', 'value'), Input('time-input', 'value'), Input('vol-input', 'value'), Input('rate-input', 'value')]
)
def calc_bs(S, K, T, r, sigma):
    if None in [S, K, T, r, sigma]: return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    call = black_scholes(S, K, T, r, sigma, 'call')
    put = black_scholes(S, K, T, r, sigma, 'put')
    fig_spot, fig_greeks = get_bs_charts(S, K, T, r, sigma)

    # CHANGE: Calculate current Greeks for the summary cards
    delta, gamma, theta, vega = calculate_greeks(S, K, T, r, sigma, 'call')

    def greek_card_content(name, value, fmt=".4f"):
        return [
            html.Div(name, style={'color': colors['muted'], 'fontSize': '0.7em', 'textTransform': 'uppercase'}),
            html.Div(f"{value:{fmt}}", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '1em'})
        ]

    return (
        f"${call:.2f}", f"${put:.2f}", fig_spot, fig_greeks,
        greek_card_content("Delta", delta),
        greek_card_content("Gamma", gamma),
        greek_card_content("Theta", theta),
        greek_card_content("Vega", vega),
    )

# --- SCANNER CALLBACK ---
@app.callback(
    [Output('scanner-results-table', 'children'), Output('scanner-status', 'children')],
    [Input('scanner-submit-btn', 'n_clicks')],
    [State('scanner-tickers-input', 'value')]
)
def run_scanner(n_clicks, tickers_raw):
    if not n_clicks or not tickers_raw:
        return html.Div("Enter tickers and click Scan.", style={'color': colors['muted'], 'fontStyle': 'italic', 'padding': '20px'}), ""
    tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
    rows = [r for r in (scan_ticker(t) for t in tickers) if r]
    if not rows:
        return html.Div("No data returned. Check tickers.", style={'color': colors['danger'], 'padding': '20px'}), ""

    col_order = ['Ticker', 'IV %', 'HV %', 'VRP', 'IV/HV', 'Skew', 'P/C Vol', 'HV Rank', 'Bias', 'Call Rec', 'Put Rec', 'Verdict']
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
            # Verdict coloring
            {'if': {'filter_query': '{Verdict} = "Expensive"'}, 'backgroundColor': '#150000', 'color': colors['put_text']},
            {'if': {'filter_query': '{Verdict} = "Cheap"'},     'backgroundColor': '#001500', 'color': colors['call_text']},
            # Bias coloring
            {'if': {'filter_query': '{Bias} = "Bearish"', 'column_id': 'Bias'}, 'color': colors['put_text'], 'fontWeight': 'bold'},
            {'if': {'filter_query': '{Bias} = "Bullish"', 'column_id': 'Bias'}, 'color': colors['call_text'], 'fontWeight': 'bold'},
            {'if': {'filter_query': '{Bias} = "Neutral"', 'column_id': 'Bias'}, 'color': colors['muted']},
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
        ],
        sort_action='native',
        tooltip_header={
            'Skew':    'OTM 90% put IV ÷ ATM call IV. >1.15 = bearish skew, <0.88 = bullish',
            'P/C Vol': 'Total put volume ÷ total call volume. >1.10 = bearish, <0.75 = bullish',
            'Bias':    'Directional vote from Skew + P/C Vol. Both must agree for Bullish/Bearish.',
            'Call Rec':'Buy = bullish bias + cheap IV. Sell = bearish bias + expensive IV.',
            'Put Rec': 'Buy = bearish bias + cheap IV. Sell = bullish bias + expensive IV.',
            'Verdict': 'IV/HV ratio. >1.25x = Expensive (sell premium). <0.80x = Cheap (buy premium).',
        },
        tooltip_delay=0,
        tooltip_duration=None,
    )
    return table, f"Scanned {len(rows)} of {len(tickers)} tickers."



if __name__ == '__main__':
    app.run(debug=True)
