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
DEFAULT_SCANNER_TICKERS = "SPY, QQQ, AAPL, TSLA, NVDA, AMD, MSFT, AMZN, META, GOOGL"

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
                           margin=dict(l=20, r=20, t=40, b=20), **layout_settings)

    greeks_call = [calculate_greeks(s, K, T, r, sigma, 'call') for s in spot_range]
    delta_c, gamma_c, theta_c, vega_c = zip(*greeks_call)

    fig_greeks = make_subplots(rows=2, cols=2, subplot_titles=("Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)"))
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=delta_c, name='Call Delta', line=dict(color=colors['call_text']), showlegend=False), 1, 1)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=gamma_c, name='Gamma', line=dict(color=colors['accent']), showlegend=False), 1, 2)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=theta_c, name='Call Theta', line=dict(color=colors['call_text']), showlegend=False), 2, 1)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=vega_c, name='Vega', line=dict(color=colors['accent']), showlegend=False), 2, 2)

    fig_greeks.update_layout(title="Greeks Sensitivity", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)
    return fig_spot, fig_greeks


# CHANGE: Helper to build a metric card used across Fundamentals and other tabs
def make_metric_card(label, value, color=None):
    """Creates a compact metric card with label on top and value below."""
    return html.Div(className='metric-card', children=[
        html.Div(label, style={'color': colors['muted'], 'fontSize': '0.75em', 'marginBottom': '4px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
        html.Div(str(value), style={'color': color or colors['text'], 'fontSize': '1.1em', 'fontWeight': 'bold'})
    ])


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

        current_iv, skew_ratio = None, None
        options = ticker.options
        if options:
            today = datetime.datetime.now().date()
            valid = [e for e in options if (datetime.datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 7]
            exp = valid[0] if valid else options[0]
            chain = ticker.option_chain(exp)
            spot = hist['Close'].iloc[-1]
            atm = chain.calls.iloc[(chain.calls['strike'] - spot).abs().argsort()[:1]]
            if not atm.empty:
                current_iv = atm['impliedVolatility'].values[0] * 100
            closest_put = chain.puts.iloc[(chain.puts['strike'] - spot * 0.90).abs().argsort()[:1]]
            if not closest_put.empty and current_iv:
                skew_ratio = closest_put['impliedVolatility'].values[0] * 100 / current_iv

        vrp = (current_iv - current_hv) if current_iv else None
        vrp_ratio = (current_iv / current_hv) if current_iv and current_hv else None
        verdict = ('Expensive' if vrp_ratio and vrp_ratio > 1.25
                   else 'Cheap' if vrp_ratio and vrp_ratio < 0.80
                   else 'Neutral' if vrp_ratio else 'N/A')

        return {
            'Ticker': ticker_symbol,
            'IV %': f"{current_iv:.1f}" if current_iv else 'N/A',
            'HV %': f"{current_hv:.1f}",
            'VRP': f"{vrp:+.1f}" if vrp is not None else 'N/A',
            'IV/HV': f"{vrp_ratio:.2f}x" if vrp_ratio else 'N/A',
            'Skew': f"{skew_ratio:.2f}x" if skew_ratio else 'N/A',
            'HV Rank': f"{hv_rank:.0f}%",
            'Verdict': verdict,
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
    'maxWidth': '1400px',
    'margin': '0 auto'
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


# --- 1. FUNDAMENTAL TAB LAYOUT ---
# CHANGE: Added placeholder text and better empty-state guidance
fundamental_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.Div([
                html.H3("Stock Search", style={'color': colors['accent'], 'marginBottom': '4px'}),
                # CHANGE: Added description so users know what this tab does
                html.P("Look up any ticker to view fundamentals and price history.", className='helper-text', style={'marginTop': '0'}),
                dcc.Input(id='fund-ticker-input', type='text', value=DEFAULT_TICKER, placeholder="Enter ticker (e.g. AAPL, MSFT, TSLA)",
                          style=INPUT_STYLE),
                html.Button('Analyze', id='fund-submit-btn', n_clicks=0, style={**BUTTON_STYLE, 'marginTop': '10px'}),
                html.Hr(className='section-divider'),
                html.Div(id='fund-info-display',
                    # CHANGE: Default empty-state content so sidebar isn't blank on load
                    children=html.Div([
                        html.P("Company info will appear here after search.", style={'color': colors['muted'], 'fontStyle': 'italic'})
                    ])
                )
            ])
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(dcc.Graph(id='fund-price-chart', style={'height': '60vh', 'minHeight': '400px'}), type='circle')
        ])
    ])
])

# --- 2. BLACK-SCHOLES TAB LAYOUT ---
# CHANGE: Added helper text to each input explaining what the parameter means
bs_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Option Inputs", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Adjust parameters to price European options using the Black-Scholes model.", className='helper-text', style={'marginTop': '0'}),
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

spread_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Spread Inputs", style={'color': colors['accent'], 'marginBottom': '4px'}),
            # CHANGE: Added description explaining what spread analysis does
            html.P("Compare two tickers to find relative value and mean-reversion signals.", className='helper-text', style={'marginTop': '0'}),

            html.Label("Stock A (Numerator)", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Div("The stock you think will outperform", className='helper-text'),
            dcc.Input(id='spread-ticker-a', type='text', value=DEFAULT_SPREAD_A, placeholder="e.g. KO",
                      style={**INPUT_STYLE, 'marginBottom': '12px'}),

            html.Label("Stock B (Denominator)", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Div("The stock you are comparing against", className='helper-text'),
            dcc.Input(id='spread-ticker-b', type='text', value=DEFAULT_SPREAD_B, placeholder="e.g. PEP",
                      style={**INPUT_STYLE, 'marginBottom': '16px'}),

            html.Label("Lookback Period", style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block', 'fontSize': '0.9em'}),
            html.Div(style={'padding': '0 10px 20px 10px'}, children=[
                dcc.Slider(
                    id='spread-period-slider',
                    min=0, max=6, step=1,
                    value=2,
                    marks={0: '1M', 1: '3M', 2: '6M', 3: '1Y', 4: '2Y', 5: '5Y', 6: 'MAX'},
                )
            ]),

            html.Button('Analyze Spread', id='spread-analyze-btn', n_clicks=0, style=BUTTON_STYLE),
            html.Hr(className='section-divider'),

            html.Div(id='spread-stats-display',
                # CHANGE: Default empty state
                children=html.Div([
                    html.P("Spread statistics will appear here.", style={'color': colors['muted'], 'fontStyle': 'italic'})
                ])
            )
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Tabs(style={'color': colors['text']}, children=[
                dcc.Tab(label='Normalized Performance', style={'backgroundColor': colors['card_bg'], 'color': '#666'},
                        selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
                    dcc.Loading(dcc.Graph(id='spread-norm-chart', style={'height': '60vh', 'minHeight': '400px'}), type='circle')
                ]),
                # CHANGE: Expanded tab label from "Spread Ratio" to full text for clarity
                dcc.Tab(label='Spread Ratio', style={'backgroundColor': colors['card_bg'], 'color': '#666'},
                        selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
                    dcc.Loading(dcc.Graph(id='spread-ratio-chart', style={'height': '60vh', 'minHeight': '400px'}), type='circle')
                ]),
            ])
        ])
    ])
])

# --- 4. VOLATILITY SURFACE TAB LAYOUT ---
vol_surface_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Vol Surface", style={'color': colors['accent'], 'marginBottom': '4px'}),
            html.P("Visualize how implied volatility varies across strikes and expirations.", className='helper-text', style={'marginTop': '0'}),

            html.Label("Ticker Symbol", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            dcc.Input(id='vol-ticker-input', type='text', value=DEFAULT_TICKER, placeholder="e.g. SPY, AAPL, TSLA",
                      style={**INPUT_STYLE, 'marginBottom': '12px'}),

            html.Label("Plot Type", style={'color': colors['text'], 'fontWeight': 'bold', 'display': 'block', 'fontSize': '0.9em'}),
            # CHANGE: Added descriptions to radio options so users understand the difference
            dcc.RadioItems(
                id='vol-plot-type',
                options=[
                    {'label': ' Surface (Interpolated)', 'value': 'surface'},
                    {'label': ' Scatter (Raw Data)', 'value': 'scatter'}
                ],
                value='surface',
                labelStyle={'display': 'block', 'color': colors['text'], 'marginBottom': '5px', 'cursor': 'pointer'},
                style={'marginBottom': '10px'}
            ),
            html.Div("Surface smooths the data; Scatter shows actual market quotes.", className='helper-text'),

            html.Button('Fetch Options Data', id='vol-submit-btn', n_clicks=0, style={**BUTTON_STYLE, 'marginTop': '10px'}),
            html.Hr(className='section-divider'),
            html.Div(id='vol-info-display',
                children=html.Div([
                    html.P("Options data will load here.", style={'color': colors['muted'], 'fontStyle': 'italic'})
                ])
            )
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(dcc.Graph(id='vol-surface-chart', style={'height': '70vh', 'minHeight': '500px'}), type='circle')
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
            html.Button('Scan Options', id='scanner-submit-btn', n_clicks=0, style={**BUTTON_STYLE, 'marginTop': '10px'}),
            html.Hr(className='section-divider'),
            html.Div(id='scanner-status', style={'color': colors['muted'], 'fontSize': '0.85em', 'fontStyle': 'italic'}),
            html.Hr(className='section-divider'),
            html.Div([
                html.P("How to read the table:", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '0.85em', 'marginBottom': '6px'}),
                make_stat_row("IV %", "ATM implied vol (front-month)"),
                make_stat_row("HV %", "30-day realized vol"),
                make_stat_row("VRP", "IV minus HV (+ = options rich)"),
                make_stat_row("IV/HV", "> 1.25x = Expensive, < 0.80x = Cheap"),
                make_stat_row("Skew", "OTM put IV / ATM IV ratio"),
                make_stat_row("HV Rank", "Realized vol percentile (1yr)"),
            ])
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(html.Div(id='scanner-results-table'), type='circle')
        ])
    ])
])

# --- APP LAYOUT ---
# CHANGE: Revamped header with subtitle, added footer
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '10px 10px 0 10px', 'fontFamily': "'Segoe UI', Arial, sans-serif"}, children=[

    # CHANGE: Revamped header with subtitle and visual separator
    html.Div(style={'textAlign': 'center', 'padding': '16px 0 8px 0', 'maxWidth': '1400px', 'margin': '0 auto'}, children=[
        html.H1("Equity Research Dashboard", style={
            'color': colors['text'], 'fontSize': '1.6rem', 'margin': '0', 'fontWeight': '700', 'letterSpacing': '0.5px'
        }),
        # CHANGE: Added subtitle so users immediately understand the app's purpose
        html.P("Options pricing, volatility analysis & relative value tools", style={
            'color': colors['muted'], 'fontSize': '0.85em', 'margin': '6px 0 0 0'
        }),
    ]),

    # Navigation Tabs
    dcc.Tabs(id='main-tabs', value='tab-fundamental',
             style={'marginTop': '16px', 'marginBottom': '20px', 'maxWidth': '1400px', 'margin': '16px auto 20px auto'},
             children=[
                # CHANGE: Expanded tab labels for clarity (e.g. "Spread" -> "Spread Analysis")
                dcc.Tab(label='Fundamentals', value='tab-fundamental',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Black-Scholes', value='tab-bs',
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
                dcc.Tab(label='Scanner', value='tab-scanner',
                        style={'backgroundColor': colors['card_bg'], 'color': '#666', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'},
                        selected_style={'backgroundColor': '#1a1a1a', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
    ]),

    html.Div(id='fund-content-wrapper', children=fundamental_layout, style={'display': 'block'}),
    html.Div(id='bs-content-wrapper', children=bs_layout, style={'display': 'none'}),
    html.Div(id='spread-content-wrapper', children=spread_layout, style={'display': 'none'}),
    html.Div(id='vol-content-wrapper', children=vol_surface_layout, style={'display': 'none'}),
    html.Div(id='va-content-wrapper', children=vol_analytics_layout, style={'display': 'none'}),
    html.Div(id='scanner-content-wrapper', children=scanner_layout, style={'display': 'none'}),

    # CHANGE: Added footer with context so the app feels polished
    html.Div(style={
        'textAlign': 'center', 'padding': '20px 0', 'marginTop': '30px',
        'borderTop': f"1px solid {colors['card_border']}", 'maxWidth': '1400px', 'margin': '30px auto 0 auto'
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
    [Output('fund-content-wrapper', 'style'), Output('bs-content-wrapper', 'style'),
     Output('spread-content-wrapper', 'style'), Output('vol-content-wrapper', 'style'),
     Output('va-content-wrapper', 'style'), Output('scanner-content-wrapper', 'style')],
    [Input('main-tabs', 'value')]
)
def toggle_tabs(tab_value):
    fund_style, bs_style, spread_style, vol_style, va_style, scanner_style = [{'display': 'none'}] * 6
    if tab_value == 'tab-fundamental': fund_style = {'display': 'block'}
    elif tab_value == 'tab-bs': bs_style = {'display': 'block'}
    elif tab_value == 'tab-spread': spread_style = {'display': 'block'}
    elif tab_value == 'tab-vol': vol_style = {'display': 'block'}
    elif tab_value == 'tab-va': va_style = {'display': 'block'}
    elif tab_value == 'tab-scanner': scanner_style = {'display': 'block'}
    return fund_style, bs_style, spread_style, vol_style, va_style, scanner_style

# Main Fundamental Analysis Callback
# CHANGE: Revamped the info display to use a grid of metric cards instead of plain text
@app.callback(
    [Output('fund-info-display', 'children'), Output('fund-price-chart', 'figure'),
     Output('spot-input', 'value'), Output('spot-slider', 'value'),
     Output('strike-input', 'value'), Output('strike-slider', 'value')],
    [Input('fund-submit-btn', 'n_clicks'), Input('fund-ticker-input', 'value')],
    [State('fund-ticker-input', 'value')]
)
def update_fundamental_and_sync(n_clicks, input_val_trigger, ticker_symbol):
    if not ticker_symbol: return (no_update,) * 6

    ticker_symbol = ticker_symbol.upper().strip()

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="6mo")

        peg = info.get('pegRatio')
        pe = info.get('trailingPE')
        peg_display = f"{peg:.2f}" if peg is not None else "N/A"
        pe_display = f"{pe:.2f}" if pe is not None else "N/A"

        # CHANGE: Format market cap into readable form (e.g. $1.5T, $230B)
        market_cap = info.get('marketCap')
        if market_cap:
            if market_cap >= 1e12:
                mc_display = f"${market_cap/1e12:.1f}T"
            elif market_cap >= 1e9:
                mc_display = f"${market_cap/1e9:.1f}B"
            elif market_cap >= 1e6:
                mc_display = f"${market_cap/1e6:.1f}M"
            else:
                mc_display = f"${market_cap:,.0f}"
        else:
            mc_display = "N/A"

        beta = info.get('beta')
        beta_display = f"{beta:.2f}" if beta is not None else "N/A"

        div_yield = info.get('dividendYield')
        div_display = f"{div_yield*100:.2f}%" if div_yield is not None else "N/A"

        fifty_two_high = info.get('fiftyTwoWeekHigh')
        fifty_two_low = info.get('fiftyTwoWeekLow')

        # CHANGE: Redesigned info display with company name header + grid of metric cards
        info_html = html.Div([
            html.H2(f"{info.get('shortName', ticker_symbol)}", style={'marginTop': 0, 'marginBottom': '4px', 'color': colors['accent'], 'fontSize': '1.2em'}),
            html.P(f"{info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}",
                   style={'color': colors['muted'], 'fontSize': '0.85em', 'marginTop': '0', 'marginBottom': '16px'}),

            # CHANGE: Metric grid layout instead of stacked paragraphs
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '10px'}, children=[
                make_metric_card("Market Cap", mc_display),
                make_metric_card("Beta", beta_display),
                make_metric_card("P/E (TTM)", pe_display),
                make_metric_card("PEG Ratio", peg_display),
                make_metric_card("Div. Yield", div_display),
                make_metric_card("52W High", f"${fifty_two_high:.2f}" if fifty_two_high else "N/A"),
                make_metric_card("52W Low", f"${fifty_two_low:.2f}" if fifty_two_low else "N/A"),
                # CHANGE: Added 52W Range bar showing where current price sits
                html.Div(className='metric-card', children=[
                    html.Div("52W Range", style={'color': colors['muted'], 'fontSize': '0.75em', 'marginBottom': '6px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                    html.Div(style={'position': 'relative', 'height': '6px', 'backgroundColor': '#333', 'borderRadius': '3px', 'overflow': 'hidden'}, children=[
                        html.Div(style={
                            'position': 'absolute', 'left': '0', 'top': '0', 'height': '100%',
                            'width': f"{((hist['Close'].iloc[-1] - fifty_two_low) / (fifty_two_high - fifty_two_low) * 100) if fifty_two_high and fifty_two_low and fifty_two_high != fifty_two_low else 50}%",
                            'backgroundColor': colors['accent'], 'borderRadius': '3px'
                        })
                    ]) if fifty_two_high and fifty_two_low else html.Div("--", style={'color': colors['text']})
                ])
            ])
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close', line=dict(color=colors['accent'], width=2)))
        # CHANGE: Added range fill beneath the line for visual clarity
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], fill='tozeroy',
                                 fillcolor='rgba(255, 102, 0, 0.08)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.update_layout(title=f"{ticker_symbol} - 6 Month History", yaxis_title="Price ($)", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)

        if not hist.empty:
            current_price = round(hist['Close'].iloc[-1], 2)
            return info_html, fig, current_price, current_price, current_price, current_price

        return info_html, fig, no_update, no_update, no_update, no_update

    except Exception as e:
        err = html.Div([
            html.Div("Could not fetch data", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(f"{e}", style={'color': colors['muted'], 'fontSize': '0.85em'})
        ])
        return err, go.Figure(layout=layout_settings), no_update, no_update, no_update, no_update

# --- SPREAD ANALYSIS CALLBACK ---
# CHANGE: Added z-score to stats, colored stat values, improved stat card layout
@app.callback(
    [Output('spread-norm-chart', 'figure'), Output('spread-ratio-chart', 'figure'), Output('spread-stats-display', 'children')],
    [Input('spread-analyze-btn', 'n_clicks')],
    [State('spread-ticker-a', 'value'), State('spread-ticker-b', 'value'), State('spread-period-slider', 'value')]
)
def update_spread_analysis(n_clicks, ticker_a, ticker_b, slider_val):
    if not ticker_a or not ticker_b:
        return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div()

    selected_period = PERIOD_MAP.get(slider_val, '6mo')

    try:
        df_a = yf.Ticker(ticker_a).history(period=selected_period)['Close']
        df_b = yf.Ticker(ticker_b).history(period=selected_period)['Close']
        df = pd.DataFrame({ticker_a: df_a, ticker_b: df_b}).dropna()

        if df.empty:
            return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div("No overlapping data found.", style={'color': colors['danger']})

        norm_a = (df[ticker_a] / df[ticker_a].iloc[0]) * 100
        norm_b = (df[ticker_b] / df[ticker_b].iloc[0]) * 100

        fig_norm = go.Figure()
        fig_norm.add_trace(go.Scatter(x=df.index, y=norm_a, mode='lines', name=f"{ticker_a}", line=dict(color=colors['accent'], width=2)))
        fig_norm.add_trace(go.Scatter(x=df.index, y=norm_b, mode='lines', name=f"{ticker_b}", line=dict(color=colors['put_text'], width=2)))
        fig_norm.update_layout(title=f"Relative Performance - {selected_period.upper()}", yaxis_title="Normalized Price (100 = Start)", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)

        ratio = df[ticker_a] / df[ticker_b]
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=df.index, y=ratio, mode='lines', name='Ratio', line=dict(color=colors['success'], width=2)))
        fig_ratio.add_hline(y=ratio.mean(), line_dash="dash", line_color="white", annotation_text="Mean")
        # CHANGE: Added +/- 1 std dev bands to ratio chart for mean-reversion context
        fig_ratio.add_hline(y=ratio.mean() + ratio.std(), line_dash="dot", line_color=colors['call_text'], annotation_text="+1σ")
        fig_ratio.add_hline(y=ratio.mean() - ratio.std(), line_dash="dot", line_color=colors['put_text'], annotation_text="-1σ")
        fig_ratio.update_layout(title=f"Ratio ({ticker_a} / {ticker_b})", yaxis_title="Ratio", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)

        corr = df[ticker_a].corr(df[ticker_b])
        curr_ratio = ratio.iloc[-1]

        # CHANGE: Compute z-score to show how far the spread is from its mean
        z_score = (curr_ratio - ratio.mean()) / ratio.std() if ratio.std() > 0 else 0
        z_color = colors['success'] if abs(z_score) < 1 else (colors['put_text'] if abs(z_score) > 2 else colors['accent'])

        # CHANGE: Color-coded correlation (green=high, red=low)
        corr_color = colors['success'] if corr > 0.7 else (colors['put_text'] if corr < 0.3 else colors['text'])

        stats_html = html.Div([
            html.H4("Spread Statistics", style={'color': colors['text'], 'marginBottom': '12px', 'fontSize': '1em'}),
            make_stat_row("Period", selected_period.upper()),
            make_stat_row("Correlation", f"{corr:.3f}", corr_color),
            html.Hr(className='section-divider'),
            make_stat_row("Current Ratio", f"{curr_ratio:.4f}"),
            make_stat_row("Mean Ratio", f"{ratio.mean():.4f}"),
            make_stat_row("Std Dev", f"{ratio.std():.4f}"),
            html.Hr(className='section-divider'),
            # CHANGE: Z-Score indicator with color coding
            make_stat_row("Z-Score", f"{z_score:+.2f}", z_color),
            html.Div(style={'marginTop': '8px'}, children=[
                html.Span(
                    "Near Mean" if abs(z_score) < 1 else ("Extended" if abs(z_score) < 2 else "Extreme"),
                    className=f"badge {'badge-green' if abs(z_score) < 1 else ('badge-orange' if abs(z_score) < 2 else 'badge-red')}"
                )
            ])
        ])

        return fig_norm, fig_ratio, stats_html

    except Exception as e:
        return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div([
            html.Div("Could not fetch data", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(f"{e}", style={'color': colors['muted'], 'fontSize': '0.85em'})
        ])

# --- VOLATILITY SURFACE CALLBACK ---
# CHANGE: Improved info panel with more context about the data
@app.callback(
    [Output('vol-surface-chart', 'figure'), Output('vol-info-display', 'children')],
    [Input('vol-submit-btn', 'n_clicks'), Input('vol-plot-type', 'value')],
    [State('vol-ticker-input', 'value')]
)
def update_vol_surface(n_clicks, plot_type, ticker_symbol):
    if not ticker_symbol:
        return go.Figure(layout=layout_settings), html.Div()

    ticker_symbol = ticker_symbol.upper().strip()

    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options

        if not expirations:
            return go.Figure(layout=layout_settings), html.Div("No options data available for this ticker.", style={'color': colors['danger']})

        expirations = list(expirations)[:8]

        hist = ticker.history(period="1d")
        if hist.empty:
            return go.Figure(layout=layout_settings), html.Div("Could not fetch underlying price.", style={'color': colors['danger']})

        spot_price = hist['Close'].iloc[-1]

        strikes, dtes, ivs = [], [], []
        today = datetime.datetime.now().replace(tzinfo=None)

        for exp in expirations:
            exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_date - today).days
            if dte <= 0: dte = 0.5

            chain = ticker.option_chain(exp)
            calls = chain.calls

            calls = calls[(calls['strike'] >= spot_price * 0.7) & (calls['strike'] <= spot_price * 1.3)]
            calls = calls[(calls['impliedVolatility'] > 0.01) & (calls['volume'] > 0)]

            for _, row in calls.iterrows():
                strikes.append(row['strike'])
                dtes.append(dte)
                ivs.append(row['impliedVolatility'])

        if len(strikes) < 5:
            return go.Figure(layout=layout_settings), html.Div("Not enough liquid options data to plot. Try a more popular ticker.", style={'color': colors['danger']})

        min_strike, max_strike = min(strikes), max(strikes)
        min_dte, max_dte = min(dtes), max(dtes)

        fig = go.Figure()

        if plot_type == 'surface':
            strike_grid = np.linspace(min_strike, max_strike, 40)
            dte_grid = np.linspace(min_dte, max_dte, 40)
            X, Y = np.meshgrid(strike_grid, dte_grid)

            Z = griddata((strikes, dtes), ivs, (X, Y), method='cubic')
            if np.isnan(Z).all():
                 Z = griddata((strikes, dtes), ivs, (X, Y), method='linear')

            fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Jet', colorbar=dict(title="IV")))

        else:
            fig.add_trace(go.Scatter3d(
                x=strikes, y=dtes, z=ivs,
                mode='markers',
                marker=dict(
                    size=4,
                    color=ivs,
                    colorscale='Jet',
                    opacity=0.8,
                    colorbar=dict(title="IV")
                ),
                name='Raw IV'
            ))

        fig.update_layout(
            title=f"{ticker_symbol} Call Implied Volatility ({plot_type.title()})",
            scene=dict(
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.8, y=-1.2, z=1.0)
                ),
                xaxis_title='Strike Price ($)',
                yaxis_title='Days to Expiration (DTE)',
                zaxis_title='Implied Volatility',

                xaxis=dict(
                    backgroundcolor=colors['card_bg'],
                    gridcolor="#333",
                    showbackground=True,
                    range=[min_strike, max_strike]
                ),
                yaxis=dict(
                    backgroundcolor=colors['card_bg'],
                    gridcolor="#333",
                    showbackground=True,
                    range=[min_dte, max_dte]
                ),
                zaxis=dict(
                    backgroundcolor=colors['card_bg'],
                    gridcolor="#333",
                    showbackground=True
                )
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            **layout_settings
        )

        # CHANGE: Enhanced info panel with more context about the loaded data
        info_html = html.Div([
            html.H4("Surface Data", style={'color': colors['text'], 'marginBottom': '10px', 'fontSize': '1em'}),
            make_stat_row("Spot Price", f"${spot_price:.2f}", colors['accent']),
            make_stat_row("Data Points", f"{len(strikes):,}"),
            make_stat_row("Expirations", f"{len(expirations)}"),
            make_stat_row("Strike Range", f"${min_strike:.0f} - ${max_strike:.0f}"),
            make_stat_row("DTE Range", f"{min_dte:.0f} - {max_dte:.0f} days"),
            make_stat_row("IV Range", f"{min(ivs)*100:.1f}% - {max(ivs)*100:.1f}%"),
        ])

        return fig, info_html

    except Exception as e:
        return go.Figure(layout=layout_settings), html.Div([
            html.Div("Could not fetch data", style={'color': colors['danger'], 'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Div(f"{e}", style={'color': colors['muted'], 'fontSize': '0.85em'})
        ])

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

            fig_skew.update_layout(title=f"Live Volatility Skew (Expiry: {target_exp})", xaxis_title="Strike Price ($)", yaxis_title="Implied Volatility (%)", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)

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
            fig_skew.update_layout(title="No Options Data Available for Skew", **layout_settings)

        fig_hv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            row_heights=[0.6, 0.4], subplot_titles=(f"{ticker_symbol} Price", f"{window}-Day Rolling Historical Volatility (HV)"))
        fig_hv.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price', line=dict(color=colors['accent'], width=2)), row=1, col=1)
        fig_hv.add_trace(go.Scatter(x=hist.index, y=hist['HV'], mode='lines', name=f'{window}d HV', line=dict(color=colors['put_text'], width=2)), row=2, col=1)

        fig_hv.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False, **layout_settings)
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
                    "Options Rich" if current_iv and (current_iv - current_hv) > 0 else "Options Cheap",
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

    table = dash_table.DataTable(
        data=rows,
        columns=[{'name': c, 'id': c} for c in ['Ticker', 'IV %', 'HV %', 'VRP', 'IV/HV', 'Skew', 'HV Rank', 'Verdict']],
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
            {'if': {'filter_query': '{Verdict} = "Expensive"'}, 'backgroundColor': '#150000', 'color': colors['put_text']},
            {'if': {'filter_query': '{Verdict} = "Cheap"'}, 'backgroundColor': '#001500', 'color': colors['call_text']},
            {'if': {'column_id': 'Ticker'}, 'fontWeight': 'bold', 'color': colors['accent']},
        ],
        sort_action='native',
    )
    return table, f"Scanned {len(rows)} of {len(tickers)} tickers."


if __name__ == '__main__':
    app.run(debug=True)
