import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
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

colors = {
    'background': '#1e1e1e', 'text': '#e0e0e0', 'card_bg': '#2c2c2c',
    'input_bg': '#3a3a3a', 'call_text': '#69f0ae', 'put_text': '#ff8a80', 
    'accent': '#90caf9', 'success': '#00e676', 'danger': '#ff5252'
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
    fig_spot.add_vline(x=S, line_width=1, line_dash="dash", line_color="#888", annotation_text="Spot")
    
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
    'marginBottom': '20px'
}

CONTENT_STYLE = {
    'flex': '3 1 500px', 
    'backgroundColor': colors['card_bg'], 'padding': '10px', 
    'borderRadius': '10px', 'minHeight': '400px', 'boxSizing': 'border-box',
    'overflow': 'hidden' 
}

FLEX_WRAPPER_STYLE = {
    'display': 'flex', 
    'flexWrap': 'wrap', 
    'gap': '20px', 
    'maxWidth': '1400px', 
    'margin': '0 auto'
}

def make_control_row(label, id_prefix, min_val, max_val, step, default_val):
    return html.Div(style={'marginBottom': '20px'}, children=[
        html.Label(label, style={'color': colors['text'], 'fontWeight': 'bold'}),
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
            dcc.Input(id=f'{id_prefix}-input', type='number', value=default_val, step=step, 
                      style={'width': '70px', 'padding': '5px', 'backgroundColor': colors['input_bg'], 'color': 'white', 'border': '1px solid #555', 'borderRadius': '4px'}),
            html.Div(style={'flex': '1'}, children=[
                dcc.Slider(id=f'{id_prefix}-slider', min=min_val, max=max_val, step=step, value=default_val, marks=None, tooltip={"placement": "bottom", "always_visible": True})
            ])
        ])
    ])

# --- 1. FUNDAMENTAL TAB LAYOUT ---
fundamental_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.Div([
                html.H3("Stock Search", style={'color': colors['accent']}),
                dcc.Input(id='fund-ticker-input', type='text', value=DEFAULT_TICKER, placeholder="e.g. AAPL", 
                          style={'width': '100%', 'boxSizing': 'border-box', 'padding': '10px', 'backgroundColor': colors['input_bg'], 'color': 'white', 'border': '1px solid #555', 'borderRadius': '4px'}),
                html.Button('Analyze', id='fund-submit-btn', n_clicks=0, 
                            style={'marginTop': '10px', 'width': '100%', 'boxSizing': 'border-box', 'padding': '10px', 'backgroundColor': colors['accent'], 'border': 'none', 'borderRadius': '4px', 'fontWeight': 'bold', 'cursor': 'pointer'}),
                html.Hr(style={'borderColor': '#555'}),
                html.Div(id='fund-info-display') 
            ])
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(dcc.Graph(id='fund-price-chart', style={'height': '60vh', 'minHeight': '400px'}), type='circle')
        ])
    ])
])

# --- 2. BLACK-SCHOLES TAB LAYOUT ---
bs_layout = html.Div([
    html.Div(style=FLEX_WRAPPER_STYLE, children=[
        html.Div(style=SIDEBAR_STYLE, children=[
            html.H3("Option Inputs", style={'color': colors['accent']}),
            make_control_row("Spot Price ($)", "spot", 0, 800, 0.01, initial_spot),
            make_control_row("Strike Price ($)", "strike", 0, 800, 0.01, initial_strike),
            make_control_row("Time (Years)", "time", 0.01, 5, 0.01, DEFAULT_TIME),
            make_control_row("Volatility (σ)", "vol", 0.01, 1.5, 0.01, DEFAULT_VOL),
            make_control_row("Risk-Free Rate (r)", "rate", 0.0, 0.2, 0.001, DEFAULT_RATE),
        ]),
        html.Div(style=CONTENT_STYLE, children=[
             html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '20px'}, children=[
                html.Div(style={'flex': 1, 'backgroundColor': colors['card_bg'], 'padding': '10px', 'borderRadius': '8px', 'textAlign': 'center', 'border': f"1px solid {colors['call_text']}"}, children=[
                    html.H4("Call Price", style={'margin': '0', 'fontSize': '1rem', 'color': colors['call_text']}),
                    html.H2(id='call-price-display', style={'margin': '5px 0', 'color': colors['text']})
                ]),
                html.Div(style={'flex': 1, 'backgroundColor': colors['card_bg'], 'padding': '10px', 'borderRadius': '8px', 'textAlign': 'center', 'border': f"1px solid {colors['put_text']}"}, children=[
                    html.H4("Put Price", style={'margin': '0', 'fontSize': '1rem', 'color': colors['put_text']}),
                    html.H2(id='put-price-display', style={'margin': '5px 0', 'color': colors['text']})
                ]),
            ]),
            html.Div(style={'backgroundColor': colors['card_bg'], 'borderRadius': '10px'}, children=[
                dcc.Tabs(style={'color': colors['text']}, children=[
                    dcc.Tab(label='Payoff', style={'backgroundColor': colors['card_bg'], 'color': '#888'}, selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
                        dcc.Graph(id='payoff-graph', style={'height': '60vh', 'minHeight': '400px'})
                    ]),
                    dcc.Tab(label='Greeks', style={'backgroundColor': colors['card_bg'], 'color': '#888'}, selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
                        dcc.Graph(id='greeks-graph', style={'height': '60vh', 'minHeight': '450px'})
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
            html.H3("Spread Inputs", style={'color': colors['accent']}),
            html.Label("Stock A (Numerator)", style={'color': colors['text'], 'fontWeight': 'bold'}),
            dcc.Input(id='spread-ticker-a', type='text', value=DEFAULT_SPREAD_A, placeholder="e.g. KO", 
                      style={'width': '100%', 'padding': '10px', 'backgroundColor': colors['input_bg'], 'color': 'white', 'border': '1px solid #555', 'borderRadius': '4px', 'marginBottom': '10px', 'boxSizing': 'border-box'}),
            
            html.Label("Stock B (Denominator)", style={'color': colors['text'], 'fontWeight': 'bold'}),
            dcc.Input(id='spread-ticker-b', type='text', value=DEFAULT_SPREAD_B, placeholder="e.g. PEP", 
                      style={'width': '100%', 'padding': '10px', 'backgroundColor': colors['input_bg'], 'color': 'white', 'border': '1px solid #555', 'borderRadius': '4px', 'marginBottom': '20px', 'boxSizing': 'border-box'}),
            
            html.Label("Lookback Period", style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            html.Div(style={'padding': '0 10px 20px 10px'}, children=[
                dcc.Slider(
                    id='spread-period-slider',
                    min=0, max=6, step=1,
                    value=2,
                    marks={0: '1M', 1: '3M', 2: '6M', 3: '1Y', 4: '2Y', 5: '5Y', 6: 'MAX'},
                )
            ]),

            html.Button('Analyze Spread', id='spread-analyze-btn', n_clicks=0, 
                        style={'width': '100%', 'padding': '10px', 'backgroundColor': colors['accent'], 'border': 'none', 'borderRadius': '4px', 'fontWeight': 'bold', 'cursor': 'pointer'}),
            html.Hr(style={'borderColor': '#555'}),
            
            html.Div(id='spread-stats-display')
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Tabs(style={'color': colors['text']}, children=[
                dcc.Tab(label='Norm Perf.', style={'backgroundColor': colors['card_bg'], 'color': '#888'}, selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
                    dcc.Loading(dcc.Graph(id='spread-norm-chart', style={'height': '60vh', 'minHeight': '400px'}), type='circle')
                ]),
                dcc.Tab(label='Spread Ratio', style={'backgroundColor': colors['card_bg'], 'color': '#888'}, selected_style={'backgroundColor': colors['card_bg'], 'color': colors['accent'], 'borderTop': f"2px solid {colors['accent']}"}, children=[
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
            html.H3("Vol Surface Inputs", style={'color': colors['accent']}),
            html.P("Generates a 3D Implied Volatility Surface using current Options Chain data.", style={'color': colors['text'], 'fontSize': '0.9em'}),
            
            dcc.Input(id='vol-ticker-input', type='text', value=DEFAULT_TICKER, placeholder="e.g. SPY", 
                      style={'width': '100%', 'boxSizing': 'border-box', 'padding': '10px', 'backgroundColor': colors['input_bg'], 'color': 'white', 'border': '1px solid #555', 'borderRadius': '4px'}),
            
            html.Label("Plot Type", style={'color': colors['text'], 'fontWeight': 'bold', 'marginTop': '15px', 'display': 'block'}),
            dcc.RadioItems(
                id='vol-plot-type',
                options=[
                    {'label': ' Surface (Interpolated)', 'value': 'surface'},
                    {'label': ' Scatter (Raw Data)', 'value': 'scatter'}
                ],
                value='surface',  # Default selection
                labelStyle={'display': 'block', 'color': colors['text'], 'marginBottom': '5px', 'cursor': 'pointer'},
                style={'marginBottom': '10px'}
            ),
            
            html.Button('Fetch Options Data', id='vol-submit-btn', n_clicks=0, 
                        style={'marginTop': '10px', 'width': '100%', 'boxSizing': 'border-box', 'padding': '10px', 'backgroundColor': colors['accent'], 'border': 'none', 'borderRadius': '4px', 'fontWeight': 'bold', 'cursor': 'pointer'}),
            html.Hr(style={'borderColor': '#555'}),
            html.Div(id='vol-info-display')
        ]),
        html.Div(style=CONTENT_STYLE, children=[
            dcc.Loading(dcc.Graph(id='vol-surface-chart', style={'height': '70vh', 'minHeight': '500px'}), type='circle')
        ])
    ])
])

# --- APP LAYOUT (With Padding) ---
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '10px', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1("Equity Research", style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '1.5rem'}),
    
    # Navigation Tabs with Padding
    dcc.Tabs(id='main-tabs', value='tab-fundamental', 
             style={'marginTop': '20px', 'marginBottom': '20px'}, 
             children=[
                dcc.Tab(label='Fundamentals', value='tab-fundamental', 
                        style={'backgroundColor': colors['card_bg'], 'color': '#888', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'}, 
                        selected_style={'backgroundColor': '#444', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Black-Scholes', value='tab-bs', 
                        style={'backgroundColor': colors['card_bg'], 'color': '#888', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'}, 
                        selected_style={'backgroundColor': '#444', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Spread', value='tab-spread', 
                        style={'backgroundColor': colors['card_bg'], 'color': '#888', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'}, 
                        selected_style={'backgroundColor': '#444', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
                dcc.Tab(label='Vol Surface', value='tab-vol', 
                        style={'backgroundColor': colors['card_bg'], 'color': '#888', 'border': 'none', 'padding': '12px', 'fontWeight': 'bold'}, 
                        selected_style={'backgroundColor': '#444', 'color': colors['accent'], 'borderTop': f"3px solid {colors['accent']}", 'padding': '12px'}),
    ]),
    
    html.Div(id='fund-content-wrapper', children=fundamental_layout, style={'display': 'block'}),
    html.Div(id='bs-content-wrapper', children=bs_layout, style={'display': 'none'}),
    html.Div(id='spread-content-wrapper', children=spread_layout, style={'display': 'none'}),
    html.Div(id='vol-content-wrapper', children=vol_surface_layout, style={'display': 'none'})
])

# -----------------------------------------------------------------------------
# 5. CALLBACKS
# -----------------------------------------------------------------------------

# Tab Visibility Toggle
@app.callback(
    [Output('fund-content-wrapper', 'style'), Output('bs-content-wrapper', 'style'), 
     Output('spread-content-wrapper', 'style'), Output('vol-content-wrapper', 'style')],
    [Input('main-tabs', 'value')]
)
def toggle_tabs(tab_value):
    fund_style, bs_style, spread_style, vol_style = [{'display': 'none'}] * 4
    if tab_value == 'tab-fundamental': fund_style = {'display': 'block'}
    elif tab_value == 'tab-bs': bs_style = {'display': 'block'}
    elif tab_value == 'tab-spread': spread_style = {'display': 'block'}
    elif tab_value == 'tab-vol': vol_style = {'display': 'block'}
    return fund_style, bs_style, spread_style, vol_style

# Main Fundamental Analysis Callback
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
        peg_display = f"{peg}" if peg is not None else "N/A"
        pe_display = f"{pe}" if pe is not None else "N/A"

        info_html = html.Div([
            html.H2(f"{info.get('shortName', ticker_symbol)}", style={'marginTop': 0, 'color': colors['accent']}),
            html.P(f"Sector: {info.get('sector', 'N/A')}", style={'color': colors['text']}),
            html.P(f"Industry: {info.get('industry', 'N/A')}", style={'color': colors['text']}),
            html.P(f"P/E Ratio (Trailing): {pe_display}", style={'color': colors['text']}),
            html.P(f"Forward PEG Ratio: {peg_display}", style={'color': colors['text']}),
            html.P(f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}", style={'color': colors['text']}),
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close', line=dict(color=colors['accent'])))
        fig.update_layout(title=f"{ticker_symbol} - 6 Month History", yaxis_title="Price ($)", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)
        
        if not hist.empty:
            current_price = round(hist['Close'].iloc[-1], 2)
            return info_html, fig, current_price, current_price, current_price, current_price
        
        return info_html, fig, no_update, no_update, no_update, no_update

    except Exception as e:
        err = html.Div(f"Error: {e}", style={'color': 'red'})
        return err, go.Figure(layout=layout_settings), no_update, no_update, no_update, no_update

# --- SPREAD ANALYSIS CALLBACK ---
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
            return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div("No overlapping data found.")

        norm_a = (df[ticker_a] / df[ticker_a].iloc[0]) * 100
        norm_b = (df[ticker_b] / df[ticker_b].iloc[0]) * 100
        
        fig_norm = go.Figure()
        fig_norm.add_trace(go.Scatter(x=df.index, y=norm_a, mode='lines', name=f"{ticker_a} (Norm)", line=dict(color=colors['accent'])))
        fig_norm.add_trace(go.Scatter(x=df.index, y=norm_b, mode='lines', name=f"{ticker_b} (Norm)", line=dict(color=colors['put_text'])))
        fig_norm.update_layout(title=f"Relative Perf. - {selected_period.upper()}", yaxis_title="Norm Price", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)
        
        ratio = df[ticker_a] / df[ticker_b]
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=df.index, y=ratio, mode='lines', name='Ratio', line=dict(color=colors['success'])))
        fig_ratio.add_hline(y=ratio.mean(), line_dash="dash", line_color="white", annotation_text="Mean")
        fig_ratio.update_layout(title=f"Ratio ({ticker_a} / {ticker_b})", yaxis_title="Ratio", margin=dict(l=20, r=20, t=40, b=20), **layout_settings)
        
        corr = df[ticker_a].corr(df[ticker_b])
        curr_ratio = ratio.iloc[-1]
        
        stats_html = html.Div([
            html.H4("Spread Statistics", style={'color': colors['text']}),
            html.P(f"Period: {selected_period.upper()}", style={'color': colors['text']}),
            html.P(f"Correlation: {corr:.2f}", style={'color': colors['text']}),
            html.P(f"Current Ratio: {curr_ratio:.4f}", style={'color': colors['text']}),
            html.P(f"Mean Ratio: {ratio.mean():.4f}", style={'color': colors['text']})
        ])
        
        return fig_norm, fig_ratio, stats_html
        
    except Exception as e:
        return go.Figure(layout=layout_settings), go.Figure(layout=layout_settings), html.Div(f"Error: {e}", style={'color': 'red'})

# --- VOLATILITY SURFACE CALLBACK ---
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
            return go.Figure(layout=layout_settings), html.Div("No options data available.", style={'color': colors['danger']})

        # To prevent API timeouts, limit to the first 8 expirations
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
            
            # Filter out extreme deep in/out of money and illiquid strikes
            calls = calls[(calls['strike'] >= spot_price * 0.7) & (calls['strike'] <= spot_price * 1.3)]
            calls = calls[(calls['impliedVolatility'] > 0.01) & (calls['volume'] > 0)]
            
            for _, row in calls.iterrows():
                strikes.append(row['strike'])
                dtes.append(dte)
                ivs.append(row['impliedVolatility'])
                
        if len(strikes) < 5:
            return go.Figure(layout=layout_settings), html.Div("Not enough liquid options data to plot.", style={'color': colors['danger']})

        # --- DYNAMIC RANGE CALCULATIONS ---
        min_strike, max_strike = min(strikes), max(strikes)
        min_dte, max_dte = min(dtes), max(dtes)

        fig = go.Figure()

        # --- DYNAMIC PLOTTING LOGIC ---
        if plot_type == 'surface':
            # Interpolation to create the 2D meshgrid for go.Surface
            strike_grid = np.linspace(min_strike, max_strike, 40)
            dte_grid = np.linspace(min_dte, max_dte, 40)
            X, Y = np.meshgrid(strike_grid, dte_grid)
            
            Z = griddata((strikes, dtes), ivs, (X, Y), method='cubic')
            if np.isnan(Z).all():
                 Z = griddata((strikes, dtes), ivs, (X, Y), method='linear')

            fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', colorbar=dict(title="IV")))
        
        else: # plot_type == 'scatter'
            # Plot the raw un-interpolated data
            fig.add_trace(go.Scatter3d(
                x=strikes, y=dtes, z=ivs,
                mode='markers',
                marker=dict(
                    size=4,
                    color=ivs,                
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="IV")
                ),
                name='Raw IV'
            ))
        
        # Standardize layout for both chart types
        fig.update_layout(
            title=f"{ticker_symbol} Call Implied Volatility ({plot_type.title()})",
            scene=dict(
                # --- NEW: Custom Default Camera Angle ---
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.8, y=-1.2, z=1.0) # Adjusted for a clear view of ascending strikes
                ),
                xaxis_title='Strike Price ($)',
                yaxis_title='Days to Expiration (DTE)',
                zaxis_title='Implied Volatility',
                
                # --- NEW: Dynamic ascending ranges ---
                xaxis=dict(
                    backgroundcolor=colors['card_bg'], 
                    gridcolor="#555", 
                    showbackground=True, 
                    range=[min_strike, max_strike]
                ),
                yaxis=dict(
                    backgroundcolor=colors['card_bg'], 
                    gridcolor="#555", 
                    showbackground=True, 
                    range=[min_dte, max_dte]
                ),
                zaxis=dict(
                    backgroundcolor=colors['card_bg'], 
                    gridcolor="#555", 
                    showbackground=True
                )
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            **layout_settings
        )
        
        info_html = html.Div([
            html.P(f"Data points: {len(strikes)}", style={'color': colors['text']}),
            html.P(f"Spot Reference: ${spot_price:.2f}", style={'color': colors['text']})
        ])
        
        return fig, info_html

    except Exception as e:
        return go.Figure(layout=layout_settings), html.Div(f"Error: {e}", style={'color': 'red'})

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

@app.callback(
    [Output('call-price-display', 'children'), Output('put-price-display', 'children'),
     Output('payoff-graph', 'figure'), Output('greeks-graph', 'figure')],
    [Input('spot-input', 'value'), Input('strike-input', 'value'), Input('time-input', 'value'), Input('vol-input', 'value'), Input('rate-input', 'value')]
)
def calc_bs(S, K, T, r, sigma):
    if None in [S, K, T, r, sigma]: return no_update, no_update, no_update, no_update
    call = black_scholes(S, K, T, r, sigma, 'call')
    put = black_scholes(S, K, T, r, sigma, 'put')
    fig_spot, fig_greeks = get_bs_charts(S, K, T, r, sigma)
    return f"${call:.2f}", f"${put:.2f}", fig_spot, fig_greeks

if __name__ == '__main__':
    app.run(debug=True)