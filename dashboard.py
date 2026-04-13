"""
Mortgage AI Decision System Dashboard
Fintech UI/UX - Deep Navy Theme with Cyan Accents
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import numpy as np
from datetime import datetime
import json

# API Configuration
API_BASE = "http://localhost:8000"

# Design System Colors
COLORS = {
    'bg': '#0A0E1A',
    'card': '#0F1629',
    'card_glass': 'rgba(15, 22, 41, 0.8)',
    'accent': '#00D4FF',
    'success': '#00FF88',
    'danger': '#FF4444',
    'warning': '#FFB800',
    'border': 'rgba(0, 212, 255, 0.2)',
    'text': '#FFFFFF',
    'text_muted': '#94A3B8',
    'text_dark': '#0F172A'
}

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

app.title = "Mortgage AI Decision System"

# Custom styles
custom_styles = {
    'bg': {'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': 'Inter, -apple-system, sans-serif'},
    'header': {
        'backgroundColor': COLORS['card'],
        'borderBottom': f'1px solid {COLORS["border"]}',
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'height': '60px',
        'zIndex': 1000,
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'padding': '0 24px'
    },
    'sidebar': {
        'backgroundColor': COLORS['card'],
        'borderRight': f'1px solid {COLORS["border"]}',
        'position': 'fixed',
        'top': '60px',
        'left': 0,
        'width': '28%',
        'height': 'calc(100vh - 60px)',
        'overflowY': 'auto',
        'padding': '24px',
        'zIndex': 100
    },
    'content': {
        'marginLeft': '28%',
        'marginTop': '60px',
        'padding': '24px',
        'minHeight': 'calc(100vh - 60px)',
        'backgroundColor': COLORS['bg']
    },
    'input_group': {'marginBottom': '20px'},
    'label': {'color': COLORS['accent'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '8px'},
    'slider': {'marginTop': '10px'},
    'number_input': {
        'backgroundColor': COLORS['bg'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': '8px',
        'color': COLORS['text'],
        'padding': '10px 14px',
        'width': '100%',
        'marginTop': '8px'
    },
    'analyze_btn': {
        'background': f'linear-gradient(135deg, {COLORS["accent"]} 0%, #00B8E6 100%)',
        'border': 'none',
        'borderRadius': '12px',
        'color': COLORS['text_dark'],
        'fontSize': '18px',
        'fontWeight': 'bold',
        'padding': '16px 32px',
        'width': '100%',
        'cursor': 'pointer',
        'boxShadow': f'0 0 30px rgba(0, 212, 255, 0.4)',
        'marginTop': '24px',
        'textTransform': 'uppercase',
        'letterSpacing': '2px'
    },
    'card': {
        'backgroundColor': COLORS['card_glass'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': '12px',
        'padding': '20px',
        'marginBottom': '20px'
    },
    'metric_card': {
        'backgroundColor': COLORS['card_glass'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': '12px',
        'padding': '20px',
        'textAlign': 'center',
        'height': '100%'
    },
    'ai_card': {
        'backgroundColor': COLORS['card_glass'],
        'border': f'1px solid {COLORS["border"]}',
        'borderLeft': f'4px solid {COLORS["accent"]}',
        'borderRadius': '12px',
        'padding': '20px',
        'marginBottom': '20px'
    }
}

def create_input_with_slider(label, id_prefix, min_val, max_val, default_val, step=1, prefix="", suffix=""):
    """Create input group with label, number input, and slider"""
    return html.Div([
        html.Label(label, style=custom_styles['label']),
        dbc.Input(
            id=f'{id_prefix}-number',
            type='number',
            value=default_val,
            min=min_val,
            max=max_val,
            step=step,
            style=custom_styles['number_input']
        ),
        dcc.Slider(
            id=f'{id_prefix}-slider',
            min=min_val,
            max=max_val,
            step=step,
            value=default_val,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
            className='cyan-slider'
        )
    ], style=custom_styles['input_group'])

# Header component
header = html.Div([
    html.Div([
        html.Span("MORTGAGE", style={'color': COLORS['text'], 'fontSize': '24px', 'fontWeight': 'bold'}),
        html.Span(" AI", style={'color': COLORS['accent'], 'fontSize': '24px', 'fontWeight': 'bold'})
    ]),
    html.Div([
        html.Div(id='status-indicator', children=[
            html.Span("●", style={'color': COLORS['success'], 'marginRight': '8px'}),
            html.Span("System Online", style={'color': COLORS['text_muted'], 'fontSize': '14px'})
        ])
    ])
], style=custom_styles['header'])

# Left Sidebar
sidebar = html.Div([
    html.H3("Loan Parameters", style={'color': COLORS['accent'], 'fontSize': '18px', 'marginBottom': '24px'}),

    create_input_with_slider("Monthly Income (Rs)", "income", 10000, 500000, 50000, step=1000),
    create_input_with_slider("Loan Amount (Rs)", "loan-amount", 100000, 5000000, 1000000, step=10000),
    create_input_with_slider("Interest Rate (%)", "interest", 1.0, 20.0, 8.5, step=0.1),
    create_input_with_slider("Loan Term (months)", "term", 12, 360, 240, step=12),
    create_input_with_slider("Credit Score", "credit", 300, 850, 700, step=1),
    create_input_with_slider("Existing Loans", "existing", 0, 10, 0, step=1),

    html.Button(
        "ANALYZE",
        id='analyze-btn',
        n_clicks=0,
        style=custom_styles['analyze_btn']
    ),

    html.Div(id='error-message', style={'color': COLORS['danger'], 'marginTop': '16px', 'fontSize': '14px'})

], style=custom_styles['sidebar'])

# Section 1: Decision Banner
def create_decision_banner(decision="waiting"):
    if decision == "APPROVED":
        bg_color = COLORS['success']
        text_color = COLORS['text_dark']
        text = "✓ APPROVED"
    elif decision == "REJECTED":
        bg_color = COLORS['danger']
        text_color = COLORS['text']
        text = "✗ REJECTED"
    elif decision == "CONDITIONAL":
        bg_color = COLORS['warning']
        text_color = COLORS['text_dark']
        text = "⚠ CONDITIONAL"
    else:
        bg_color = COLORS['card']
        text_color = COLORS['text_muted']
        text = "Click Analyze to Begin"

    return html.Div([
        html.H1(text, style={
            'color': text_color,
            'fontSize': '32px',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'lineHeight': '80px',
            'margin': 0
        })
    ], style={
        'backgroundColor': bg_color,
        'borderRadius': '12px',
        'height': '80px',
        'marginBottom': '20px',
        'transition': 'all 0.3s ease'
    })

# Section 2: Metric Cards
def create_metric_cards(emi=None, risk=None, default_prob=None, approval_prob=None):
    # Determine colors based on values
    risk_color = COLORS['text_muted']
    if risk == "LOW":
        risk_color = COLORS['success']
    elif risk == "MEDIUM":
        risk_color = COLORS['warning']
    elif risk == "HIGH":
        risk_color = COLORS['danger']

    default_color = COLORS['text_muted']
    if default_prob is not None:
        if default_prob > 35:
            default_color = COLORS['danger']
        elif default_prob > 15:
            default_color = COLORS['warning']
        else:
            default_color = COLORS['success']

    approval_color = COLORS['text_muted']
    if approval_prob is not None:
        if approval_prob > 60:
            approval_color = COLORS['success']
        else:
            approval_color = COLORS['warning']

    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("MONTHLY EMI", style={'color': COLORS['accent'], 'fontSize': '12px', 'marginBottom': '8px'}),
                html.Div(
                    f"Rs {emi:,.0f}" if emi else "--",
                    style={'color': COLORS['accent'], 'fontSize': '28px', 'fontWeight': 'bold'}
                )
            ], style=custom_styles['metric_card'])
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div("RISK LEVEL", style={'color': COLORS['accent'], 'fontSize': '12px', 'marginBottom': '8px'}),
                html.Div(
                    risk or "--",
                    style={'color': risk_color, 'fontSize': '28px', 'fontWeight': 'bold'}
                )
            ], style=custom_styles['metric_card'])
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div("DEFAULT PROBABILITY", style={'color': COLORS['accent'], 'fontSize': '12px', 'marginBottom': '8px'}),
                html.Div(
                    f"{default_prob:.1f}%" if default_prob else "--",
                    style={'color': default_color, 'fontSize': '28px', 'fontWeight': 'bold'}
                )
            ], style=custom_styles['metric_card'])
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div("APPROVAL PROBABILITY", style={'color': COLORS['accent'], 'fontSize': '12px', 'marginBottom': '8px'}),
                html.Div(
                    f"{approval_prob:.1f}%" if approval_prob else "--",
                    style={'color': approval_color, 'fontSize': '28px', 'fontWeight': 'bold'}
                )
            ], style=custom_styles['metric_card'])
        ], width=3)
    ], className="g-3", style={'marginBottom': '20px'})

# Section 3: AI Advice
def create_ai_advice(advice=""):
    return html.Div([
        html.Div("AI ADVICE", style={'color': COLORS['accent'], 'fontSize': '12px', 'letterSpacing': '2px', 'marginBottom': '12px'}),
        html.Div(advice or "AI advice will appear here after analysis...", style={'color': COLORS['text'], 'fontSize': '16px', 'lineHeight': '1.6'})
    ], style=custom_styles['ai_card'], id='ai-advice-section')

# Section 4: Monte Carlo 3D Chart
def create_monte_carlo_chart(income, loan_amount, interest, term):
    """Generate 500 random scenarios for Monte Carlo simulation"""
    np.random.seed(42)
    n_scenarios = 500

    # Generate variations
    income_var = income * np.random.uniform(0.7, 1.3, n_scenarios)
    rate_var = interest * np.random.uniform(0.8, 1.5, n_scenarios)

    # Calculate EMI for each scenario
    r = rate_var / 100 / 12
    n = np.full(n_scenarios, term)
    emi = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)

    # EMI ratio
    emi_ratio = emi / income_var

    # Color based on threshold
    colors = [COLORS['success'] if ratio < 0.4 else COLORS['danger'] for ratio in emi_ratio]

    fig = go.Figure(data=[go.Scatter3d(
        x=income_var / 1000,  # in thousands
        y=rate_var,
        z=emi_ratio,
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            opacity=0.7
        ),
        text=[f"Income: Rs {inc:,.0f}<br>Rate: {rate:.2f}%<br>EMI Ratio: {ratio:.2f}" for inc, rate, ratio in zip(income_var, rate_var, emi_ratio)],
        hovertemplate='%{text}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text='Monte Carlo Risk Surface (500 Scenarios)', font=dict(color=COLORS['accent'], size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title='Income (Rs 1000s)', titlefont=dict(color=COLORS['accent']), tickfont=dict(color=COLORS['text_muted']), backgroundcolor='rgba(0,0,0,0)', gridcolor=COLORS['border']),
            yaxis=dict(title='Interest Rate (%)', titlefont=dict(color=COLORS['accent']), tickfont=dict(color=COLORS['text_muted']), backgroundcolor='rgba(0,0,0,0)', gridcolor=COLORS['border']),
            zaxis=dict(title='EMI Ratio', titlefont=dict(color=COLORS['accent']), tickfont=dict(color=COLORS['text_muted']), backgroundcolor='rgba(0,0,0,0)', gridcolor=COLORS['border']),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        height=450
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# Section 5: Sensitivity Analysis
def create_sensitivity_chart(base_income, loan_amount, interest, term, credit_score, existing_loans):
    """Call API 9 times with varying income levels"""
    income_levels = np.linspace(base_income * 0.5, base_income * 1.5, 9)
    default_probs = []

    for inc in income_levels:
        try:
            response = requests.post(
                f"{API_BASE}/analyze",
                json={
                    "monthly_income": float(inc),
                    "loan_amount": float(loan_amount),
                    "interest_rate": float(interest),
                    "loan_term": int(term),
                    "credit_score": int(credit_score),
                    "existing_loans": int(existing_loans)
                },
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                default_probs.append(data.get('default_probability', 0))
            else:
                default_probs.append(None)
        except Exception:
            default_probs.append(None)

    # Filter out None values
    valid_data = [(inc, prob) for inc, prob in zip(income_levels, default_probs) if prob is not None]

    if not valid_data:
        return html.Div("Unable to load sensitivity data", style={'color': COLORS['text_muted'], 'textAlign': 'center', 'padding': '40px'})

    incomes, probs = zip(*valid_data)

    fig = go.Figure()

    # Sensitivity line
    fig.add_trace(go.Scatter(
        x=[inc/1000 for inc in incomes],
        y=probs,
        mode='lines+markers',
        name='Default Probability',
        line=dict(color=COLORS['accent'], width=3),
        marker=dict(size=8, color=COLORS['accent'])
    ))

    # Danger zone line
    fig.add_hline(y=35, line_dash="dash", line_color=COLORS['danger'], line_width=2, annotation_text="Danger Zone (35%)", annotation_position="right")

    # Safe zone line
    fig.add_hline(y=15, line_dash="dash", line_color=COLORS['success'], line_width=2, annotation_text="Safe Zone (15%)", annotation_position="right")

    fig.update_layout(
        title=dict(text='Sensitivity: Income vs Default Probability', font=dict(color=COLORS['accent'], size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Monthly Income (Rs 1000s)', titlefont=dict(color=COLORS['accent']), tickfont=dict(color=COLORS['text_muted']), gridcolor=COLORS['border']),
        yaxis=dict(title='Default Probability (%)', titlefont=dict(color=COLORS['accent']), tickfont=dict(color=COLORS['text_muted']), gridcolor=COLORS['border'], range=[0, 100]),
        margin=dict(l=60, r=40, t=40, b=40),
        showlegend=False,
        height=350
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# Section 6: Recent Decisions Table
def create_history_table():
    try:
        response = requests.get(f"{API_BASE}/history", timeout=5)
        if response.status_code == 200:
            history = response.json()
        else:
            history = []
    except Exception:
        history = []

    if not history:
        return html.Div("No history available. Run an analysis to see results.", style={'color': COLORS['text_muted'], 'textAlign': 'center', 'padding': '40px'})

    # Take last 10 entries
    history = history[-10:]

    # Create table
    rows = []
    for entry in history:
        decision = entry.get('decision', '')
        row_color = COLORS['text_muted']
        if decision == 'APPROVED' or decision == 'APPROVE':
            row_color = COLORS['success']
        elif decision == 'REJECTED' or decision == 'REJECT':
            row_color = COLORS['danger']
        elif decision == 'CONDITIONAL':
            row_color = COLORS['warning']

        rows.append(html.Tr([
            html.Td(entry.get('timestamp', 'N/A')[:19], style={'color': COLORS['text_muted'], 'padding': '12px', 'borderBottom': f'1px solid {COLORS["border"]}'}),
            html.Td(f"Rs {entry.get('monthly_income', 0):,.0f}", style={'color': COLORS['text'], 'padding': '12px', 'borderBottom': f'1px solid {COLORS["border"]}'}),
            html.Td(f"Rs {entry.get('loan_amount', 0):,.0f}", style={'color': COLORS['text'], 'padding': '12px', 'borderBottom': f'1px solid {COLORS["border"]}'}),
            html.Td(f"Rs {entry.get('emi', 0):,.0f}", style={'color': COLORS['text'], 'padding': '12px', 'borderBottom': f'1px solid {COLORS["border"]}'}),
            html.Td(entry.get('decision', 'N/A'), style={'color': row_color, 'fontWeight': 'bold', 'padding': '12px', 'borderBottom': f'1px solid {COLORS["border"]}'}),
            html.Td(entry.get('risk_level', 'N/A'), style={'color': COLORS['text'], 'padding': '12px', 'borderBottom': f'1px solid {COLORS["border"]}'})
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Time", style={'color': COLORS['accent'], 'textAlign': 'left', 'padding': '12px', 'borderBottom': f'2px solid {COLORS["accent"]}'}),
            html.Th("Income", style={'color': COLORS['accent'], 'textAlign': 'left', 'padding': '12px', 'borderBottom': f'2px solid {COLORS["accent"]}'}),
            html.Th("Loan Amount", style={'color': COLORS['accent'], 'textAlign': 'left', 'padding': '12px', 'borderBottom': f'2px solid {COLORS["accent"]}'}),
            html.Th("EMI", style={'color': COLORS['accent'], 'textAlign': 'left', 'padding': '12px', 'borderBottom': f'2px solid {COLORS["accent"]}'}),
            html.Th("Decision", style={'color': COLORS['accent'], 'textAlign': 'left', 'padding': '12px', 'borderBottom': f'2px solid {COLORS["accent"]}'}),
            html.Th("Risk", style={'color': COLORS['accent'], 'textAlign': 'left', 'padding': '12px', 'borderBottom': f'2px solid {COLORS["accent"]}'})
        ])),
        html.Tbody(rows)
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

    return html.Div([
        html.H4("Recent Decisions", style={'color': COLORS['accent'], 'marginBottom': '16px'}),
        html.Div(table, style={'maxHeight': '300px', 'overflowY': 'auto'})
    ], style=custom_styles['card'])

# Section 7: Loan Comparison
def create_comparison_chart():
    try:
        response = requests.get(f"{API_BASE}/compare", timeout=5)
        if response.status_code == 200:
            data = response.json()
        else:
            data = None
    except Exception:
        data = None

    if not data:
        return html.Div("Comparison data unavailable", style={'color': COLORS['text_muted'], 'textAlign': 'center', 'padding': '40px'})

    # Extract data
    categories = ['LOW', 'MEDIUM', 'HIGH']
    emi_values = [data.get('low', {}).get('emi', 0), data.get('medium', {}).get('emi', 0), data.get('high', {}).get('emi', 0)]
    default_values = [data.get('low', {}).get('default_probability', 0), data.get('medium', {}).get('default_probability', 0), data.get('high', {}).get('default_probability', 0)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # EMI bars
    fig.add_trace(
        go.Bar(x=categories, y=emi_values, name='EMI (Rs)', marker_color=COLORS['accent']),
        secondary_y=False
    )

    # Default probability bars
    fig.add_trace(
        go.Bar(x=categories, y=default_values, name='Default Probability (%)', marker_color=COLORS['danger']),
        secondary_y=True
    )

    fig.update_layout(
        title=dict(text='Loan Amount Comparison', font=dict(color=COLORS['accent'], size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLORS['text'])),
        margin=dict(l=60, r=60, t=80, b=40),
        height=350
    )

    fig.update_yaxes(title_text="EMI (Rs)", titlefont=dict(color=COLORS['accent']), tickfont=dict(color=COLORS['text_muted']), gridcolor=COLORS['border'], secondary_y=False)
    fig.update_yaxes(title_text="Default Probability (%)", titlefont=dict(color=COLORS['danger']), tickfont=dict(color=COLORS['text_muted']), secondary_y=True)
    fig.update_xaxes(tickfont=dict(color=COLORS['text_muted']))

    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# Main Content Area
content = html.Div([
    # Section 1: Decision Banner
    html.Div(id='decision-banner'),

    # Section 2: Metric Cards
    html.Div(id='metric-cards'),

    # Section 3: AI Advice
    html.Div(id='ai-advice'),

    # Section 4: Monte Carlo
    html.Div([
        html.Div(id='monte-carlo-chart')
    ], style=custom_styles['card']),

    # Section 5: Sensitivity
    html.Div([
        html.Div(id='sensitivity-chart')
    ], style=custom_styles['card']),

    # Section 6: History Table
    html.Div(id='history-table'),

    # Section 7: Comparison
    html.Div([
        html.Div(id='comparison-chart')
    ], style=custom_styles['card']),

    # Store for results
    dcc.Store(id='analysis-results')

], style=custom_styles['content'])

# App layout
app.layout = html.Div([header, sidebar, content], style=custom_styles['bg'])

# Callback: Sync slider and number inputs
for prefix in ['income', 'loan-amount', 'interest', 'term', 'credit', 'existing']:
    @app.callback(
        Output(f'{prefix}-number', 'value'),
        Output(f'{prefix}-slider', 'value'),
        Input(f'{prefix}-number', 'value'),
        Input(f'{prefix}-slider', 'value'),
        prevent_initial_call=True
    )
    def sync_inputs(num_val, slider_val, prefix=prefix):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == f'{prefix}-number':
            return dash.no_update, num_val
        else:
            return slider_val, dash.no_update

# Callback: Analyze button - updates all sections
@app.callback(
    Output('decision-banner', 'children'),
    Output('metric-cards', 'children'),
    Output('ai-advice', 'children'),
    Output('monte-carlo-chart', 'children'),
    Output('sensitivity-chart', 'children'),
    Output('error-message', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('income-number', 'value'),
    State('loan-amount-number', 'value'),
    State('interest-number', 'value'),
    State('term-number', 'value'),
    State('credit-number', 'value'),
    State('existing-number', 'value')
)
def analyze_loan(n_clicks, income, loan_amount, interest, term, credit_score, existing_loans):
    if n_clicks == 0:
        # Initial state
        return (
            create_decision_banner("waiting"),
            create_metric_cards(),
            create_ai_advice(),
            create_monte_carlo_chart(50000, 1000000, 8.5, 240),
            html.Div("Click Analyze to see sensitivity analysis"),
            ""
        )

    try:
        # Call analyze API
        response = requests.post(
            f"{API_BASE}/analyze",
            json={
                "monthly_income": float(income),
                "loan_amount": float(loan_amount),
                "interest_rate": float(interest),
                "loan_term": int(term),
                "credit_score": int(credit_score),
                "existing_loans": int(existing_loans)
            },
            timeout=10
        )

        if response.status_code != 200:
            return (
                create_decision_banner("waiting"),
                create_metric_cards(),
                create_ai_advice(),
                create_monte_carlo_chart(income, loan_amount, interest, term),
                html.Div(f"API Error: {response.status_code}", style={'color': COLORS['danger'], 'textAlign': 'center'}),
                f"Server returned error {response.status_code}. Please try again."
            )

        data = response.json()

        decision = data.get('decision', 'CONDITIONAL')
        emi = data.get('emi', 0)
        risk = data.get('risk_level', 'MEDIUM')
        default_prob = data.get('default_probability', 0)
        approval_prob = data.get('approval_probability', 0)
        advice = data.get('advice', 'No advice available')

        # Update all sections
        banner = create_decision_banner(decision)
        metrics = create_metric_cards(emi, risk, default_prob, approval_prob)
        ai_section = create_ai_advice(advice)
        monte = create_monte_carlo_chart(income, loan_amount, interest, term)
        sensitivity = create_sensitivity_chart(income, loan_amount, interest, term, credit_score, existing_loans)

        return banner, metrics, ai_section, monte, sensitivity, ""

    except requests.exceptions.ConnectionError:
        return (
            create_decision_banner("waiting"),
            create_metric_cards(),
            create_ai_advice(),
            create_monte_carlo_chart(income, loan_amount, interest, term),
            html.Div("API connection failed", style={'color': COLORS['danger'], 'textAlign': 'center'}),
            "Cannot connect to API server. Make sure the backend is running on localhost:8000"
        )
    except Exception as e:
        return (
            create_decision_banner("waiting"),
            create_metric_cards(),
            create_ai_advice(),
            create_monte_carlo_chart(income, loan_amount, interest, term),
            html.Div("Analysis failed", style={'color': COLORS['danger'], 'textAlign': 'center'}),
            f"An error occurred: {str(e)[:50]}..."
        )

# Callback: Load history and comparison on page load
@app.callback(
    Output('history-table', 'children'),
    Output('comparison-chart', 'children'),
    Input('analyze-btn', 'n_clicks')
)
def load_static_data(n_clicks):
    # Load on any trigger (including initial load)
    history = create_history_table()
    comparison = create_comparison_chart()
    return history, comparison

# Add custom CSS for slider styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .cyan-slider .rc-slider-track {
                background-color: #00D4FF !important;
            }
            .cyan-slider .rc-slider-handle {
                border-color: #00D4FF !important;
                background-color: #00D4FF !important;
            }
            .cyan-slider .rc-slider-rail {
                background-color: rgba(0, 212, 255, 0.2) !important;
            }
            .cyan-slider .rc-slider-dot {
                border-color: rgba(0, 212, 255, 0.3) !important;
            }
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #0A0E1A;
            }
            ::-webkit-scrollbar-thumb {
                background: #00D4FF;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #00B8E6;
            }
            body {
                margin: 0;
                font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("=" * 60)
    print("MORTGAGE AI DASHBOARD")
    print("=" * 60)
    print(f"API Endpoint: {API_BASE}")
    print(f"Dashboard URL: http://localhost:8050")
    print("=" * 60)
    app.run(debug=False, port=8050, host='0.0.0.0')
