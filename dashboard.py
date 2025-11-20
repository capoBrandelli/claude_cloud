"""
Interactive Dashboard for Financial Sentiment Analysis
Uses Plotly Dash for visualization
"""
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from database import Database
from typing import Dict, List
import json


class SentimentDashboard:
    """Interactive dashboard for sentiment analysis visualization"""

    def __init__(self, db: Database):
        self.db = db
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.Div([
                html.H1('Financial Market Sentiment Analyzer',
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
                html.P('Real-time sentiment analysis across Forex, Commodities, Stocks, and ETFs',
                      style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),

            # Filters
            html.Div([
                html.Div([
                    html.Label('Market Type:'),
                    dcc.Dropdown(
                        id='market-type-filter',
                        options=[
                            {'label': 'All Markets', 'value': 'all'},
                            {'label': 'Forex', 'value': 'forex'},
                            {'label': 'Commodities', 'value': 'commodity'},
                            {'label': 'Stocks', 'value': 'stocks'},
                            {'label': 'ETFs', 'value': 'etf'}
                        ],
                        value='all',
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '20px'}),

                html.Div([
                    html.Label('Time Period (days):'),
                    dcc.Dropdown(
                        id='time-period-filter',
                        options=[
                            {'label': '1 Day', 'value': 1},
                            {'label': '3 Days', 'value': 3},
                            {'label': '7 Days', 'value': 7},
                            {'label': '14 Days', 'value': 14},
                            {'label': '30 Days', 'value': 30}
                        ],
                        value=7,
                        style={'width': '150px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '20px'}),

                html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                           style={'padding': '10px 20px', 'marginTop': '22px'})
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),

            # Summary Cards
            html.Div(id='summary-cards', style={'marginBottom': '20px'}),

            # Main Visualizations
            html.Div([
                # Sentiment Comparison Chart
                html.Div([
                    html.H3('News vs Futures Sentiment Comparison'),
                    dcc.Graph(id='sentiment-comparison-chart')
                ], style={'width': '100%', 'marginBottom': '20px'}),

                # Alignment Status
                html.Div([
                    html.H3('Sentiment Alignment Heatmap'),
                    dcc.Graph(id='alignment-heatmap')
                ], style={'width': '100%', 'marginBottom': '20px'}),

                # Time Series
                html.Div([
                    html.H3('Sentiment Trends Over Time'),
                    dcc.Graph(id='sentiment-timeseries')
                ], style={'width': '100%', 'marginBottom': '20px'}),

                # Divergence Opportunities
                html.Div([
                    html.H3('Top Divergence Opportunities'),
                    html.Div(id='divergence-table')
                ], style={'width': '100%', 'marginBottom': '20px'}),

                # Detailed Comparison Table
                html.Div([
                    html.H3('Detailed Sentiment Analysis'),
                    html.Div(id='detailed-table')
                ], style={'width': '100%'})
            ])
        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'})

    def setup_callbacks(self):
        """Setup interactive callbacks"""

        @self.app.callback(
            [Output('summary-cards', 'children'),
             Output('sentiment-comparison-chart', 'figure'),
             Output('alignment-heatmap', 'figure'),
             Output('sentiment-timeseries', 'figure'),
             Output('divergence-table', 'children'),
             Output('detailed-table', 'children')],
            [Input('market-type-filter', 'value'),
             Input('time-period-filter', 'value'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_dashboard(market_type, days, n_clicks):
            # Fetch data from database
            market_filter = None if market_type == 'all' else market_type

            news_data = self.db.get_news_sentiment(market_type=market_filter, days=days)
            futures_data = self.db.get_futures_sentiment(market_type=market_filter, days=days)
            comparisons = self.db.get_sentiment_comparisons(market_type=market_filter, days=days)

            # Generate visualizations
            summary = self.create_summary_cards(news_data, futures_data, comparisons)
            comparison_chart = self.create_comparison_chart(comparisons)
            heatmap = self.create_alignment_heatmap(comparisons)
            timeseries = self.create_timeseries(news_data, futures_data)
            divergence_table = self.create_divergence_table(comparisons)
            detailed_table = self.create_detailed_table(comparisons)

            return summary, comparison_chart, heatmap, timeseries, divergence_table, detailed_table

    def create_summary_cards(self, news_data: List[Dict],
                            futures_data: List[Dict],
                            comparisons: List[Dict]) -> html.Div:
        """Create summary statistics cards"""
        total_news = len(news_data)
        total_futures = len(futures_data)
        total_comparisons = len(comparisons)

        # Calculate average alignment
        if comparisons:
            avg_alignment = sum(c['alignment_score'] for c in comparisons) / len(comparisons)
            strong_divergence = sum(1 for c in comparisons if c['alignment_status'] == 'Strong Divergence')
        else:
            avg_alignment = 0
            strong_divergence = 0

        cards = html.Div([
            self.create_card('Total News Articles', str(total_news), '#3498db'),
            self.create_card('Futures Data Points', str(total_futures), '#2ecc71'),
            self.create_card('Avg Alignment Score', f'{avg_alignment:.2f}', '#e74c3c'),
            self.create_card('Strong Divergences', str(strong_divergence), '#f39c12')
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})

        return cards

    def create_card(self, title: str, value: str, color: str) -> html.Div:
        """Create a summary card"""
        return html.Div([
            html.H4(title, style={'color': '#2c3e50', 'marginBottom': '10px'}),
            html.H2(value, style={'color': color, 'margin': '0'})
        ], style={
            'backgroundColor': '#fff',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'minWidth': '200px',
            'margin': '10px',
            'textAlign': 'center'
        })

    def create_comparison_chart(self, comparisons: List[Dict]) -> go.Figure:
        """Create sentiment comparison bar chart"""
        if not comparisons:
            return go.Figure().add_annotation(
                text="No data available",
                showarrow=False,
                font=dict(size=20)
            )

        df = pd.DataFrame(comparisons)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='News Sentiment',
            x=df['symbol'],
            y=df['news_sentiment_avg'],
            marker_color='#3498db'
        ))

        fig.add_trace(go.Bar(
            name='Futures Sentiment',
            x=df['symbol'],
            y=df['futures_sentiment_avg'],
            marker_color='#2ecc71'
        ))

        fig.update_layout(
            barmode='group',
            xaxis_title='Symbol',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1, 1]),
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def create_alignment_heatmap(self, comparisons: List[Dict]) -> go.Figure:
        """Create alignment status heatmap"""
        if not comparisons:
            return go.Figure().add_annotation(
                text="No data available",
                showarrow=False,
                font=dict(size=20)
            )

        df = pd.DataFrame(comparisons)

        # Create matrix data
        symbols = df['symbol'].unique()
        z_data = []
        hover_text = []

        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) > 0:
                z_data.append([symbol_data['alignment_score'].values[0]])
                hover_text.append([
                    f"{symbol}<br>"
                    f"Alignment: {symbol_data['alignment_score'].values[0]:.2f}<br>"
                    f"Status: {symbol_data['alignment_status'].values[0]}"
                ])

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=['Alignment Score'],
            y=symbols,
            hovertext=hover_text,
            hoverinfo='text',
            colorscale='RdYlGn',
            zmid=0.5
        ))

        fig.update_layout(
            xaxis_title='',
            yaxis_title='Symbol',
            template='plotly_white'
        )

        return fig

    def create_timeseries(self, news_data: List[Dict],
                         futures_data: List[Dict]) -> go.Figure:
        """Create sentiment time series chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('News Sentiment Over Time', 'Futures Sentiment Over Time'),
            vertical_spacing=0.15
        )

        if news_data:
            news_df = pd.DataFrame(news_data)
            news_df['publication_date'] = pd.to_datetime(news_df['publication_date'])
            news_grouped = news_df.groupby(['publication_date', 'symbol'])['sentiment_score'].mean().reset_index()

            for symbol in news_grouped['symbol'].unique():
                symbol_data = news_grouped[news_grouped['symbol'] == symbol]
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['publication_date'],
                        y=symbol_data['sentiment_score'],
                        mode='lines+markers',
                        name=f'{symbol} (News)',
                        showlegend=True
                    ),
                    row=1, col=1
                )

        if futures_data:
            futures_df = pd.DataFrame(futures_data)
            futures_df['data_date'] = pd.to_datetime(futures_df['data_date'])
            futures_grouped = futures_df.groupby(['data_date', 'symbol'])['sentiment_score'].mean().reset_index()

            for symbol in futures_grouped['symbol'].unique():
                symbol_data = futures_grouped[futures_grouped['symbol'] == symbol]
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['data_date'],
                        y=symbol_data['sentiment_score'],
                        mode='lines+markers',
                        name=f'{symbol} (Futures)',
                        showlegend=True
                    ),
                    row=2, col=1
                )

        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Sentiment Score', range=[-1, 1], row=1, col=1)
        fig.update_yaxes(title_text='Sentiment Score', range=[-1, 1], row=2, col=1)
        fig.update_layout(height=700, template='plotly_white')

        return fig

    def create_divergence_table(self, comparisons: List[Dict]) -> dash_table.DataTable:
        """Create table of divergence opportunities"""
        if not comparisons:
            return html.P('No divergence data available')

        # Filter for divergences
        divergences = [c for c in comparisons if 'Divergence' in c.get('alignment_status', '')]
        divergences.sort(key=lambda x: x.get('divergence_magnitude', 0), reverse=True)

        if not divergences:
            return html.P('No significant divergences detected')

        df = pd.DataFrame(divergences)
        display_columns = ['symbol', 'market_type', 'news_sentiment_avg', 'futures_sentiment_avg',
                          'divergence_magnitude', 'alignment_status', 'confidence_score']

        df = df[display_columns]
        df.columns = ['Symbol', 'Market', 'News Sent.', 'Futures Sent.',
                     'Divergence', 'Status', 'Confidence']

        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} contains "Strong"'},
                    'backgroundColor': '#ffcccc',
                    'color': '#c0392b'
                }
            ]
        )

    def create_detailed_table(self, comparisons: List[Dict]) -> dash_table.DataTable:
        """Create detailed comparison table"""
        if not comparisons:
            return html.P('No comparison data available')

        df = pd.DataFrame(comparisons)
        display_columns = ['symbol', 'market_type', 'news_sentiment_avg', 'futures_sentiment_avg',
                          'alignment_score', 'alignment_status', 'news_count', 'futures_count', 'insights']

        df = df[display_columns]
        df.columns = ['Symbol', 'Market', 'News Sent.', 'Futures Sent.',
                     'Alignment', 'Status', 'News #', 'Futures #', 'Insights']

        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'maxWidth': '300px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            },
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} = "Strong Alignment"'},
                    'backgroundColor': '#d4edda',
                    'color': '#155724'
                },
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} contains "Divergence"'},
                    'backgroundColor': '#f8d7da',
                    'color': '#721c24'
                }
            ],
            tooltip_data=[
                {
                    'Insights': {'value': str(row['Insights']), 'type': 'markdown'}
                } for row in df.to_dict('records')
            ],
            tooltip_duration=None
        )

    def run(self, debug=True, port=8050):
        """Run the dashboard server"""
        print(f"\n{'='*60}")
        print("Financial Sentiment Analyzer Dashboard")
        print(f"{'='*60}")
        print(f"\nStarting dashboard server...")
        print(f"Access the dashboard at: http://localhost:{port}")
        print(f"\nPress Ctrl+C to stop the server\n")
        print(f"{'='*60}\n")

        self.app.run_server(debug=debug, port=port, host='0.0.0.0')
