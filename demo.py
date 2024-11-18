import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# Step 1: Load Real-World Dataset
file_path = "world_bank_gdp_inflation.csv"
country_data = pd.read_csv(file_path)

# Ensure the dataset contains required columns
indicators = ['Year', 'GDP', 'GDP_per_capita', 'Inflation', 'Adjusted_Net_National_Income']
country_data = country_data[indicators]
country_data.set_index("Year", inplace=True)

# Handle missing data
original_data = country_data.copy()
country_data = country_data.interpolate(method='linear', limit_direction='forward', axis=0)

# Step 2: Identify Optimal ARIMA Orders
def find_best_arima(series):
    """Automatically find the best ARIMA order using auto_arima."""
    model = auto_arima(series, seasonal=False, trace=True, suppress_warnings=True, stepwise=True)
    return model.order

# Find optimal ARIMA orders
gdp_order = find_best_arima(country_data['GDP'])
gdp_pc_order = find_best_arima(country_data['GDP_per_capita'])
inflation_order = find_best_arima(country_data['Inflation'])
anni_order = find_best_arima(country_data['Adjusted_Net_National_Income'])

# Train ARIMA Models
gdp_model_fit = ARIMA(country_data['GDP'], order=gdp_order).fit()
gdp_pc_model_fit = ARIMA(country_data['GDP_per_capita'], order=gdp_pc_order).fit()
inflation_model_fit = ARIMA(country_data['Inflation'], order=inflation_order).fit()
anni_model_fit = ARIMA(country_data['Adjusted_Net_National_Income'], order=anni_order).fit()

# Function to format years in Arabic
def format_years_arabic(years):
    arabic_numerals = {0: "٠", 1: "١", 2: "٢", 3: "٣", 4: "٤", 5: "٥", 6: "٦", 7: "٧", 8: "٨", 9: "٩"}
    return ["".join(arabic_numerals[int(digit)] for digit in str(year)) for year in years]

# Step 3: Create Dash App
app = Dash(__name__)

# App Layout
app.layout = html.Div([
    html.Div([
        dcc.RadioItems(
            id='language-switch',
            options=[
                {'label': 'English', 'value': 'EN'},
                {'label': 'العربية', 'value': 'AR'}
            ],
            value='AR',
            inline=True,
            style={"textAlign": "center", "marginBottom": "20px"}
        ),
        dcc.Dropdown(
            id='chart-type',
            style={"width": "50%", "margin": "auto"}
        )
    ], style={"textAlign": "center"}),

    html.Div(id='main-title', style={"textAlign": "center", "marginBottom": "20px"}),

    dcc.Graph(id="gdp-graph"),
    dcc.Graph(id="gdp-per-capita-graph"),
    dcc.Graph(id="inflation-graph"),
    dcc.Graph(id="anni-graph"),

    html.Div([
        html.Label(id="slider-label", style={"fontSize": "18px", "marginBottom": "10px", "color": "#FF5722"}),
        dcc.Slider(
            id='forecast-slider',
            min=1,
            max=10,
            step=1,
            value=5,
            marks={i: str(i) for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True},
        )
    ], style={"width": "80%", "margin": "auto", "textAlign": "center"})
], id='layout-container')

@app.callback(
    [Output('main-title', 'children'),
     Output('layout-container', 'style'),
     Output('slider-label', 'children'),
     Output('chart-type', 'options')],
    Input('language-switch', 'value')
)
def update_language(language):
    if language == 'EN':
        return (
            "Economic Indicators Forecast for Oman",
            {"direction": "ltr", "backgroundColor": "#f9f9f9"},
            "Adjust the Forecast Period:",
            [
                {'label': 'Line', 'value': 'lines+markers'},
                {'label': 'Bar', 'value': 'bar'},
                {'label': 'Scatter', 'value': 'markers'}
            ]
        )
    else:
        return (
            "لوحة توقعات المؤشرات الاقتصادية في عمان",
            {"direction": "rtl", "backgroundColor": "#f9f9f9"},
            "تغيير عدد سنوات التوقع:",
            [
                {'label': 'خط', 'value': 'lines+markers'},
                {'label': 'شريط', 'value': 'bar'},
                {'label': 'منتشر', 'value': 'markers'}
            ]
        )

@app.callback(
    [Output("gdp-graph", "figure"), 
     Output("gdp-per-capita-graph", "figure"), 
     Output("inflation-graph", "figure"), 
     Output("anni-graph", "figure")],
    [Input("forecast-slider", "value"), Input('language-switch', 'value'), Input('chart-type', 'value')]
)
def update_graphs(forecast_years, language, chart_type):
    # Forecasts
    gdp_forecast = gdp_model_fit.forecast(steps=forecast_years)
    gdp_pc_forecast = gdp_pc_model_fit.forecast(steps=forecast_years)
    inflation_forecast = inflation_model_fit.forecast(steps=forecast_years)
    anni_forecast = anni_model_fit.forecast(steps=forecast_years)
    
    # Ensure forecasts start after the last year in the dataset
    last_year = int(country_data.index.max())
    new_forecast_years = [last_year + i for i in range(1, forecast_years + 1)]

    # Format years and labels based on language
    if language == 'AR':
        historical_years = format_years_arabic(country_data.index)
        forecast_years_formatted = format_years_arabic(new_forecast_years)
        gdp_title, gdp_pc_title = "الناتج المحلي الإجمالي", "نصيب الفرد من الناتج المحلي الإجمالي"
        inflation_title, anni_title = "التضخم", "الدخل القومي الصافي المعدل"
        xaxis_title, yaxis_title = "السنة", "القيمة"
        legend_historical, legend_forecast = "تاريخي", "توقع"
    else:
        historical_years = country_data.index.tolist()
        forecast_years_formatted = new_forecast_years
        gdp_title, gdp_pc_title = "GDP", "GDP per Capita"
        inflation_title, anni_title = "Inflation", "Adjusted Net National Income"
        xaxis_title, yaxis_title = "Year", "Value"
        legend_historical, legend_forecast = "Historical", "Forecast"

    def create_figure(title, historical_data, forecast_data, y_title):
        fig = go.Figure()
        if chart_type == 'bar':
            fig.add_trace(go.Bar(x=historical_years, y=historical_data, name=legend_historical))
            fig.add_trace(go.Bar(x=forecast_years_formatted, y=forecast_data, name=legend_forecast))
        else:
            fig.add_trace(go.Scatter(x=historical_years, y=historical_data, mode=chart_type, name=legend_historical))
            fig.add_trace(go.Scatter(x=forecast_years_formatted, y=forecast_data, mode=chart_type, name=legend_forecast))
        fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=y_title)
        return fig

    return (
        create_figure(gdp_title, country_data['GDP'], gdp_forecast, gdp_title),
        create_figure(gdp_pc_title, country_data['GDP_per_capita'], gdp_pc_forecast, gdp_pc_title),
        create_figure(inflation_title, country_data['Inflation'], inflation_forecast, inflation_title),
        create_figure(anni_title, country_data['Adjusted_Net_National_Income'], anni_forecast, anni_title)
    )

# Step 4: Run the Dash App
if __name__ == "__main__":
    app.run_server(debug=True)
