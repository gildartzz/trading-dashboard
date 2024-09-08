# Trading-dashboard


# Advanced Trading Dashboard

## Overview

This Advanced Trading Dashboard is a Streamlit-based web application designed to provide comprehensive analysis and visualization of trading data. It offers various features including performance metrics, equity curves, profit analysis, and Monte Carlo simulations.

## Features

- **Key Metrics**: Total Net Profit, Win Rate, Max Drawdown, Sharpe Ratio, and more.
- **Performance Overview**: Equity curve and commission impact analysis.
- **Profit Analysis**: Daily profit/loss charts and symbol distribution.
- **Time-based Analysis**: Hourly returns heatmap and trade duration analysis.
- **Monte Carlo Simulation**: Forecast potential future performance based on historical data.
- **PDF Report Generation**: Download key metrics and charts in PDF format.

## Installation

To run this dashboard locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/trading-dashboard.git
   cd trading-dashboard
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Launch the application using the installation steps above.
2. Upload your trading data in Excel (.xlsx) format.
3. Navigate through different sections using the sidebar:
   - Overview
   - Analysis
   - Trade History
   - Monte Carlo Simulation
4. Use the "Generate PDF Report" button to download a summary of key metrics and charts.

## Data Format

The dashboard expects an Excel file with the following columns:
- Time
- Position
- Symbol
- Type
- Volume
- Price
- S/L (Stop Loss)
- T/P (Take Profit)
- Close_Time
- Close_Price
- Commission
- Swap
- Profit

## Dependencies

- streamlit
- pandas
- matplotlib
- seaborn
- numpy
- Pillow
- plotly
- reportlab

## Contributing

Contributions to improve the dashboard are welcome. Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Disclaimer

This tool is for informational purposes only. It is not intended to be investment advice. Please use it at your own risk.
