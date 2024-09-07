import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os
from PIL import Image
from scipy import stats
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Advanced Trading Dashboard", layout="wide")
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    """, unsafe_allow_html=True)

if 'seen_welcome' not in st.session_state:
    st.session_state.seen_welcome = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Overview"
if 'df' not in st.session_state:
    st.session_state.df = None

# Load logo
logo_path = os.path.join('assets', 'LOGO.png')
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
else:
    logo = None
    
def get_streamlit_theme():
    try:
        theme = st.get_option("theme.primaryColor")
        if theme == None:
            return "light"
        r, g, b = int(theme[1:3], 16), int(theme[3:5], 16), int(theme[5:7], 16)
        return "light" if (r * 0.299 + g * 0.587 + b * 0.114) > 186 else "dark"
    except:
        return "light"

def set_chart_style(fig, ax):
    theme = get_streamlit_theme()
    bg_color = '#0E1117' if theme == "dark" else '#1E1E1E'
    text_color = '#FAFAFA'

    plt.style.use("dark_background")
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    ax.tick_params(colors=text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    
    ax.grid(True, linestyle='--', alpha=0.2, color=text_color)
    fig.tight_layout()

    return bg_color, text_color

def extract_and_clean_table(df, start_row, end_row):
    table_df = df.iloc[start_row:end_row].reset_index(drop=True)
    table_df.columns = range(len(table_df.columns))
    table_df = table_df.drop(0).reset_index(drop=True)
    
    for col in table_df.columns:
        try:
            table_df[col] = pd.to_datetime(table_df[col], format='%Y.%m.%d %H:%M:%S', errors='raise')
        except ValueError:
            try:
                table_df[col] = pd.to_numeric(table_df[col], errors='raise')
            except ValueError:
                if table_df[col].dtype == 'object':
                    try:
                        table_df[col] = table_df[col].str.split(' / ').str[0].astype(float)
                    except ValueError:
                        pass
    
    return table_df

def extract_tables(excel_file_path):
    df = pd.read_excel(excel_file_path, header=None)
    positions_start = df[df.iloc[:, 0] == 'Positions'].index[0] + 1
    orders_start = df[df.iloc[:, 0] == 'Orders'].index[0] + 1
    positions_table = extract_and_clean_table(df, positions_start, orders_start - 1)
    return positions_table

def finalize_positions_table(positions_table):
    expected_columns = [
        'Time', 'Position', 'Symbol', 'Type', 'Volume', 'Price', 'S/L', 'T/P',
        'Close_Time', 'Close_Price', 'Commission', 'Swap', 'Profit', 'Unused'
    ]
    positions_table.columns = expected_columns
    if 'Unused' in positions_table.columns:
        positions_table = positions_table.drop('Unused', axis=1)
    
    type_dict = {
        'Time': 'datetime64[ns]', 'Position': 'int64', 'Symbol': 'str', 'Type': 'str',
        'Volume': 'float64', 'Price': 'float64', 'S/L': 'float64', 'T/P': 'float64',
        'Close_Time': 'datetime64[ns]', 'Close_Price': 'float64', 'Commission': 'float64',
        'Swap': 'float64', 'Profit': 'float64'
    }
    
    for col, dtype in type_dict.items():
        if col in positions_table.columns:
            if dtype == 'datetime64[ns]':
                positions_table[col] = pd.to_datetime(positions_table[col], format='%Y.%m.%d %H:%M:%S', errors='coerce')
            elif dtype == 'float64':
                positions_table[col] = pd.to_numeric(positions_table[col], errors='coerce')
            elif dtype == 'int64':
                positions_table[col] = pd.to_numeric(positions_table[col], errors='coerce').astype('Int64')
            else:
                positions_table[col] = positions_table[col].astype(dtype)
    
    positions_table['S/L'].fillna(0, inplace=True)
    positions_table['T/P'].fillna(0, inplace=True)
    
    return positions_table

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            positions_table = extract_tables(uploaded_file)
            df = finalize_positions_table(positions_table)
            if 'Profit' not in df.columns:
                st.error("'Profit' column not found in the processed data. Please check your file.")
                return None
            return df
        except Exception as e:
            st.error(f"An error occurred while loading the data: {str(e)}")
            return None
    return None

def clear_history():
    st.session_state.df = None
    st.sidebar.success("All history has been cleared.")

def calculate_metrics(df, starting_balance):
    if 'Profit' not in df.columns or df.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0
    
    total_commission = df['Commission'].sum() if 'Commission' in df.columns else 0
    total_profit = df['Profit'].sum() + total_commission
    win_rate = (df['Profit'] > 0).mean()
    
    cumulative_returns = (df['Profit'].cumsum() + starting_balance) / starting_balance
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    daily_returns = df.groupby(df['Time'].dt.date)['Profit'].sum() / starting_balance
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    
    profit_factor = df[df['Profit'] > 0]['Profit'].sum() / abs(df[df['Profit'] <= 0]['Profit'].sum())
    
    avg_win = df[df['Profit'] > 0]['Profit'].mean()
    avg_loss = abs(df[df['Profit'] <= 0]['Profit'].mean())
    
    return total_profit, win_rate, max_drawdown, sharpe_ratio, profit_factor, avg_win, avg_loss, total_commission

def create_equity_curve(df, starting_balance):
    if 'Profit' not in df.columns or df.empty:
        return None

    df['Account_Balance'] = df['Profit'].cumsum() + starting_balance

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Account_Balance'], mode='lines', line=dict(color='#00C853', width=2), name='Account Balance'))
    fig.add_trace(go.Scatter(x=df['Time'], y=[starting_balance] * len(df['Time']), mode='lines', line=dict(color='#FFA000', width=2, dash='dash'), name='Starting Balance'))

    fig.update_layout(
        title='Account Balance Over Time',
        xaxis_title='Date',
        yaxis_title='Account Balance ($)',
        xaxis_tickformat='%Y-%m-%d',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA',
        legend=dict(
            font_color='#FAFAFA',
            bgcolor='#1E1E1E'
        )
    )

    return fig

def create_commission_impact_chart(df, starting_balance):
    if 'Profit' not in df.columns or 'Commission' not in df.columns or df.empty:
        return None

    df['Net_Balance'] = df['Profit'].cumsum() + starting_balance
    df['Gross_Balance'] = df['Profit'].cumsum() - df['Commission'].cumsum() + starting_balance

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Gross_Balance'], mode='lines', line=dict(color='#4CAF50', width=4), name='Gross Balance'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Net_Balance'], mode='lines', line=dict(color='#2196F3', width=4), name='Net Balance'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Gross_Balance'], fill='tonexty', mode='none', fillcolor='#634545', opacity=0.05, name='Commission Impact'))

    fig.add_hline(y=starting_balance, line_color='#FFA000', line_width=2, line_dash='dash', annotation_text='Starting Balance', annotation_position='top left')

    fig.update_layout(
        title='Impact of Commissions on Account Balance',
        xaxis_title='Date',
        yaxis_title='Account Balance ($)',
        xaxis_tickformat='%Y-%m-%d',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA',
        legend=dict(
            font_color='#FAFAFA',
            bgcolor='#1E1E1E'
        )
    )

    return fig

def create_daily_pnl(df):
    if 'Profit' not in df.columns or df.empty:
        return None

    daily_pnl = df.groupby(df['Time'].dt.date)['Profit'].sum()

    fig = go.Figure()
    colors = ['#00C853' if x >= 0 else '#FF5252' for x in daily_pnl]
    fig.add_trace(go.Bar(x=daily_pnl.index, y=daily_pnl.values, marker_color=colors))

    fig.update_layout(
        title='Daily GROSS Profit/Loss',
        xaxis_title='Date',
        yaxis_title='Profit/Loss',
        xaxis_tickformat='%Y-%m-%d',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA'
    )

    return fig

def create_hourly_heatmap(df):
    if 'Profit' not in df.columns or df.empty:
        return None

    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day_name()
    hourly_returns = df.pivot_table(values='Profit', index='Day', columns='Hour', aggfunc='sum')

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hourly_returns = hourly_returns.reindex(days_order)

    fig = go.Figure(data=go.Heatmap(
        x=list(range(24)),
        y=days_order,
        z=hourly_returns.values,
        colorscale='RdYlGn',
        zmin=-hourly_returns.abs().max().max(),
        zmax=hourly_returns.abs().max().max()
    ))

    fig.update_layout(
        title='Hourly Returns Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        xaxis_tickangle=0,
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA',
        coloraxis_colorbar=dict(
            title='Profit',
            titleside='right',
            ticksuffix='$'
        )
    )

    return fig

def create_symbol_distribution(df):
    if 'Symbol' not in df.columns or df.empty:
        return None

    symbol_counts = df['Symbol'].value_counts()

    fig = go.Figure(data=[go.Pie(labels=symbol_counts.index, values=symbol_counts.values)])

    fig.update_layout(
        title='Symbol Distribution',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA',
        legend=dict(
            font_color='#FAFAFA',
            bgcolor='#1E1E1E'
        )
    )

    return fig

def calculate_trade_returns(df, starting_balance):
    return df['Profit'] / starting_balance

def create_trade_duration_analysis(df):
    if 'Time' not in df.columns or 'Time.1' not in df.columns or 'Profit' not in df.columns or df.empty:
        return None

    df['Duration'] = (df['Time.1'] - df['Time']).dt.total_seconds() / 60

    duration_bins = [0, 15, 30, 45, 60, 90, 120, 180, 240, 360, float('inf')]
    duration_labels = ['0-15m', '15-30m', '30-45m', '45-60m', '60-90m', '90-120m', '2-3h', '3-4h', '4-6h', '>6h']
    df['Duration_Category'] = pd.cut(df['Duration'], bins=duration_bins, labels=duration_labels, include_lowest=True)

    profits_by_duration = df[df['Profit'] > 0].groupby('Duration_Category')['Profit'].sum()
    losses_by_duration = df[df['Profit'] < 0].groupby('Duration_Category')['Profit'].sum()
    trades_by_duration = df['Duration_Category'].value_counts().sort_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=profits_by_duration.index, y=profits_by_duration.values, name='Profits', marker_color='#00C853'))
    fig.add_trace(go.Bar(x=losses_by_duration.index, y=losses_by_duration.values, name='Losses', marker_color='#FF5252'))

    fig.add_trace(go.Scatter(x=trades_by_duration.index, y=trades_by_duration.values, mode='lines+markers', name='Number of Trades', line=dict(color='#2196F3'), marker=dict(symbol='square')))

    fig.update_layout(
        title='Profit/Loss and Trade Count by Duration',
        xaxis_title='Trade Duration',
        yaxis_title='Profit/Loss ($)',
        yaxis_tickformat='${:.0f}',
        xaxis_tickangle=45,
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA',
        legend=dict(
            font_color='#FAFAFA',
            bgcolor='#1E1E1E'
        )
    )

    return fig



def monte_carlo_simulation(df, starting_balance, num_simulations, num_trades_to_simulate):
    trade_returns = calculate_trade_returns(df, starting_balance)
    
    if len(trade_returns) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no returns data
    
    simulation_df = pd.DataFrame()
    
    for x in range(num_simulations):
        trade_returns_sample = np.random.choice(trade_returns, size=num_trades_to_simulate)
        account_values = [starting_balance]
        
        for trade_return in trade_returns_sample:
            account_values.append(account_values[-1] * (1 + trade_return))
        
        simulation_df[x] = account_values
    
    return simulation_df




def plot_advanced_monte_carlo(simulation_df, starting_balance):
    # Monte Carlo Simulation Plot
    fig_simulation = go.Figure()

    # Plot all simulation paths with low alpha using RGBA color for transparency
    for i in range(simulation_df.shape[1]):
        fig_simulation.add_trace(go.Scatter(
            x=simulation_df.index, 
            y=simulation_df.iloc[:, i], 
            mode='lines', 
            line=dict(color='rgba(76, 175, 80, 0.1)', width=0.5),
            showlegend=False
        ))

    # Calculate and plot statistics
    median_line = simulation_df.median(axis=1)
    mean_line = simulation_df.mean(axis=1)
    percentile_5 = simulation_df.quantile(0.05, axis=1)
    percentile_95 = simulation_df.quantile(0.95, axis=1)

    fig_simulation.add_trace(go.Scatter(
        x=simulation_df.index, 
        y=median_line, 
        mode='lines', 
        line=dict(color='#FFA000', width=2), 
        name='Median'
    ))

    fig_simulation.add_trace(go.Scatter(
        x=simulation_df.index, 
        y=mean_line, 
        mode='lines', 
        line=dict(color='#2196F3', width=2), 
        name='Mean'
    ))

    fig_simulation.add_trace(go.Scatter(
        x=simulation_df.index, 
        y=percentile_5, 
        mode='lines', 
        line=dict(color='#3074c2', width=0.5, dash='dot'), 
        showlegend=False
    ))

    fig_simulation.add_trace(go.Scatter(
        x=simulation_df.index, 
        y=percentile_95, 
        mode='lines', 
        line=dict(color='#3074c2', width=0.5, dash='dot'), 
        fillcolor='rgba(76, 175, 80, 0.3)', 
        fill='tonexty', 
        name='90% Confidence Interval'
    ))

    fig_simulation.add_hline(
        y=starting_balance, 
        line_color='#FF5252', 
        line_width=2, 
        line_dash='dash', 
        annotation_text='Starting Balance', 
        annotation_position='top left'
    )

    fig_simulation.update_layout(
        title='Monte Carlo Simulation',
        xaxis_title='Number of Trades',
        yaxis_title='Account Value ($)',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA',
        legend=dict(
            font_color='#FAFAFA',
            bgcolor='#1E1E1E'
        )
    )

    # Histogram of Final Values
    fig_histogram = go.Figure()

    fig_histogram.add_trace(go.Histogram(
        x=simulation_df.iloc[-1], 
        nbinsx=30, 
        marker_color='#4CAF50'
    ))

    fig_histogram.update_layout(
        title='Distribution of Final Values',
        xaxis_title='Account Value ($)',
        yaxis_title='Frequency',
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font_color='#FAFAFA',
        bargap=0.2  # Gap between bars
    )

    return fig_simulation, fig_histogram


def local_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def inject_custom_css():
    st.markdown("""
    <style>
        .sidebar-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            text-align: left;
            background-color: #1E1E1E;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .sidebar-button:hover {
            background-color: #2E2E2E;
        }
        .sidebar-button i {
            margin-right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)


def centered_button(label, key):
    return f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        ">
            <button kind="primary" style="
                background-color: #1E1E1E;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            " onclick="document.getElementById('{key}').click()">
                {label}
            </button>
        </div>
        <div style="display:none;">
            <button id="{key}"></button>
        </div>
    """
    
def main():
    inject_custom_css()

    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
        """, unsafe_allow_html=True)

    with st.sidebar:
        if logo is not None:
            st.image(logo, use_column_width=True, output_format="PNG")
        else:
            st.warning("Logo not found. Please check the assets folder.")
        
        st.sidebar.markdown('<br><br><br><br>', unsafe_allow_html=True)

        if not st.session_state.seen_welcome:
            st.markdown("""
                <style>
                    div.stButton > button:first-child {
                        display: block;
                        margin: 0 auto;
                        background-color: #4CAF50;
                        color: white;
                        padding: 15px 32px;
                        text-align: center;
                        text-decoration: none;
                        font-size: 18px;
                        border-radius: 8px;
                        border: none;
                        cursor: pointer;
                        transition: background-color 0.3s;
                    }
                    div.stButton > button:first-child:hover {
                        background-color: #45a049;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='padding: 15vh'></div>", unsafe_allow_html=True)
            
            if st.button("Enter Dashboard", key="enter_dashboard"):
                st.session_state.seen_welcome = True
                st.rerun()
            
        else:
            starting_balance = st.number_input("Enter starting balance ($)", min_value=1.0, value=10000.0, step=1000.0)
            
            uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
            if uploaded_file is not None:
                st.session_state.df = load_data(uploaded_file)
                if st.session_state.df is not None:
                    st.success("Data loaded and processed successfully!")

            if st.button("Clear All History"):
                clear_history()

            st.markdown("---")

            if st.button("📊 Overview", key="btn_overview", use_container_width=True):
                st.session_state.current_page = "Overview"
            
            if st.button("🔍 Analysis", key="btn_analysis", use_container_width=True):
                st.session_state.current_page = "Analysis"
            
            if st.button("📜 Trade History", key="btn_trade_history", use_container_width=True):
                st.session_state.current_page = "Trade History"
            
            if st.button("🎲 Monte Carlo Simulation", key="btn_monte_carlo", use_container_width=True):
                st.session_state.current_page = "Monte Carlo"

    if not st.session_state.seen_welcome:
        st.title("Welcome Associate!")
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.markdown("""
    <h1 style='color: #ADD8E6;'>
    To access the Advanced Trading Dashboard, Click 'Enter Dashboard' in the sidebar to begin.
    </h1>
    """, unsafe_allow_html=True)
    elif st.session_state.df is None or 'Profit' not in st.session_state.df.columns:
        st.info("No valid data available. Please upload a file with the required columns to get started.")
    else:
        df = st.session_state.df
        if st.session_state.current_page == "Overview":
            total_profit, win_rate, max_drawdown, sharpe_ratio, profit_factor, avg_win, avg_loss, total_commission = calculate_metrics(df, starting_balance)
            
            st.header("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Net Profit", f"${total_profit:.2f}")
            col2.metric("Win Rate", f"{win_rate:.2%}")
            col3.metric("Max Drawdown", f"{max_drawdown:.2%}")
            col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Profit Factor", f"{profit_factor:.2f}")
            col6.metric("Avg Win", f"${avg_win:.2f}")
            col7.metric("Avg Loss", f"${avg_loss:.2f}")
            col8.metric("Total Commissions", f"${total_commission:.2f}")

            st.header("Performance Overview")
            with st.expander("Equity Curve (After Commissions)", expanded=True):
                equity_curve = create_equity_curve(df, starting_balance)
                if equity_curve:
                    st.plotly_chart(equity_curve, use_container_width=True)

            with st.expander("Commission Impact", expanded=True):
                commission_impact = create_commission_impact_chart(df, starting_balance)
                if commission_impact:
                    st.plotly_chart(commission_impact, use_container_width=True)

        elif st.session_state.current_page == "Analysis":
            st.header("Profit Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("Daily Profit/Loss", expanded=True):
                    daily_pnl = create_daily_pnl(df)
                    if daily_pnl:
                        st.plotly_chart(daily_pnl, use_container_width=True)

            with col2:
                if 'Symbol' in df.columns:
                    with st.expander("Symbol Distribution", expanded=True):
                        symbol_chart = create_symbol_distribution(df)
                        if symbol_chart:
                            st.plotly_chart(symbol_chart, use_container_width=True)

            st.header("Time-based Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                with st.expander("Hourly Returns Heatmap", expanded=True):
                    hourly_heatmap = create_hourly_heatmap(df)
                    if hourly_heatmap:
                        st.plotly_chart(hourly_heatmap, use_container_width=True)

            with col4:
                with st.expander("Trade Duration Analysis", expanded=True):
                    trade_duration_chart = create_trade_duration_analysis(df)
                    if trade_duration_chart:
                        st.plotly_chart(trade_duration_chart, use_container_width=True)

        elif st.session_state.current_page == "Trade History":
            st.header("Recent Trades")
            st.dataframe(df.sort_values('Time', ascending=False).head(100))
            
        elif st.session_state.current_page == "Monte Carlo":
            st.header("Monte Carlo Simulation")
        
            num_trades = len(df)
        
            st.write(f"Number of trades: {num_trades}")
        
            if num_trades < 10:
                st.error(f"You have only {num_trades} trades. For a Monte Carlo simulation, it's recommended to have at least 10 trades.")
            elif num_trades < 100:
                st.warning(f"You have {num_trades} trades. While this is sufficient for a basic simulation, having more trades would provide more robust results.")
            else:
                st.success(f"You have {num_trades} trades, which is good for a comprehensive Monte Carlo simulation.")

            if num_trades >= 10:
                num_simulations = st.slider("Number of Simulations", min_value=100, max_value=1000, value=500, step=100)
                num_trades_to_simulate = st.slider("Number of Trades to Simulate", min_value=10, max_value=1000, value=min(252, num_trades), step=10)
        
                if st.button("Run Monte Carlo Simulation"):
                    with st.spinner("Running simulation..."):
                        simulation_df = monte_carlo_simulation(df, starting_balance, num_simulations, num_trades_to_simulate)
                
                    if not simulation_df.empty:
                        fig_simulation, fig_histogram = plot_advanced_monte_carlo(simulation_df, starting_balance)
                        st.plotly_chart(fig_simulation, use_container_width=True)
                        st.plotly_chart(fig_histogram, use_container_width=True)
            else:
                st.error("Unable to run Monte Carlo simulation. Please check your data.")

if __name__ == "__main__":
    main()
