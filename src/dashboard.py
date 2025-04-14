# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import time

from data.mongodb_connector import EnergyDatabaseConnector

# Setup
st.set_page_config(page_title="Energy Consumption Dashboard", layout="wide")
st.title("Real-Time Energy Consumption Dashboard")

# Initialize database connector


def get_db_connector():
    """Initialize database connector with caching for performance."""
    conn_string = st.secrets.get("mongodb_uri")
    return EnergyDatabaseConnector(connection_string=conn_string)


db_connector = get_db_connector()

# Sidebar controls
st.sidebar.header("Dashboard Controls")
hours_to_display = st.sidebar.slider("Hours to Display", 1, 48, 6)
update_interval = st.sidebar.slider("Update Interval (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto Refresh", True)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Status")
    current_stats = st.empty()  # Placeholder for current stats

with col2:
    st.subheader("Last 10 Data Points")
    recent_data_table = st.empty()  # Placeholder for recent data table

# Charts
demand_chart = st.empty()  # Placeholder for demand chart
temp_chart = st.empty()  # Placeholder for temperature chart
scatter_chart = st.empty()  # Placeholder for scatter plot

# Function to update dashboard


def update_dashboard():
    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours_to_display)

    data = db_connector.fetch_data(start_date, end_date)

    if len(data) == 0 or data.empty:
        st.warning("No data available for the selected time range.")
        return

    # Update current stats
    latest_point = data.iloc[-1]
    with current_stats.container():
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric(
            "Current Demand", f"{latest_point['demand']:.2f} kWh")
        metrics_col2.metric("Current Temperature",
                            f"{latest_point['temperature']:.1f}°C")

        # Calculate trend
        if len(data) > 1:
            prev_point = data.iloc[-2]
            demand_change = latest_point['demand'] - prev_point['demand']
            metrics_col1.metric("Current Demand", f"{latest_point['demand']:.2f} kWh",
                                f"{demand_change:.2f}")

            temp_change = latest_point['temperature'] - \
                prev_point['temperature']
            metrics_col2.metric("Current Temperature", f"{latest_point['temperature']:.1f}°C",
                                f"{temp_change:.1f}°C")

        # Add timestamp
        metrics_col3.metric(
            "Last Updated", latest_point['timestamp'].strftime("%H:%M:%S"))

    # Update recent data table
    recent_data = data.tail(10).sort_values('timestamp', ascending=False)
    recent_data_display = recent_data[[
        'timestamp', 'demand', 'temperature']].copy()
    recent_data_display.columns = ['Timestamp', 'Demand', 'Temperature']
    recent_data_table.dataframe(recent_data_display)

    # Update demand chart
    fig_demand = px.line(data, x='timestamp', y='demand',
                         title=f"Energy Demand - Last {hours_to_display} Hours")
    demand_chart.plotly_chart(
        fig_demand, use_container_width=True, key=f"demand_chart_{hours_to_display}")

    # Update temperature chart
    fig_temp = px.line(data, x='timestamp', y='temperature',
                       title=f"Temperature - Last {hours_to_display} Hours")
    fig_temp.update_traces(line_color='red')
    temp_chart.plotly_chart(
        fig_temp, use_container_width=True, key=f"temp_chart_{hours_to_display}")

    # Update scatter plot
    fig_scatter = px.scatter(data, x='temperature', y='demand',
                             title="Temperature vs Demand",
                             opacity=0.6)
    scatter_chart.plotly_chart(
        fig_scatter, use_container_width=True, key=f"scatter_chart_{hours_to_display}")


# Initial update
update_dashboard()

# Auto refresh using JavaScript only (remove the experimental_rerun part)
if auto_refresh:
    refresh_interval_ms = update_interval * 1000
    st.markdown(f"""
    <script>
        var timer = setTimeout(function() {{
            window.location.reload();
        }}, {refresh_interval_ms});
    </script>
    """, unsafe_allow_html=True)
