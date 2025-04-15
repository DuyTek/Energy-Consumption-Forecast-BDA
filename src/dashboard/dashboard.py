# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
import seaborn as sns

from data.mongodb_connector import EnergyDatabaseConnector
from models.prophet_model import EnergyProphetModel

# Setup
st.set_page_config(page_title="Energy Consumption Dashboard", layout="wide")
st.title("Real-Time Energy Consumption Dashboard")

# Initialize database connector


@st.cache_resource
def get_db_connector():
    """Initialize database connector with caching for performance."""
    conn_string = st.secrets.get("mongodb_uri")
    return EnergyDatabaseConnector(connection_string=conn_string)


db_connector = get_db_connector()

# Initialize Prophet model


@st.cache_resource
def get_prophet_model():
    """Initialize Prophet model with caching for performance."""
    return EnergyProphetModel()


prophet_model = get_prophet_model()

# Sidebar controls
st.sidebar.header("Dashboard Controls")
hours_to_display = st.sidebar.slider("Hours to Display", 1, 48, 6)
update_interval = st.sidebar.slider("Update Interval (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto Refresh", True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Real-Time Monitoring", "Patterns Analysis", "Heatmap View", "Forecasting"]
)

with tab1:
    # Main content for real-time monitoring
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

with tab2:
    st.subheader("Patterns Analysis")
    # Add pattern analysis content here
    patterns_placeholder = st.empty()

with tab3:
    st.subheader("Energy Demand Heatmap")
    heatmap_description = st.markdown("""
    This heatmap shows the average energy demand by hour of day and day of week.
    Darker colors represent higher energy demand.
    """)

    # Controls for heatmap
    heatmap_col1, heatmap_col2 = st.columns(2)
    with heatmap_col1:
        days_for_heatmap = st.slider("Days of data for heatmap", 1, 30, 7)

    with heatmap_col2:
        heatmap_metric = st.selectbox(
            "Heatmap metric",
            ["demand", "temperature"],
            index=0
        )

    heatmap_placeholder = st.empty()  # Placeholder for heatmap

# New Forecasting tab
with tab4:
    st.subheader("Energy Demand Forecasting")

    # Controls for forecasting
    forecast_col1, forecast_col2 = st.columns(2)

    with forecast_col1:
        forecast_days = st.slider("Forecast Days", 1, 90, 30)

    with forecast_col2:
        train_days = st.slider("Training Data (days)", 7, 365, 30)
        include_temperature = st.checkbox("Include Temperature", value=True)

    forecast_description = st.markdown("""
    This forecast uses Prophet to predict future energy demand based on historical patterns.
    Use the controls to adjust the forecast period and training data window.
    """)

    # Placeholders for forecast outputs
    forecast_status = st.empty()
    forecast_chart = st.empty()
    components_chart = st.empty()
    forecast_metrics = st.empty()

# Function to update dashboard


def update_dashboard():
    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours_to_display)
    heatmap_start_date = end_date - timedelta(days=days_for_heatmap)

    # For forecasting - get longer training period
    forecast_start_date = end_date - timedelta(days=train_days)

    data = db_connector.fetch_data(start_date, end_date)
    heatmap_data = db_connector.fetch_data(heatmap_start_date, end_date)
    forecast_data = db_connector.fetch_data(forecast_start_date, end_date)

    if len(data) == 0 or data.empty:
        st.warning("No data available for the selected time range.")
        return

    # Update tab 1: Real-Time Monitoring
    with tab1:
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

    # Update tab 2: Patterns Analysis
    with tab2:
        # Simple pattern analysis example
        if len(data) > 0:
            # Group by hour
            hourly_data = data.groupby(data['timestamp'].dt.hour)[
                'demand'].mean().reset_index()
            hourly_data.columns = ['Hour', 'Avg Demand']

            fig_patterns = px.bar(
                hourly_data,
                x='Hour',
                y='Avg Demand',
                title=f"Average Demand by Hour (Last {hours_to_display} Hours)"
            )
            patterns_placeholder.plotly_chart(
                fig_patterns, use_container_width=True)

    # Update tab 3: Heatmap View
    with tab3:
        if len(heatmap_data) > 0:
            # Ensure we have day_of_week and hour
            if 'day_of_week' not in heatmap_data.columns:
                heatmap_data['day_of_week'] = heatmap_data['timestamp'].dt.dayofweek
            if 'hour' not in heatmap_data.columns:
                heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour

            # Create pivot table
            pivot_data = heatmap_data.pivot_table(
                values=heatmap_metric,
                index='hour',
                columns='day_of_week',
                aggfunc='mean'
            )

            # Day names for better readability
            day_names = ['Monday', 'Tuesday', 'Wednesday',
                         'Thursday', 'Friday', 'Saturday', 'Sunday']

            # Create heatmap using plotly
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=[day_names[i] for i in pivot_data.columns],
                y=pivot_data.index,
                colorscale='YlOrRd',
                colorbar=dict(title=f'Avg {heatmap_metric.capitalize()}')
            ))

            fig_heatmap.update_layout(
                title=f'Average {heatmap_metric.capitalize()} by Hour and Day of Week (Last {days_for_heatmap} Days)',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day',
                height=600
            )

            heatmap_placeholder.plotly_chart(
                fig_heatmap, use_container_width=True)

    # Update tab 4: Forecasting
    with tab4:
        if len(forecast_data) > 0:
            with forecast_status.container():
                st.info(
                    f"Training Prophet model on {len(forecast_data)} data points from the last {train_days} days...")

            try:
                # Prepare data for Prophet
                # Prophet requires columns 'ds' and 'y'
                prophet_df = forecast_data.copy()

                if 'ds' not in prophet_df.columns and 'timestamp' in prophet_df.columns:
                    prophet_df['ds'] = prophet_df['timestamp']

                if 'y' not in prophet_df.columns and 'demand' in prophet_df.columns:
                    prophet_df['y'] = prophet_df['demand']

                # Train the model
                prophet_model.train(
                    prophet_df, include_temperature=include_temperature)

                # Generate forecast
                # Convert days to hours for hourly forecasting
                future_periods = forecast_days * 24
                forecast = prophet_model.predict(periods=future_periods)

                # Convert the forecast to a Plotly figure
                fig_forecast = go.Figure()

                # Add actual values
                fig_forecast.add_trace(go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    mode='markers',
                    name='Actual',
                    marker=dict(size=4, color='blue', opacity=0.6)
                ))

                # Add forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))

                # Add uncertainty intervals
                fig_forecast.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'].iloc[::-1]]),
                    y=pd.concat([forecast['yhat_upper'],
                                forecast['yhat_lower'].iloc[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Uncertainty Interval'
                ))

                fig_forecast.update_layout(
                    title='Energy Demand Forecast',
                    xaxis_title='Date',
                    yaxis_title='Demand',
                    height=500
                )

                forecast_chart.plotly_chart(
                    fig_forecast, use_container_width=True)

                # Display forecast components
                fig_components = go.Figure()

                # Weekly seasonality
                if 'weekly' in forecast.columns:
                    weekly_df = forecast[['ds', 'weekly']].drop_duplicates()
                    # Get day names
                    weekly_df['day'] = weekly_df['ds'].dt.day_name()
                    weekly_df = weekly_df.sort_values(
                        'ds').drop_duplicates('day')

                    fig_components.add_trace(go.Bar(
                        x=weekly_df['day'],
                        y=weekly_df['weekly'],
                        name='Weekly Pattern'
                    ))

                # Daily seasonality
                if 'daily' in forecast.columns:
                    daily_df = forecast[['ds', 'daily']].drop_duplicates()
                    # Get hour
                    daily_df['hour'] = daily_df['ds'].dt.hour
                    daily_df = daily_df.sort_values(
                        'hour').drop_duplicates('hour')

                    fig_components.add_trace(go.Scatter(
                        x=daily_df['hour'],
                        y=daily_df['daily'],
                        mode='lines+markers',
                        name='Daily Pattern'
                    ))

                fig_components.update_layout(
                    title='Forecast Components',
                    xaxis_title='Time Component',
                    yaxis_title='Effect on Demand',
                    height=400
                )

                components_chart.plotly_chart(
                    fig_components, use_container_width=True)

                # Display forecast metrics
                with forecast_metrics.container():
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    # Calculate average metrics over the forecast period
                    forecast_avg = forecast['yhat'].mean()
                    forecast_max = forecast['yhat'].max()
                    forecast_min = forecast['yhat'].min()

                    metrics_col1.metric(
                        "Average Forecasted Demand", f"{forecast_avg:.2f} kWh")
                    metrics_col2.metric(
                        "Maximum Forecasted Demand", f"{forecast_max:.2f} kWh")
                    metrics_col3.metric(
                        "Minimum Forecasted Demand", f"{forecast_min:.2f} kWh")

                    # Extract peak hours
                    forecast_hourly = forecast.copy()
                    forecast_hourly['hour'] = forecast_hourly['ds'].dt.hour
                    peak_hours = forecast_hourly.groupby(
                        'hour')['yhat'].mean().nlargest(3)

                    st.markdown(
                        f"**Peak Demand Hours:** {', '.join([f'{h}:00' for h in peak_hours.index])}")

                with forecast_status.container():
                    st.success("Forecast generated successfully!")

            except Exception as e:
                with forecast_status.container():
                    st.error(f"Error generating forecast: {str(e)}")


# Initial update
update_dashboard()

# Auto refresh using JavaScript only
if auto_refresh:
    refresh_interval_ms = update_interval * 1000
    st.markdown(f"""
    <script>
        var timer = setTimeout(function() {{
            window.location.reload();
        }}, {refresh_interval_ms});
    </script>
    """, unsafe_allow_html=True)
