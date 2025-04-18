# dashboard/model_evaluation.py
from models.prophet_model import EnergyProphetModel
from data.mongodb_connector import EnergyDatabaseConnector
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import json

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules

# Page configuration
st.set_page_config(
    page_title="Prophet Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Energy Demand Forecasting - Model Evaluation")
st.markdown("""
This dashboard evaluates the Prophet forecasting model performance using Simulated Historical Forecasts (SHFs).
The model is trained on data up to several historical cutoff points and evaluated on actual values that follow.
""")

# Sidebar - Evaluation Parameters
st.sidebar.header("Evaluation Parameters")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["MongoDB", "Upload CSV"],
    help="Select where to get the data for evaluation"
)

if data_source == "MongoDB":
    # Date range selection for MongoDB data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Default to 1 year

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(start_date.date(), end_date.date()),
        help="Select date range for evaluation data"
    )

    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
else:
    # File uploader for CSV
    uploaded_file = st.sidebar.file_uploader(
        "Upload Energy Data CSV",
        type=["csv"],
        help="Upload a CSV file with energy consumption data"
    )

# Model parameters
st.sidebar.subheader("Model Parameters")

n_cutoffs = st.sidebar.slider(
    "Number of Historical Cutoff Points",
    min_value=3,
    max_value=10,
    value=5,
    help="Number of historical points where forecasts will be generated"
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=60,
    value=30,
    help="How many days to forecast at each cutoff point"
)

error_threshold = st.sidebar.slider(
    "Error Threshold (%)",
    min_value=10,
    max_value=50,
    value=20,
    help="Threshold for flagging problematic forecasts"
)

include_temperature = st.sidebar.checkbox(
    "Include Temperature in Model",
    value=True,
    help="Whether to use temperature as a regressor in the model"
)

# Add a button to run evaluation
evaluate_button = st.sidebar.button("Run Evaluation", use_container_width=True)

# Initialize database connector


@st.cache_resource
def get_db_connector():
    """Initialize database connector with caching for performance."""
    return EnergyDatabaseConnector()

# Initialize Prophet model


@st.cache_resource
def get_prophet_model():
    """Initialize Prophet model with caching for performance."""
    return EnergyProphetModel()

# Function to load data


@st.cache_data(ttl=3600)
def load_data(source, start_date=None, end_date=None, csv_file=None):
    """Load data from MongoDB or uploaded CSV."""
    if source == "MongoDB":
        db_connector = get_db_connector()
        data = db_connector.fetch_data(start_date, end_date)
        return data
    else:
        if csv_file is not None:
            try:
                data = pd.read_csv(csv_file)

                # Convert timestamp to datetime if present
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                elif 'ds' in data.columns:
                    data['ds'] = pd.to_datetime(data['ds'])

                return data
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                return None
        return None

# Function to run cross-validation with simulated historical forecasts


def run_cross_validation(df, n_cutoffs, horizon_days, include_temperature=False):
    """
    Run cross-validation on Prophet model using historical cutoffs.

    Args:
        df: DataFrame with ds/timestamp, demand, and temperature
        n_cutoffs: Number of cutoff points to use
        horizon_days: Forecast horizon in days
        include_temperature: Whether to include temperature as a regressor

    Returns:
        Dictionary with evaluation results or None if insufficient data
    """
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("Preparing data for evaluation...")

    # Check if we have data
    if df is None or df.empty:
        status_text.error("No data available for evaluation")
        return None

    # Make a copy to avoid modifying the original dataframe
    prophet_df = df.copy()

    # Check for required columns
    if 'ds' not in prophet_df.columns:
        if 'timestamp' in prophet_df.columns:
            prophet_df['ds'] = prophet_df['timestamp']
        else:
            status_text.error("No datetime column found (ds or timestamp)")
            return None

    if 'demand' not in prophet_df.columns:
        status_text.error("Required 'demand' column not found")
        return None

    # Convert to datetime if not already
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Create the 'y' column that Prophet requires (using demand values)
    prophet_df['y'] = prophet_df['demand']

    # Drop any rows with NaN in key columns
    prophet_df = prophet_df.dropna(subset=['ds', 'demand'])

    # Check again if we have data after cleaning
    if len(prophet_df) < 50:  # Minimum data requirement
        status_text.error(
            f"Insufficient data points for evaluation: {len(prophet_df)} < 50 minimum")
        return None

    progress_bar.progress(10)
    status_text.info("Calculating cutoff dates...")

    # Sort by date
    prophet_df = prophet_df.sort_values('ds')

    # Calculate cutoff dates
    min_date = prophet_df['ds'].min()
    max_date = prophet_df['ds'].max()
    date_range = (max_date - min_date).days

    # Ensure we have enough data for the cutoffs
    if date_range < max(30, horizon_days * 2):  # Need at least this many days of data
        status_text.error(
            f"Insufficient date range: {date_range} days < {max(30, horizon_days * 2)} days minimum")
        return None

    # Adjust number of cutoffs if needed
    if date_range < n_cutoffs * horizon_days * 1.5:  # Need some buffer
        old_n_cutoffs = n_cutoffs
        n_cutoffs = max(1, date_range // (horizon_days * 1.5))
        status_text.warning(
            f"Reduced cutoff points from {old_n_cutoffs} to {n_cutoffs} due to data constraints")

    # Generate cutoff dates - evenly spaced excluding very recent and very old data
    # Don't use the very oldest or newest data
    buffer_days = max(7, horizon_days // 4)
    usable_range = date_range - (2 * buffer_days)

    if usable_range <= 0 or n_cutoffs <= 0:
        status_text.error("Insufficient data range after applying buffers")
        return None

    cutoff_step = usable_range // (n_cutoffs + 1)

    # Ensure cutoff step is at least 1 day
    if cutoff_step < 1:
        cutoff_step = 1
        n_cutoffs = min(n_cutoffs, usable_range - 1)

    cutoff_dates = [min_date + timedelta(days=buffer_days + (cutoff_step * (i+1)))
                    for i in range(n_cutoffs)]

    progress_bar.progress(20)
    status_text.info(f"Running evaluation with {n_cutoffs} cutoff points...")

    # Run cross-validation for each cutoff
    all_results = []
    cutoff_metrics = []

    # Create a prophet model
    model_provider = get_prophet_model

    for i, cutoff_date in enumerate(cutoff_dates):
        try:
            cutoff_progress = 20 + (i * 70 // n_cutoffs)
            progress_bar.progress(cutoff_progress)
            status_text.info(
                f"Evaluating cutoff {i+1}/{n_cutoffs}: {cutoff_date.strftime('%Y-%m-%d')}")

            # Split data based on cutoff date
            train_df = prophet_df[prophet_df['ds'] <= cutoff_date].copy()
            test_df = prophet_df[(prophet_df['ds'] > cutoff_date) &
                                 (prophet_df['ds'] <= cutoff_date + timedelta(days=horizon_days))].copy()

            # Skip if insufficient data
            if len(train_df) < 30 or len(test_df) < 5:
                continue

            # Train model on training data
            model = model_provider()

            # Prepare data for Prophet
            prophet_train = train_df[['ds', 'y']]

            # Add temperature if requested and available
            if include_temperature and 'temperature' in train_df.columns:
                prophet_train['temperature'] = train_df['temperature']
                # Make sure we have no NaN values in temperature column
                prophet_train = prophet_train.dropna(subset=['temperature'])

            # Train the model
            model.train(prophet_train, include_temperature=include_temperature)

            # Generate forecast for test period
            hours_to_forecast = horizon_days * 24
            forecast = model.predict(periods=hours_to_forecast)

            # Merge forecast with actual values for evaluation
            eval_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            # Now merge with test data using demand column as the actual values
            eval_df = eval_df.merge(
                test_df[['ds', 'demand']], on='ds', how='inner')

            # Skip if no matching datapoints
            if len(eval_df) < 5:
                continue

            # Calculate error metrics - using demand column for actual values
            eval_df['error'] = eval_df['demand'] - eval_df['yhat']
            eval_df['abs_error'] = abs(eval_df['error'])
            eval_df['pct_error'] = 100 * \
                abs(eval_df['error'] / eval_df['demand'])

            # Handle divide-by-zero or near-zero values in percentage error calculation
            mask = (eval_df['demand'].abs() < 0.01)
            if mask.any():
                eval_df.loc[mask, 'pct_error'] = eval_df.loc[mask,
                                                             'abs_error'] * 100

            # Add cutoff identifier
            eval_df['cutoff'] = cutoff_date

            # Calculate cutoff-specific metrics
            cutoff_mae = eval_df['abs_error'].mean()
            cutoff_rmse = np.sqrt((eval_df['error'] ** 2).mean())
            cutoff_mape = eval_df['pct_error'].mean()
            cutoff_coverage = ((eval_df['demand'] >= eval_df['yhat_lower']) &
                               (eval_df['demand'] <= eval_df['yhat_upper'])).mean() * 100

            cutoff_metrics.append({
                'cutoff': cutoff_date,
                'mae': cutoff_mae,
                'rmse': cutoff_rmse,
                'mape': cutoff_mape,
                'coverage': cutoff_coverage,
                'samples': len(eval_df)
            })

            all_results.append(eval_df)
        except Exception as e:
            st.error(f"Error processing cutoff {cutoff_date}: {str(e)}")
            continue

    progress_bar.progress(90)
    status_text.info("Combining results...")

    # Combine all evaluation results
    if all_results and len(all_results) > 0:
        try:
            combined_results = pd.concat(all_results)

            # Calculate overall metrics
            metrics = {
                'mae': combined_results['abs_error'].mean(),
                'rmse': np.sqrt((combined_results['error'] ** 2).mean()),
                'mape': combined_results['pct_error'].mean(),
                'median_ape': combined_results['pct_error'].median(),
                'coverage': ((combined_results['demand'] >= combined_results['yhat_lower']) &
                             (combined_results['demand'] <= combined_results['yhat_upper'])).mean() * 100
            }

            progress_bar.progress(100)
            status_text.success("Evaluation completed successfully!")

            return {
                'metrics': metrics,
                'evaluation_df': combined_results,
                'cutoffs': cutoff_dates,
                'cutoff_metrics': pd.DataFrame(cutoff_metrics)
            }
        except Exception as e:
            status_text.error(f"Error combining results: {str(e)}")
            return None
    else:
        status_text.error("No valid results from any cutoff points")
        return None

# Function to display evaluation results


def display_evaluation_results(results):
    """Display evaluation results with visualizations."""
    if not results:
        st.error("No evaluation results to display")
        return

    # 1. Overall Metrics Summary
    st.header("Model Performance Metrics")
    metrics = results['metrics']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    col3.metric("MAPE", f"{metrics['mape']:.2f}%")
    col4.metric("Prediction Interval Coverage", f"{metrics['coverage']:.1f}%")

    st.markdown(f"""
    ### Key Insights
    
    - Model forecasts have an average error of **{metrics['mape']:.2f}%** (MAPE)
    - **{metrics['coverage']:.1f}%** of actual values fall within the prediction intervals
    - The model was evaluated on {len(results['cutoffs'])} different cutoff dates with a {forecast_horizon}-day horizon
    """)

    # 2. Performance by Cutoff Date
    st.header("Performance by Cutoff Date")

    cutoff_metrics_df = results['cutoff_metrics']
    cutoff_metrics_df['cutoff_str'] = cutoff_metrics_df['cutoff'].dt.strftime(
        '%Y-%m-%d')

    # Create the performance chart
    fig_cutoffs = go.Figure()

    # Add MAPE bars
    fig_cutoffs.add_trace(go.Bar(
        x=cutoff_metrics_df['cutoff_str'],
        y=cutoff_metrics_df['mape'],
        name='MAPE (%)',
        marker_color='coral'
    ))

    # Add coverage line
    fig_cutoffs.add_trace(go.Scatter(
        x=cutoff_metrics_df['cutoff_str'],
        y=cutoff_metrics_df['coverage'],
        name='Coverage (%)',
        mode='lines+markers',
        marker=dict(color='royalblue'),
        line=dict(color='royalblue'),
        yaxis='y2'
    ))

    # Update layout
    fig_cutoffs.update_layout(
        title='Model Performance by Cutoff Date',
        xaxis_title='Cutoff Date',
        yaxis=dict(
            title='MAPE (%)',
            side='left'
        ),
        yaxis2=dict(
            title='Coverage (%)',
            side='right',
            overlaying='y',
            range=[0, 100]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )

    st.plotly_chart(fig_cutoffs, use_container_width=True)

    # 3. Detailed Metrics Table
    with st.expander("Detailed Metrics by Cutoff"):
        display_df = cutoff_metrics_df[[
            'cutoff_str', 'mae', 'rmse', 'mape', 'coverage', 'samples']]
        display_df.columns = ['Cutoff Date', 'MAE',
                              'RMSE', 'MAPE (%)', 'Coverage (%)', 'Samples']
        st.dataframe(display_df)

        # Add download button for detailed metrics
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name="model_evaluation_metrics.csv",
            mime="text/csv"
        )

    # 4. Forecast Error Analysis
    st.header("Forecast Error Analysis")

    eval_df = results['evaluation_df']

    # Calculate days from cutoff
    eval_df['days_from_cutoff'] = (
        eval_df['ds'] - eval_df['cutoff']).dt.total_seconds() / (24 * 3600)

    # Error vs Forecast Horizon
    horizon_error = eval_df.groupby(eval_df['days_from_cutoff'].round().astype(int))[
        'pct_error'].mean().reset_index()
    horizon_error.columns = ['Days from Cutoff', 'Average Error (%)']

    fig_horizon = px.line(
        horizon_error,
        x='Days from Cutoff',
        y='Average Error (%)',
        title='Error by Forecast Horizon',
        markers=True
    )

    fig_horizon.update_layout(
        xaxis_title='Days from Cutoff Date',
        yaxis_title='Average Percentage Error (%)',
        height=400
    )

    st.plotly_chart(fig_horizon, use_container_width=True)

    # 5. Error Distribution
    col1, col2 = st.columns(2)

    with col1:
        # Error distribution histogram
        fig_error_dist = px.histogram(
            eval_df,
            x='pct_error',
            nbins=30,
            title='Error Distribution',
            color_discrete_sequence=['lightseagreen']
        )

        fig_error_dist.update_layout(
            xaxis_title='Percentage Error (%)',
            yaxis_title='Frequency',
            height=400
        )

        # Add vertical line for threshold
        fig_error_dist.add_vline(
            x=error_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({error_threshold}%)",
            annotation_position="top right"
        )

        st.plotly_chart(fig_error_dist, use_container_width=True)

    with col2:
        # Actual vs Predicted scatter plot
        fig_scatter = px.scatter(
            eval_df,
            x='demand',
            y='yhat',
            title='Actual vs Predicted Values',
            opacity=0.6,
            color='pct_error',
            color_continuous_scale='RdYlGn_r'
        )

        # Add perfect prediction line
        max_val = max(eval_df['demand'].max(), eval_df['yhat'].max())
        min_val = min(eval_df['demand'].min(), eval_df['yhat'].min())

        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Perfect Prediction'
            )
        )

        fig_scatter.update_layout(
            xaxis_title='Actual Demand',
            yaxis_title='Predicted Demand',
            height=400
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    # 6. Problematic Forecasts Analysis
    st.header("Problematic Forecasts Analysis")

    # Find forecasts with errors above threshold
    problem_df = eval_df[eval_df['pct_error'] > error_threshold].copy()

    if len(problem_df) > 0:
        problem_percent = (len(problem_df) / len(eval_df)) * 100

        st.warning(
            f"Found {len(problem_df)} problematic forecasts ({problem_percent:.1f}% of all forecasts) with errors > {error_threshold}%")

        # Group problems by cutoff
        problem_summary = problem_df.groupby('cutoff').agg(
            avg_error=('pct_error', 'mean'),
            max_error=('pct_error', 'max'),
            count=('pct_error', 'count')
        ).reset_index()

        problem_summary['cutoff'] = problem_summary['cutoff'].dt.strftime(
            '%Y-%m-%d')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Error Summary by Cutoff Date")

            # Format the dataframe for display
            display_summary = problem_summary.copy()
            display_summary.columns = ['Cutoff Date',
                                       'Avg Error (%)', 'Max Error (%)', 'Count']
            display_summary['Avg Error (%)'] = display_summary['Avg Error (%)'].round(
                2)
            display_summary['Max Error (%)'] = display_summary['Max Error (%)'].round(
                2)

            st.dataframe(display_summary)

        with col2:
            # Create bar chart of problem counts by cutoff
            fig_problems = px.bar(
                problem_summary,
                x='cutoff',
                y='count',
                title='Problem Forecast Count by Cutoff Date',
                color='avg_error',
                color_continuous_scale='Reds',
                text='count'
            )

            fig_problems.update_layout(
                xaxis_title='Cutoff Date',
                yaxis_title='Number of Problem Forecasts',
                coloraxis_colorbar_title='Avg Error (%)'
            )

            st.plotly_chart(fig_problems, use_container_width=True)

        # Top 10 largest errors
        st.subheader("Top 10 Largest Errors")

        worst_forecasts = problem_df.sort_values(
            'pct_error', ascending=False).head(10).copy()
        worst_forecasts['cutoff'] = worst_forecasts['cutoff'].dt.strftime(
            '%Y-%m-%d')
        worst_forecasts['ds'] = worst_forecasts['ds'].dt.strftime(
            '%Y-%m-%d %H:%M')

        worst_display = worst_forecasts[[
            'ds', 'demand', 'yhat', 'error', 'pct_error', 'cutoff']].copy()
        worst_display.columns = ['Timestamp', 'Actual',
                                 'Predicted', 'Error', 'Error (%)', 'Cutoff Date']
        worst_display['Error (%)'] = worst_display['Error (%)'].round(2)

        st.dataframe(worst_display)
    else:
        st.success(
            f"No forecasts with errors greater than {error_threshold}% were found!")


# Main app flow
if data_source == "MongoDB":
    # Load data from MongoDB based on date range
    data = load_data("MongoDB", start_date, end_date)

    if data is not None:
        st.info(
            f"Loaded {len(data)} records from MongoDB from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    else:
        st.error("Failed to load data from MongoDB")
else:
    # Handle uploaded CSV file
    if uploaded_file is not None:
        data = load_data("Upload", csv_file=uploaded_file)

        if data is not None:
            st.info(f"Loaded {len(data)} records from uploaded CSV")
        else:
            st.error("Failed to load data from uploaded CSV")
    else:
        st.warning("Please upload a CSV file")
        data = None

# Data preview
if data is not None:
    with st.expander("Data Preview"):
        st.dataframe(data.head())

        # Show column info
        st.subheader("Column Information")

        # Check for important columns
        required_cols = ["ds", "timestamp", "demand", "temperature"]
        present_cols = [col for col in required_cols if col in data.columns]
        missing_cols = [
            col for col in required_cols if col not in data.columns]

        if missing_cols:
            st.warning(
                f"Missing some useful columns: {', '.join(missing_cols)}")

        # Display column types and stats
        col_info = pd.DataFrame({
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Unique Values': [data[col].nunique() for col in data.columns],
            'Min': [data[col].min() if pd.api.types.is_numeric_dtype(data[col]) else None for col in data.columns],
            'Max': [data[col].max() if pd.api.types.is_numeric_dtype(data[col]) else None for col in data.columns]
        })

        st.dataframe(col_info)

# Run evaluation if button is clicked
if evaluate_button and data is not None:
    with st.spinner("Running model evaluation..."):
        results = run_cross_validation(
            data,
            n_cutoffs=n_cutoffs,
            horizon_days=forecast_horizon,
            include_temperature=include_temperature
        )

    if results:
        display_evaluation_results(results)
    else:
        st.error("Evaluation failed. Please check the data and parameters.")

# Simple instructions at the bottom
st.markdown("""
---
### How to Use This Dashboard

1. **Select Data Source**: Choose MongoDB or upload a CSV file
2. **Configure Parameters**: Adjust the evaluation settings in the sidebar
3. **Run Evaluation**: Click the "Run Evaluation" button
4. **Analyze Results**: Explore the various visualizations and metrics

The dashboard will evaluate the model at multiple historical cutoff points and compute accuracy metrics.
""")

# Add information about the error metrics
with st.expander("Understanding Error Metrics"):
    st.markdown("""
    ### Evaluation Metrics Explained
    
    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
    - **RMSE (Root Mean Squared Error)**: Square root of the average squared differences
    - **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predicted and actual values
    - **Coverage**: Percentage of actual values that fall within the prediction intervals
    
    ### Simulated Historical Forecasts
    
    This evaluation approach:
    1. Divides the historical data into multiple cutoff points
    2. For each cutoff, trains the model on data up to that point
    3. Generates forecasts for the future period (horizon)
    4. Compares these forecasts to the actual values that follow the cutoff
    5. Aggregates metrics across all cutoffs to assess overall performance
    
    This simulates how the model would have performed if it had been used at different points in the past.
    """)

if __name__ == "__main__":
    # This allows the file to be run directly
    # streamlit run src/dashboard/model_evaluation.py
    pass
