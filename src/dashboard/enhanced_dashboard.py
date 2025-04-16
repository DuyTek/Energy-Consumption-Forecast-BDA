# enhanced_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Import your own modules
from data.mongodb_connector import EnergyDatabaseConnector
from analysis.enhanced_energy_analysis import EnhancedEnergyAnalysis

# Page configuration
st.set_page_config(
    page_title="Enhanced Energy Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connector


@st.cache_resource
def get_db_connector():
    """Initialize database connector with caching for performance."""
    # You may need to modify this based on your actual MongoDB configuration
    return EnergyDatabaseConnector()


db_connector = get_db_connector()

# Sidebar controls
st.sidebar.header("Analysis Controls")

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Default to last 30 days

start_date_input = st.sidebar.date_input(
    "Start Date",
    value=start_date,
    max_value=end_date
)

end_date_input = st.sidebar.date_input(
    "End Date",
    value=end_date,
    min_value=start_date_input,
    max_value=end_date
)

# Ensure datetime format
start_date = datetime.combine(start_date_input, datetime.min.time())
end_date = datetime.combine(end_date_input, datetime.max.time())

# Peak threshold selection
peak_threshold = st.sidebar.slider(
    "Peak Threshold Percentile",
    min_value=80,
    max_value=99,
    value=90,
    step=1,
    help="Percentile threshold for classifying peak demand periods"
)

# Optional data file upload
uploaded_file = st.sidebar.file_uploader(
    "Or upload data file (CSV)",
    type=["csv"]
)

# Load data


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(start_date, end_date, uploaded_file=None):
    """Load data from database or uploaded file."""
    if uploaded_file is not None:
        # Load from uploaded file
        data = pd.read_csv(uploaded_file)

        # Convert timestamp to datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        elif 'ds' in data.columns:
            data['timestamp'] = pd.to_datetime(data['ds'])
            if 'y' in data.columns and 'demand' not in data.columns:
                data['demand'] = data['y']

        st.sidebar.success(f"Loaded {len(data)} records from uploaded file")
        return data
    else:
        # Load from database
        data = db_connector.fetch_data(start_date, end_date)

        if data is None or len(data) == 0:
            st.warning(
                "No data available for the selected date range. Please upload a file or change the date range.")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['timestamp', 'demand', 'temperature'])

        st.sidebar.success(f"Loaded {len(data)} records from database")
        return data


# Load data based on selection
data = load_data(start_date, end_date, uploaded_file)

# Main content
st.title("Enhanced Energy Consumption Analysis")

# Data overview
with st.expander("Data Overview", expanded=False):
    st.write(
        f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    st.write(f"Total records: {len(data)}")

    if not data.empty:
        # Show basic statistics
        st.subheader("Basic Statistics")
        if 'demand' in data.columns:
            demand_stats = data['demand'].describe()
            temp_stats = data['temperature'].describe(
            ) if 'temperature' in data.columns else None

            col1, col2 = st.columns(2)
            with col1:
                st.write("Demand Statistics")
                st.dataframe(demand_stats)

            with col2:
                if temp_stats is not None:
                    st.write("Temperature Statistics")
                    st.dataframe(temp_stats)

        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(data.head())

# Only proceed with analysis if we have data
if not data.empty and 'demand' in data.columns:
    # Initialize the enhanced analysis
    analyzer = EnhancedEnergyAnalysis(data)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Hourly Patterns",
        "Weekday vs Weekend",
        "Peak Period Analysis",
        "Temperature Impact"
    ])

    with tab1:
        st.header("Hourly Energy Consumption Patterns")

        # Get hourly analysis results
        hourly_analysis = analyzer.analyze_hourly_patterns()

        # Display key insights
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Peak Hours",
                ", ".join(
                    [f"{hour}:00" for hour in hourly_analysis['peak_hours']])
            )

        with col2:
            st.metric(
                "Peak vs Average",
                f"{hourly_analysis['peak_hour_pct_above_avg']:.1f}% higher"
            )

        with col3:
            st.metric(
                "Overall Average Demand",
                f"{hourly_analysis['overall_avg']:.2f} kWh"
            )

        # Plot hourly patterns
        st.plotly_chart(analyzer.plot_hourly_patterns(),
                        use_container_width=True)

        # Show detailed hourly statistics
        st.subheader("Detailed Hourly Statistics")
        hourly_stats_df = pd.DataFrame(hourly_analysis['hourly_stats'])
        st.dataframe(hourly_stats_df)

    with tab2:
        st.header("Weekday vs Weekend Comparison")

        # Get weekday/weekend comparison
        weekday_weekend = analyzer.compare_weekday_weekend()

        # Display key insights
        col1, col2, col3 = st.columns(3)

        with col1:
            weekday_peak = ", ".join(
                [f"{hour}:00" for hour in weekday_weekend['weekday_peak_hours']])
            weekend_peak = ", ".join(
                [f"{hour}:00" for hour in weekday_weekend['weekend_peak_hours']])
            st.metric("Weekday Peak Hours", weekday_peak)
            st.metric("Weekend Peak Hours", weekend_peak)

        with col2:
            st.metric("Weekday Avg Demand",
                      f"{weekday_weekend['weekday_avg']:.2f} kWh")
            st.metric("Weekend Avg Demand",
                      f"{weekday_weekend['weekend_avg']:.2f} kWh")

        with col3:
            pct_diff = weekday_weekend['weekend_weekday_pct_diff']
            direction = "higher" if pct_diff > 0 else "lower"
            st.metric(
                "Weekend vs Weekday",
                f"{abs(pct_diff):.1f}% {direction}"
            )

        # Plot weekday vs weekend comparison
        st.plotly_chart(analyzer.plot_weekday_weekend_comparison(),
                        use_container_width=True)

    with tab3:
        st.header("Peak Period Analysis")

        # Get peak period analysis
        peak_analysis = analyzer.quantify_peak_periods(peak_threshold)

        # Display key insights
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Peak Threshold",
                      f"{peak_analysis['threshold']:.2f} kWh")
            st.metric("Peak Periods", f"{peak_analysis['peak_count']} periods")

        with col2:
            st.metric(
                "Peak Percentage",
                f"{peak_analysis['peak_percentage']:.1f}% of time"
            )
            st.metric(
                "Avg Peak Duration",
                f"{peak_analysis['avg_peak_duration']:.1f} intervals"
            )

        with col3:
            st.metric(
                "Peak vs Non-Peak",
                f"{peak_analysis['peak_multiplier']:.1f}x higher"
            )
            st.metric(
                "Peak Avg Demand",
                f"{peak_analysis['peak_avg_demand']:.2f} kWh"
            )

        # Plot peak period analysis
        st.plotly_chart(analyzer.plot_peak_period_analysis(
            peak_threshold), use_container_width=True)

    with tab4:
        st.header("Temperature Impact Analysis")

        # Get temperature impact analysis
        temp_analysis = analyzer.analyze_temperature_impact()

        # Display key insights
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Temperature Correlation",
                f"{temp_analysis['overall_correlation']:.3f}"
            )

        with col2:
            st.metric(
                "Peak Hours Temp Correlation",
                f"{temp_analysis['peak_hours_temp_corr']:.3f}"
            )
            st.metric(
                "Off-Peak Hours Temp Correlation",
                f"{temp_analysis['off_peak_temp_corr']:.3f}"
            )

        with col3:
            st.metric(
                "Weekday Temp Correlation",
                f"{temp_analysis['weekday_temp_corr']:.3f}"
            )
            st.metric(
                "Weekend Temp Correlation",
                f"{temp_analysis['weekend_temp_corr']:.3f}"
            )

        # Plot temperature-demand heatmap
        st.plotly_chart(analyzer.plot_temperature_demand_heatmap(),
                        use_container_width=True)

        # Show hourly temperature sensitivity
        st.subheader("Hourly Temperature Sensitivity")

        hourly_temp_sensitivity = temp_analysis['hourly_temp_sensitivity']
        sensitivity_df = pd.DataFrame({
            'Hour': list(hourly_temp_sensitivity.keys()),
            'Temperature Correlation': list(hourly_temp_sensitivity.values())
        }).sort_values('Hour')

        # Plot hourly temperature sensitivity
        fig = px.bar(
            sensitivity_df,
            x='Hour',
            y='Temperature Correlation',
            title="Temperature Sensitivity by Hour",
            color='Temperature Correlation',
            color_continuous_scale='RdBu_r'
        )

        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Temperature-Demand Correlation",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(
        "No data available for analysis. Please upload a file or select a different date range.")

# Add explanation of the analysis at the bottom
with st.expander("About this analysis"):
    st.markdown("""
    ## Energy Consumption Pattern Analysis
    
    This dashboard provides an enhanced analysis of energy consumption patterns with a focus on:
    
    1. **Hourly Patterns**: Identifies peak consumption hours and quantifies how much higher they are compared to the average
    
    2. **Weekday vs Weekend Comparison**: Analyzes differences in consumption patterns between weekdays and weekends
    
    3. **Peak Period Analysis**: Quantifies peak periods, their frequency, and magnitude using a percentile-based threshold
    
    4. **Temperature Impact**: Analyzes how temperature affects energy demand across different times and conditions
    
    This analysis can help identify opportunities for energy optimization through:
    - Load shifting from peak to off-peak hours
    - Targeted efficiency measures during high-consumption periods
    - Better understanding of temperature impacts to optimize HVAC systems
    - Differentiated strategies for weekday vs weekend operations
    
    The results can be used in your report to provide data-driven recommendations for energy optimization.
    """)

# Add a section for downloading analysis results as JSON
st.sidebar.header("Export Results")

if not data.empty and 'demand' in data.columns:
    # Collect all analysis results
    all_results = {
        "hourly_patterns": analyzer.analyze_hourly_patterns(),
        "weekday_weekend_comparison": analyzer.compare_weekday_weekend(),
        "peak_period_analysis": analyzer.quantify_peak_periods(peak_threshold),
        "temperature_impact": analyzer.analyze_temperature_impact()
    }

    # Convert to JSON
    results_json = json.dumps(all_results, indent=2, default=str)

    st.sidebar.download_button(
        label="Download Analysis Results (JSON)",
        data=results_json,
        file_name="energy_analysis_results.json",
        mime="application/json"
    )
