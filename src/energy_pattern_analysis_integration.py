# energy_pattern_analysis_integration.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import json

# Import your existing modules
from data.mongodb_connector import EnergyDatabaseConnector
from models.prophet_model import EnergyProphetModel

# Import the new enhanced analysis module
from analysis.enhanced_energy_analysis import EnhancedEnergyAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("energy_pattern_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnergyPatternAnalysis")


def run_enhanced_analysis(config_path=None):
    """Run enhanced energy pattern analysis.

    Args:
        config_path: Path to configuration file

    Returns:
        Dict containing all analysis results and paths to saved figures
    """
    logger.info("Starting enhanced energy consumption pattern analysis")

    # Load configuration
    config = load_config(config_path)

    # Initialize database connector
    db_connector = EnergyDatabaseConnector()

    # Load data
    data_source_config = config['data_source']
    if data_source_config['type'] == 'csv':
        csv_path = data_source_config['csv_path']

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return None

        logger.info(f"Loading data from CSV: {csv_path}")

        # Either load directly or through database
        if config['database']['use_mongodb']:
            df = db_connector.load_from_csv(csv_path)
        else:
            df = pd.read_csv(csv_path)

            # Convert timestamp/ds to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])

            # Add temporal features if needed
            if 'ds' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = df['ds']

            if 'y' in df.columns and 'demand' not in df.columns:
                df['demand'] = df['y']

            if 'timestamp' in df.columns:
                if 'hour' not in df.columns:
                    df['hour'] = df['timestamp'].dt.hour
                if 'day_of_week' not in df.columns:
                    df['day_of_week'] = df['timestamp'].dt.dayofweek
                if 'month' not in df.columns:
                    df['month'] = df['timestamp'].dt.month
    else:
        # For other data source types, fetch from database
        days_to_analyze = config['analysis'].get('days_to_analyze', 30)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_analyze)

        logger.info(f"Fetching {days_to_analyze} days of data from database")
        df = db_connector.fetch_data(start_date, end_date)

    if df is None or len(df) == 0:
        logger.error("No data available for analysis")
        return None

    logger.info(f"Loaded {len(df)} records for analysis")

    # Initialize enhanced analysis
    analyzer = EnhancedEnergyAnalysis(df)

    # Run analyses
    logger.info("Running enhanced pattern analyses")

    # Set peak threshold from config
    peak_threshold = config['analysis'].get('peak_threshold_percentile', 90)

    analysis_results = {
        'hourly_patterns': analyzer.analyze_hourly_patterns(),
        'weekday_weekend_comparison': analyzer.compare_weekday_weekend(),
        'peak_period_analysis': analyzer.quantify_peak_periods(peak_threshold),
        'temperature_impact': analyzer.analyze_temperature_impact()
    }

    # Create figures directory if it doesn't exist
    figures_dir = config['analysis'].get('figures_directory', 'figures')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{figures_dir}/pattern_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save figures
    logger.info(f"Saving figures to {output_dir}")

    figure_paths = {}

    # Save hourly patterns figure
    hourly_fig = analyzer.plot_hourly_patterns()
    hourly_path = f"{output_dir}/hourly_patterns.png"
    hourly_fig.write_image(hourly_path, width=1200, height=800)
    figure_paths['hourly_patterns'] = hourly_path

    # Save weekday vs weekend comparison figure
    weekday_weekend_fig = analyzer.plot_weekday_weekend_comparison()
    weekday_weekend_path = f"{output_dir}/weekday_weekend_comparison.png"
    weekday_weekend_fig.write_image(
        weekday_weekend_path, width=1200, height=800)
    figure_paths['weekday_weekend_comparison'] = weekday_weekend_path

    # Save peak period analysis figure
    peak_fig = analyzer.plot_peak_period_analysis(peak_threshold)
    peak_path = f"{output_dir}/peak_period_analysis.png"
    peak_fig.write_image(peak_path, width=1200, height=800)
    figure_paths['peak_period_analysis'] = peak_path

    # Save temperature demand heatmap figure
    temp_heatmap_fig = analyzer.plot_temperature_demand_heatmap()
    temp_heatmap_path = f"{output_dir}/temperature_demand_heatmap.png"
    temp_heatmap_fig.write_image(temp_heatmap_path, width=1200, height=800)
    figure_paths['temperature_demand_heatmap'] = temp_heatmap_path

    # Save analysis results to JSON
    results_path = f"{output_dir}/analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    logger.info(f"Analysis results saved to {results_path}")

    # Combine results with figure paths
    combined_results = {
        'analysis_results': analysis_results,
        'figure_paths': figure_paths,
        'timestamp': timestamp
    }

    logger.info("Enhanced pattern analysis completed successfully")

    return combined_results


def load_config(config_path):
    """Load configuration from JSON file."""
    default_config = {
        'data_source': {
            'type': 'csv',
            'csv_path': 'data/energy_data.csv',
            'use_real_time': False
        },
        'database': {
            'use_mongodb': False,
            'connection_string': 'mongodb://localhost:27017/'
        },
        'analysis': {
            'forecast_periods': 365,
            'save_figures': True,
            'figures_directory': 'figures',
            'days_to_analyze': 30,
            'peak_threshold_percentile': 90
        }
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

            # Update default config with user values
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d:
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            config = update_dict(default_config, user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            config = default_config
    else:
        config = default_config

    return config


def generate_pattern_analysis_report(analysis_results):
    """Generate a summary report from pattern analysis results.

    Args:
        analysis_results: Dict containing analysis results

    Returns:
        str: Markdown-formatted report
    """
    if not analysis_results:
        return "No analysis results available."

    results = analysis_results['analysis_results']

    # Extract key insights
    hourly = results['hourly_patterns']
    weekday_weekend = results['weekday_weekend_comparison']
    peak = results['peak_period_analysis']
    temp = results['temperature_impact']

    # Format peak hours as strings with AM/PM
    def format_hour(hour):
        if hour == 0:
            return "12 AM"
        elif hour < 12:
            return f"{hour} AM"
        elif hour == 12:
            return "12 PM"
        else:
            return f"{hour-12} PM"

    peak_hours_str = ", ".join([format_hour(hour)
                               for hour in hourly['peak_hours']])
    weekday_peak_str = ", ".join(
        [format_hour(hour) for hour in weekday_weekend['weekday_peak_hours']])
    weekend_peak_str = ", ".join(
        [format_hour(hour) for hour in weekday_weekend['weekend_peak_hours']])

    # Generate report in Markdown format
    report = f"""
    # Energy Consumption Pattern Analysis Report
    
    ## Key Findings
    
    ### Hourly Consumption Patterns
    
    - **Peak Hours**: {peak_hours_str}
    - **Peak vs Average**: Peak hours are **{hourly['peak_hour_pct_above_avg']:.1f}%** higher than the overall average
    - **Overall Average Demand**: {hourly['overall_avg']:.2f} kWh
    
    ### Weekday vs Weekend Patterns
    
    - **Weekday Peak Hours**: {weekday_peak_str}
    - **Weekend Peak Hours**: {weekend_peak_str}
    - **Weekday Average**: {weekday_weekend['weekday_avg']:.2f} kWh
    - **Weekend Average**: {weekday_weekend['weekend_avg']:.2f} kWh
    - **Difference**: Weekend demand is {abs(weekday_weekend['weekend_weekday_pct_diff']):.1f}% {'higher' if weekday_weekend['weekend_weekday_pct_diff'] > 0 else 'lower'} than weekday demand
    
    ### Peak Period Analysis
    
    - **Peak Threshold**: {peak['threshold']:.2f} kWh
    - **Peak Periods**: {peak['peak_count']} periods, representing {peak['peak_percentage']:.1f}% of the time
    - **Peak vs Non-Peak**: Peak demand is {peak['peak_multiplier']:.1f}x higher than non-peak demand
    - **Average Peak Duration**: {peak['avg_peak_duration']:.1f} time intervals
    
    ### Temperature Impact
    
    - **Overall Temperature Correlation**: {temp['overall_correlation']:.3f}
    - **Peak Hours Temperature Correlation**: {temp['peak_hours_temp_corr']:.3f}
    - **Off-Peak Hours Temperature Correlation**: {temp['off_peak_temp_corr']:.3f}
    - **Weekday Temperature Correlation**: {temp['weekday_temp_corr']:.3f}
    - **Weekend Temperature Correlation**: {temp['weekend_temp_corr']:.3f}
    
    ## Optimization Opportunities
    
    Based on the analysis, the following optimization opportunities have been identified:
    
    1. **Load Shifting**: Shift non-critical operations from peak hours ({peak_hours_str}) to off-peak hours
    
    2. **Weekday-Specific Measures**: Implement specific strategies for weekdays, focusing on the {weekday_peak_str} peak periods
    
    3. **Weekend-Specific Measures**: For weekend operations, focus on {weekend_peak_str} periods
    
    4. **Temperature-Based Optimization**: 
       {'Cooling optimization is critical during peak hours as temperature has a strong positive correlation' if temp['peak_hours_temp_corr'] > 0.5 else 'Heating optimization is critical during peak hours as temperature has a strong negative correlation' if temp['peak_hours_temp_corr'] < -0.5 else 'Temperature has a moderate impact on energy consumption during peak hours'}
    
    ## Recommendations for Further Analysis
    
    - Conduct a deeper analysis of specific equipment contributions to peak demand
    - Evaluate potential savings from implementing load shifting strategies
    - Develop predictive models for peak demand events based on temperature forecasts
    - Analyze the economic impact of peak demand charges vs energy consumption costs
    """

    return report


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Energy Pattern Analysis")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--report", action="store_true",
                        help="Generate a summary report")
    args = parser.parse_args()

    # Run analysis
    results = run_enhanced_analysis(args.config)

    if results and args.report:
        report = generate_pattern_analysis_report(results)

        # Save report to file
        report_path = f"{results['figure_paths']['hourly_patterns'].rsplit('/', 1)[0]}/summary_report.md"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Summary report saved to {report_path}")
