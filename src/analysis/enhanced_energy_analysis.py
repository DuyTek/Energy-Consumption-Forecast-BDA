# analysis/enhanced_energy_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedEnergyAnalysis")


class EnhancedEnergyAnalysis:
    """Enhanced analysis methods for energy consumption data."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with energy consumption DataFrame.

        Args:
            data: DataFrame with energy consumption data
                 (must have 'timestamp', 'demand', 'temperature' columns)
        """
        self.data = data.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        elif 'ds' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['ds'])
            if 'y' in self.data.columns and 'demand' not in self.data.columns:
                self.data['demand'] = self.data['y']

        # Extract temporal features if not present
        if 'hour' not in self.data.columns:
            self.data['hour'] = self.data['timestamp'].dt.hour
        if 'day_of_week' not in self.data.columns:
            self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        if 'month' not in self.data.columns:
            self.data['month'] = self.data['timestamp'].dt.month
        if 'year' not in self.data.columns:
            self.data['year'] = self.data['timestamp'].dt.year

        # Add is_weekend flag (0-4 are weekdays, 5-6 are weekend)
        self.data['is_weekend'] = self.data['day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0)

    def analyze_hourly_patterns(self) -> Dict:
        """Analyze hourly consumption patterns with detailed metrics.

        Returns:
            Dict containing hourly pattern analysis results
        """
        logger.info("Analyzing hourly consumption patterns")

        # Group by hour and calculate statistics
        hourly_stats = self.data.groupby('hour')['demand'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).reset_index()

        # Calculate peak-to-average ratio
        hourly_stats['peak_to_avg_ratio'] = hourly_stats['max'] / \
            hourly_stats['mean']

        # Identify peak hours (top 3)
        peak_hours = hourly_stats.nlargest(3, 'mean')['hour'].tolist()

        # Calculate overall average
        overall_avg = self.data['demand'].mean()

        # Calculate how much higher peak hours are compared to overall average
        peak_hour_avg = self.data[self.data['hour'].isin(
            peak_hours)]['demand'].mean()
        peak_hour_pct_above_avg = ((peak_hour_avg / overall_avg) - 1) * 100

        return {
            'hourly_stats': hourly_stats.to_dict(orient='records'),
            'peak_hours': peak_hours,
            'overall_avg': overall_avg,
            'peak_hour_avg': peak_hour_avg,
            'peak_hour_pct_above_avg': peak_hour_pct_above_avg
        }

    def compare_weekday_weekend(self) -> Dict:
        """Compare weekday vs weekend consumption patterns.

        Returns:
            Dict containing weekday/weekend comparison results
        """
        logger.info("Comparing weekday vs weekend patterns")

        # Group by is_weekend and hour
        weekday_hourly = self.data[self.data['is_weekend'] == 0].groupby('hour')[
            'demand'].mean()
        weekend_hourly = self.data[self.data['is_weekend'] == 1].groupby('hour')[
            'demand'].mean()

        # Calculate peak hours for each
        weekday_peak_hours = weekday_hourly.nlargest(3).index.tolist()
        weekend_peak_hours = weekend_hourly.nlargest(3).index.tolist()

        # Calculate average consumption
        weekday_avg = self.data[self.data['is_weekend'] == 0]['demand'].mean()
        weekend_avg = self.data[self.data['is_weekend'] == 1]['demand'].mean()

        # Calculate percentage difference
        pct_diff = ((weekend_avg / weekday_avg) - 1) * 100

        return {
            'weekday_hourly': weekday_hourly.to_dict(),
            'weekend_hourly': weekend_hourly.to_dict(),
            'weekday_peak_hours': weekday_peak_hours,
            'weekend_peak_hours': weekend_peak_hours,
            'weekday_avg': weekday_avg,
            'weekend_avg': weekend_avg,
            'weekend_weekday_pct_diff': pct_diff
        }

    def quantify_peak_periods(self, threshold_percentile: float = 90) -> Dict:
        """Identify and quantify peak periods in detail.

        Args:
            threshold_percentile: Percentile threshold for peak demand classification

        Returns:
            Dict containing detailed peak period analysis
        """
        logger.info(
            f"Quantifying peak periods using {threshold_percentile}th percentile threshold")

        # Calculate threshold
        threshold = np.percentile(self.data['demand'], threshold_percentile)

        # Identify peak periods
        peak_periods = self.data[self.data['demand'] >= threshold].copy()

        # Count peak periods by different dimensions
        peak_by_hour = peak_periods.groupby('hour').size()
        peak_by_day = peak_periods.groupby('day_of_week').size()
        peak_by_month = peak_periods.groupby('month').size()

        # Calculate average duration of peak events (this requires time-sorted consecutive data)
        # For this analysis, we'll use a simplified approach assuming data is in time order
        self.data['is_peak'] = self.data['demand'] >= threshold
        self.data['peak_group'] = (
            self.data['is_peak'] != self.data['is_peak'].shift()).cumsum()
        peak_durations = self.data[self.data['is_peak']].groupby(
            'peak_group').size()

        # Calculate average demand during peak vs non-peak
        peak_avg_demand = peak_periods['demand'].mean()
        non_peak_avg_demand = self.data[self.data['demand']
                                        < threshold]['demand'].mean()
        peak_multiplier = peak_avg_demand / non_peak_avg_demand

        return {
            'threshold': threshold,
            'peak_count': len(peak_periods),
            'peak_percentage': (len(peak_periods) / len(self.data)) * 100,
            'peak_by_hour': peak_by_hour.to_dict(),
            'peak_by_day': peak_by_day.to_dict(),
            'peak_by_month': peak_by_month.to_dict(),
            'avg_peak_duration': peak_durations.mean() if not peak_durations.empty else 0,
            'peak_avg_demand': peak_avg_demand,
            'non_peak_avg_demand': non_peak_avg_demand,
            'peak_multiplier': peak_multiplier
        }

    def analyze_temperature_impact(self) -> Dict:
        """Analyze the impact of temperature on demand in detail.

        Returns:
            Dict containing temperature impact analysis
        """
        logger.info("Analyzing temperature impact on demand")

        # Create temperature bins
        temp_bins = pd.cut(self.data['temperature'], bins=10)
        temp_bin_stats = self.data.groupby(temp_bins)['demand'].agg(
            ['mean', 'std', 'count']).reset_index()

        # Analyze temperature impact during peak hours vs off-peak
        peak_hours_mask = self.data['hour'].isin(
            self.analyze_hourly_patterns()['peak_hours'])

        # Temperature correlation during peak vs off-peak
        peak_hours_corr = self.data[peak_hours_mask]['temperature'].corr(
            self.data[peak_hours_mask]['demand'])
        off_peak_corr = self.data[~peak_hours_mask]['temperature'].corr(
            self.data[~peak_hours_mask]['demand'])

        # Weekday vs weekend temperature sensitivity
        weekday_temp_corr = self.data[self.data['is_weekend'] == 0]['temperature'].corr(
            self.data[self.data['is_weekend'] == 0]['demand'])
        weekend_temp_corr = self.data[self.data['is_weekend'] == 1]['temperature'].corr(
            self.data[self.data['is_weekend'] == 1]['demand'])

        # Calculate temperature sensitivity for each hour
        hourly_temp_sensitivity = {}
        for hour in range(24):
            hour_data = self.data[self.data['hour'] == hour]
            if len(hour_data) > 5:  # Ensure we have enough data points
                hourly_temp_sensitivity[hour] = hour_data['temperature'].corr(
                    hour_data['demand'])

        return {
            'overall_correlation': self.data['temperature'].corr(self.data['demand']),
            'temp_bin_stats': temp_bin_stats.to_dict(orient='records'),
            'peak_hours_temp_corr': peak_hours_corr,
            'off_peak_temp_corr': off_peak_corr,
            'weekday_temp_corr': weekday_temp_corr,
            'weekend_temp_corr': weekend_temp_corr,
            'hourly_temp_sensitivity': hourly_temp_sensitivity
        }

    def plot_hourly_patterns(self) -> go.Figure:
        """Create plotly visualization of hourly consumption patterns.

        Returns:
            Plotly figure with hourly pattern visualization
        """
        # Get hourly statistics
        hourly_analysis = self.analyze_hourly_patterns()
        hourly_stats = pd.DataFrame(hourly_analysis['hourly_stats'])

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add hourly mean demand
        fig.add_trace(
            go.Bar(
                x=hourly_stats['hour'],
                y=hourly_stats['mean'],
                name="Mean Demand",
                marker_color='blue'
            ),
            secondary_y=False
        )

        # Add standard deviation as error bars
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['mean'],
                error_y=dict(
                    type='data',
                    array=hourly_stats['std'],
                    visible=True
                ),
                mode='markers',
                marker=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ),
            secondary_y=False
        )

        # Add peak-to-average ratio
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['peak_to_avg_ratio'],
                name="Peak-to-Avg Ratio",
                mode="lines+markers",
                marker_color='red',
                line=dict(width=2, dash='dot')
            ),
            secondary_y=True
        )

        # Add annotations for peak hours
        for hour in hourly_analysis['peak_hours']:
            fig.add_annotation(
                x=hour,
                y=hourly_stats[hourly_stats['hour'] == hour]['mean'].values[0],
                text="Peak",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )

        # Update layout
        fig.update_layout(
            title="Hourly Energy Demand Pattern",
            xaxis_title="Hour of Day",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )

        # Update y-axes titles
        fig.update_yaxes(title_text="Energy Demand", secondary_y=False)
        fig.update_yaxes(title_text="Peak-to-Average Ratio", secondary_y=True)

        return fig

    def plot_weekday_weekend_comparison(self) -> go.Figure:
        """Create plotly visualization comparing weekday and weekend patterns.

        Returns:
            Plotly figure with weekday vs weekend comparison
        """
        # Get weekday/weekend comparison data
        comparison = self.compare_weekday_weekend()

        # Create dataframes for plotting
        weekday_df = pd.DataFrame({
            'hour': list(comparison['weekday_hourly'].keys()),
            'demand': list(comparison['weekday_hourly'].values()),
            'type': 'Weekday'
        })

        weekend_df = pd.DataFrame({
            'hour': list(comparison['weekend_hourly'].keys()),
            'demand': list(comparison['weekend_hourly'].values()),
            'type': 'Weekend'
        })

        combined_df = pd.concat([weekday_df, weekend_df])

        # Create figure
        fig = px.line(
            combined_df,
            x="hour",
            y="demand",
            color="type",
            line_shape="spline",
            markers=True,
            title="Weekday vs Weekend Energy Demand"
        )

        # Add annotations for peak hours
        for hour in comparison['weekday_peak_hours']:
            fig.add_annotation(
                x=hour,
                y=weekday_df[weekday_df['hour'] == hour]['demand'].values[0],
                text="Weekday Peak",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )

        for hour in comparison['weekend_peak_hours']:
            fig.add_annotation(
                x=hour,
                y=weekend_df[weekend_df['hour'] == hour]['demand'].values[0],
                text="Weekend Peak",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40
            )

        # Update layout
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Energy Demand",
            template="plotly_white"
        )

        return fig

    def plot_peak_period_analysis(self, threshold_percentile: float = 90) -> go.Figure:
        """Create plotly visualization of peak period analysis.

        Args:
            threshold_percentile: Percentile threshold for peak demand classification

        Returns:
            Plotly figure with peak period analysis
        """
        # Get peak period analysis
        peak_analysis = self.quantify_peak_periods(threshold_percentile)

        # Create subplots: 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Peak Periods by Hour of Day",
                "Peak Periods by Day of Week",
                "Peak vs Non-Peak Demand",
                "Peak Periods by Month"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )

        # 1. Peak by hour
        hours = list(peak_analysis['peak_by_hour'].keys())
        peak_counts = list(peak_analysis['peak_by_hour'].values())

        fig.add_trace(
            go.Bar(x=hours, y=peak_counts, name="Peak Frequency",
                   marker_color="indianred"),
            row=1, col=1
        )

        # 2. Peak by day of week
        days = list(peak_analysis['peak_by_day'].keys())
        day_names = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_labels = [day_names[day] for day in days]
        day_counts = list(peak_analysis['peak_by_day'].values())

        fig.add_trace(
            go.Bar(x=day_labels, y=day_counts, name="Day Frequency",
                   marker_color="lightseagreen"),
            row=1, col=2
        )

        # 3. Peak vs non-peak demand
        fig.add_trace(
            go.Pie(
                labels=["Peak Demand", "Non-Peak Demand"],
                values=[peak_analysis['peak_avg_demand'],
                        peak_analysis['non_peak_avg_demand']],
                hole=.3,
                marker_colors=["crimson", "lightblue"]
            ),
            row=2, col=1
        )

        # 4. Peak by month
        months = list(peak_analysis['peak_by_month'].keys())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_labels = [month_names[month-1] for month in months]
        month_counts = list(peak_analysis['peak_by_month'].values())

        fig.add_trace(
            go.Bar(x=month_labels, y=month_counts,
                   name="Month Frequency", marker_color="goldenrod"),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"Peak Period Analysis (>{threshold_percentile}th Percentile)",
            showlegend=False,
            height=700,
            template="plotly_white"
        )

        # Update axes
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_yaxes(title_text="Number of Peak Periods", row=1, col=1)

        fig.update_xaxes(title_text="Day of Week", row=1, col=2)
        fig.update_yaxes(title_text="Number of Peak Periods", row=1, col=2)

        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Number of Peak Periods", row=2, col=2)

        return fig

    def plot_temperature_demand_heatmap(self) -> go.Figure:
        """Create plotly heatmap of temperature vs hour vs demand.

        Returns:
            Plotly figure with temperature-hour-demand heatmap
        """
        # Create temperature bins and hour bins
        self.data['temp_bin'] = pd.cut(
            self.data['temperature'],
            bins=10,
            labels=[f"{i:.1f}-{i+4:.1f}" for i in np.linspace(
                self.data['temperature'].min(),
                self.data['temperature'].max()-4,
                10
            )]
        )

        # Create pivot table
        pivot_data = self.data.pivot_table(
            values='demand',
            index='temp_bin',
            columns='hour',
            aggfunc='mean'
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='YlOrRd',
            colorbar=dict(title="Avg Demand")
        ))

        # Update layout
        fig.update_layout(
            title="Energy Demand by Temperature and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Temperature Range (°C)",
            template="plotly_white"
        )

        return fig
