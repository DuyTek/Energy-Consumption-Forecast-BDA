DEMO Steps:
1. Run `dashboard.py` to see forecasting
2. Run `enhanced_dashboard.py` for advanced analysis
3. Run `evaluation.py` for getting evaluation

# How to run
To run the scripts, use the following commands in your (zsh) terminal:
- macOS: source .zshrc
- Window: . .\env-setup.ps1



- Forecast, heatmap: python -m streamlit run dashboard/dashboard.py
- Enhanced pattern analysis (hourly, weekday vs weekend, peak period analysis, temperature impact): python -m streamlit run dashboard/enhanced_dashboard.py
- Simulate and train model with data: python main.py --config config/config.json


# Evaluation
- Evaluation: python -m streamlit run dashboard/evaluation.py
- Choose the proper date range: 2014/04/18 – 2015/04/23
- Uncheck include temperature
- Click "Run Evaluation"