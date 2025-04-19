DEMO Steps:
1. Run `dashboard.py` to see forecasting
2. Run `enhanced_dashboard.py` for advanced analysis
3. Run `evaluation.py` for getting evaluation

# How to run
To run the scripts, use the following commands in your (zsh) terminal:
~zsh - source .zshrc


- Forecast, heatmap: python -m streamlit run dashboard/dashboard.py
- Enhanced pattern analysis (hourly, weekday vs weekend, peak period analysis, temperature impact): python -m streamlit run dashboard/enhanced_dashboard.py
- Simulate data: python main.py --config config/config.json
- Train model: python main.py


# Evaluation
- Evaluation: python -m streamlit run dashboard/evaluation.py
- Choose the proper date range: 2014/04/18 – 2015/04/23
- Uncheck include temperature
- Click "Run Evaluation"