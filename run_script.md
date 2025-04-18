.env
# How to run
To run the scripts, use the following commands in your (zsh) terminal:
~zsh - source .zshrc


- Forecast, heatmap: python -m streamlit run dashboard/dashboard.py
- Enhanced pattern analysis (hourly, weekday vs weekend, peak period analysis, temperature impact): python -m streamlit run dashboard/enhanced_dashboard.py
- Main: python main.py --config config/config.json


# Evaluation
- Evaluation: python -m streamlit run evaluation/evaluation.py
- Choose the proper date range: 2014/04/18 – 2015/04/23
- Uncheck include temperature
- Click "Run Evaluation"