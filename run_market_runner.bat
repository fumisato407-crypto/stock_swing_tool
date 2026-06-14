@echo off
cd /d "C:\Users\fumi\Documents\New project\stock_swing_tool"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python market_runner.py --interval-minutes 5 --target buy --min-score 70 --max-candidates 3 --use-openai off
