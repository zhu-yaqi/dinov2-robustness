@echo off
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
python check_env.py
pause
