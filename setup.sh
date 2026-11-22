#!/usr/bin/env bash
# Mac/Linux setup
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
echo "Done. In VS Code select interpreter ./venv/bin/python"
