#!/bin/bash
DIRECTORY='venv'
if [ -d "$DIRECTORY" ]; then
    echo 'venv already exists'
else
    echo 'venv does not exist'
    mkdir venv
    python3 -m venv venv/
    source venv/bin/activate
    wget https://bootstrap.pypa.io/get-pip.py 
    python get-pip.py
    rm -rf get-pip.py
    pip install -r requirements.txt
fi