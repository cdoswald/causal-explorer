#!/bin/bash

# Set optional flags
RUN_TESTS=true

# Install package
pip install .

# Run tests to verify code works as intended 
# (takes ~40 seconds; main code will automatically run if tests pass)
if [ $RUN_TESTS ]; then
    pytest "CausalEx/tests"
fi

# Run experiments
python3 CausalEx/A_main.py
python3 CausalEx/B_main.py
