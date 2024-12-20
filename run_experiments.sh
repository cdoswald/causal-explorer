#!/bin/bash

# Set optional flags
SKIP_TESTS=false

# Install package
pip install .

# Run tests to verify code works as intended 
# (takes ~40 seconds; main code will automatically run if tests pass)
all_tests_pass=false

if [ $SKIP_TESTS ]; then
    all_tests_pass=true
else
    pytest "CausalEx/tests"
    if [$? -eq 0 ]; then
        all_tests_pass=true
    fi
fi

# Run main experiments
if [ $all_tests_pass ]; then
    python3 CausalEx/A_main.py
    python3 CausalEx/B_main.py
else
    echo "Tests failed--skipping main experiments (can override in run_experiments.sh file)"
fi
