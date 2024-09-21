#!/bin/bash

EXP_FILE="pusht_exp00.json"

echo "Running experiments for $EXP_FILE with ACT and post-hoc alignment"
echo ""
START_TIME=$(date +%s)
python gui_pusht.py -p act -ph -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with ACT and output perturbation"
echo ""
START_TIME=$(date +%s)
python gui_pusht.py -p act -op -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and post-hoc alignment"
echo ""
START_TIME=$(date +%s)
python gui_pusht.py -p dp -ph -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and output perturbation"
echo ""
START_TIME=$(date +%s)
python gui_pusht.py -p dp -op -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and biased initialization"
echo ""
START_TIME=$(date +%s)
python gui_pusht.py -p dp -bi -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and guided diffusion"
echo ""
START_TIME=$(date +%s)
python gui_pusht.py -p dp -gd -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and recurrent diffusion"
echo ""
START_TIME=$(date +%s)
python gui_pusht.py -p dp -rd -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "All experiments completed."