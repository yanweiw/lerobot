#!/bin/bash

    # parser.add_argument('-c', "--checkpoint", type=str, help="Path to the checkpoint")
    # parser.add_argument('-p', '--policy', default=None, type=str, help="Policy name")
    # parser.add_argument('-u', '--unconditional', action='store_true', help="Unconditional Maze")
    # parser.add_argument('-op', '--output-perturb', action='store_true', help="Output perturbation")
    # parser.add_argument('-ph', '--post-hoc', action='store_true', help="Post-hoc alignment")
    # parser.add_argument('-bi', '--biased-initialization', action='store_true', help="Biased initialization")
    # parser.add_argument('-gd', '--guided-diffusion', action='store_true', help="Guided diffusion")
    # parser.add_argument('-rd', '--recurrent-diffusion', action='store_true', help="Recurrent diffusion")
    # parser.add_argument('-v', '--vis_dp_dynamics', action='store_true', help="Visualize dynamics in DP")
    # parser.add_argument('-s', '--savepath', type=str, default=None, help="Filename to save the drawing")
    # parser.add_argument('-l', '--loadpath', type=str, default=None, help="Filename to load the drawing")

EXP_FILE="exp00.json"

echo "Running experiments for $EXP_FILE with ACT and post-hoc alignment"
echo ""
START_TIME=$(date +%s)
python gui_maze2d.py -p act -ph -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with ACT and output perturbation"
echo ""
START_TIME=$(date +%s)
python gui_maze2d.py -p act -op -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and post-hoc alignment"
echo ""
START_TIME=$(date +%s)
python gui_maze2d.py -p dp -ph -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and output perturbation"
echo ""
START_TIME=$(date +%s)
python gui_maze2d.py -p dp -op -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and biased initialization"
echo ""
START_TIME=$(date +%s)
python gui_maze2d.py -p dp -bi -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and guided diffusion"
echo ""
START_TIME=$(date +%s)
python gui_maze2d.py -p dp -gd -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "Running experiments for $EXP_FILE with DP and recurrent diffusion"
echo ""
START_TIME=$(date +%s)
python gui_maze2d.py -p dp -rd -l "$EXP_FILE" -s .json
END_TIME=$(date +%s)
echo ""
echo "Experiment completed in $(($END_TIME - $START_TIME)) seconds."
echo ""

echo "All experiments completed."