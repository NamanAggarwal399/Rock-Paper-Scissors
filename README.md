# OpenCV Rock, Paper, Scissors

## Overview

- The user can play a game of rock, paper, scissors against the computer.
- Uses video input from the webcam to capture the user's move (rock, paper or scissors).
- The computer also generates a random move for a fair game :).

The winner is calculated and the scores are incremented appropriately.
Ties contribute zero points while invalid hand signs are reported as invalid by the program.

## Program Execution

- Python version 3.8 is necessary for the execution. ( Conda is recommended )
```
conda create --name myenv python=3.8
conda activate myenv
pip install -r requirements.txt
python3 RockPaperScissors.py
```
