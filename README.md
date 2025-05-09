# 🧙‍♂️ Harry Potter A* Search Project

This project implements the A* search algorithm to strategically guide wizards through a complex, deterministic search environment inspired by Harry Potter. The goal is to destroy Horcruxes guarded by Death Eaters and ultimately defeat Voldemort.

## 📌 Problem Description

Wizards navigate through a grid-based world with passable and impassable regions. They must:
- Avoid or strategically encounter Death Eaters.
- Destroy all Horcruxes.
- Defeat Voldemort only after all Horcruxes have been destroyed.

## 🚩 Features
- Custom state representation for efficient search.
- Comprehensive heuristics to optimize the search process.
- Includes predefined test problems for both competitive and non-competitive evaluation.

## 🚀 Files Description
- **ex1.py**: Main logic implementing state transitions, actions, goal testing, and heuristics.
- **search.py**: Implementation of the generic A* search and other standard search algorithms.
- **utils.py**: General utility functions used by the search algorithms.
- **check.py**: Test harness for evaluating problem-solving capabilities.
- **problems.py**: Defined problem instances used for testing.
- **HW1.pdf**: Detailed description of the homework task and the problem environment.

## 🛠️ Installation & Running
```bash
git clone https://github.com/<yourusername>/harry-potter-a-star-search.git
cd harry-potter-a-star-search
python check.py

## 👨‍💻 Author

Gal Ofir  
[LinkedIn Profile](https://www.linkedin.com/in/gal-ofir-341283210/)
