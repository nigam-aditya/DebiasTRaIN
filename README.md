This is the Code Base for my MSc dissertation - "Debias TRaIN: Transferable and Interpretable Prompts for Bias Mitigation in LLMs via Discrete Prompt Optimization" , submitted to the University of Manchester.

Structure:
DebiasTRaIN/
├── Training/
│   ├── DebiasTRaIN_prompt_optimization.ipynb   # main notebook for construction of prompts using our method
│   ├── bias_template.py   # constructs templates required for training
│   ├── occupations_gender_specific.json        # data file for bias_template.py
│   ├── occupations_large.json                  # data file for bias_template.py
│   └── Readme.txt                              # Further Details on how to use this folder
│ 
├── Evaluation/
│   ├── Evaluation_Implementation.ipynb         # main notebook for evaluation of prompts u
│   ├── modified_files/                         # modified files required for evaluation
│   ├── Graphs.ipynb                            # notebook constructing graphs based on results from evaluation notebook
│   ├── comparing-prompting-fairness/           # code base for implementing comparison with the PF framework
│   └── Readme.txt                              # Further Details on how to use this folder
