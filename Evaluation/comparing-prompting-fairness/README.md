This is a codebase that is adapted from the "prompting-fairness" framework - https://github.com/ChiscaAndrei/prompting-fairness

I have used their original implementation to construct a prefix embedding using their proposed technique.
I have then added custom evaluation functions to evaluate the embedding obtained from the constructed prompts using Debias TRaIN.
This allowed me to compare them directly on the exact same instances.
The results is in 'runs/' folder.
The modified code is inside 'src/prompt_tuning_debias' folder. 

Due to github size restrictions, the large data file for stereoset couldn't be uploaded, which needs to be downloaded from the below link and placed inside 'stereoset_evaluator/data'.
( https://drive.google.com/drive/folders/1wFyGKbmmK8bBrQEb49CmCNtjX2QfIvrN?usp=sharing )

To run this, download this entire folder on your local device. Dependencies are listed in 'requirements.txt'. 
Then, run the command mentioned in 'runs/#experiment_name/experiment_commands.sh'.
