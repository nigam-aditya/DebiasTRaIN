This folder contains the code for evaluation of prompts constructed in the training folder.

Evaluation_Implementation.ipynb is the main notebook which makes use of the stereoset dataset (https://github.com/moinnadeem/StereoSet) to evaluate the performance of the prompts. It uses the files in 'modified_files' folder which are two .py files from the stereoset github repository adapted to include our prefix in the evaluation. 

Graphs.ipynb creates graphs from the results obtained in the main eval notebook.
 
comparing-prompting-fairness is the folder that contains code for comparative evaluation with the continuous prompt technique. More details in the readme file inside it.
