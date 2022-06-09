# nwmfnn
Code of Neuro-weighted Multi-functional Nearest-neighbour Classification



There are two main steps to complete the neuro-weighted multi-functional nearest-neighbour classification: 

1) Generate weights by one neural network.   

2) Features weighting and Classification.  




The folder "00-Datasets used" contains two types of data, which are used for Matlab (.dt) and Weka (.arff), respectively.



# 1 Generate weights

This part is implemented by Matlab, included in folder "01-Weights generated by diverse neural networks".

Run the "cross_validation.m" in each subfolder to generate "**weights**.txt". Note that the datasets.dt need to be copy from the corresponding position in "Dt files for Matlab".

# 2 Feature weighting and Classification 
Features weighting and classification are implemented in Weka, compressed in file "02-Features weighted Weka.zip".

This version of Weka requires the JDK11 environment to be configured.

The specific steps are as follows:
- Open the Explorer -> Preprocess -> Open file -> Choose one .arff from folder "ARFF files for Weka". 
- Switch to panel "Classify" -> Choose -> fuzzy -> FuzzyRoughNN/VQNN -> Open the Properties panel of FuzzyRoughNN -> similairty -> Choose -> WSimilarity2 -> Open the Properties panel of WSimilarity2 -> Copy weights from "**weights**.txt" in Step 1 -> Paste the weights to the last row -> Click OK
- Select Cross-validation -> Start


















