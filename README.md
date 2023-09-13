# Identifier-Renaming
Generating higher quality identifier names by using context and following conventions with RLHF <br>
0.0 is used to generate the csv dataset from the .java files after 4.0 <br>
1.0 is code for the classifier to predict the number of mask tokens to insert in for the variable name <br>
2.2 is code for finetuning the GraphCodeBert model on variable names <br>
3.1 is code for the RHLF step of the pipeline, where saved weights of model are improved with rewards and human input, and the new weights are saved for use. The number of mask tokens for identifier names is procured from the classifier <br>
4.0 is used to create the dataset. Repositories are cloned and this code file iterates over all the files and preprocesses them for creation of dataset.
