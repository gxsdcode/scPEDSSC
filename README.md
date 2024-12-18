# scPEDSSC
scPEDSSC: Proximity Enhanced Deep Subspace Clustering method for scRNA-seq Data 
## Requirement
The python environment and the main packages needed to run scPEDSSC are as follows:

python 3.8.16

pytorch 2.0.0

pandas 1.5.3

scanpy 1.9.3

scipy 1.10.1

scikit-learn 1.2.2


## Example 
Take the Darmanis dataset as an example. 

Step1:Darmanis.csv and Darmanis_label.csv as input, Run the preprocess.py file to get with the processed dataset Darmanis_select1.csv.

Step2:The processed data Darmanis_select1.csv along with Darmanis_label.csv as input to main.py and run it to get the clustering results for the corresponding dataset.
