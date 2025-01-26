# scPEDSSC
scPEDSSC: Proximity Enhanced Deep Subspace Clustering method for scRNA-seq Data 
## Requirement
The python environment and the main packages needed to run scPEDSSC are as follows:

* python 3.8.16

* pytorch 2.0.0

* pandas 1.5.3

* scanpy 1.9.3

* scipy 1.10.1

* scikit-learn 1.2.2

The above python packages can be installed via Anaconda or pip commands.For example:

```
pip install scanpy 1.9.3
```

## Program execution process 
1.You need to download the datasets and codes locally.

2.Open the preprocess.py file in the PyCharm client.Take the “X.csv” file as the input to the preprocess.py program and run it to get the preprocessed data file “X_select1.csv”. 

3.Open the main.py file in the PyCharm client.Take the preprocessed data file “X_select1.csv” and the label file “X_label.csv” as the input to the main.py program, and run it to get the results nmi and ari,where X is the specific name of the dataset.

Here,take the Darmanis dataset as an example. 

Step1:Open the preprocess.py file in the PyCharm client and change the portion of the file that codes for reading data to the name of the dataset to be preprocessed. The code in the file that reads the data is as follows:

```python
data = pd.read_csv('Darmanis.csv', header=None)
data = np.array(data)
```
Here, we take the Darmanis dataset as an example, so the data read is "Darmanis.csv".After modifying the required data files, run the extent in the PyCharm client to get the preprocessed file "Darmanis_select1.csv".

Step2:Use the preprocessed dataset "Darmanis_select1.csv" and labels "Darmanis_label.csv" as input to the main.py file, and just modify the corresponding read file section in the PyCharm client. The code for the read section is as follows:

```python
x_hat = pd.read_csv('Darmanis_select1.csv', header=None)
x_hat = np.array(x_hat)
y = pd.read_csv('Darmanis_label.csv', header=None)
y = np.array(y)
y = y.ravel()
```
Next, run the program in the PyCharm client to get the final clustering results.The final result output is as follows:

```
nmi 0.861 , ari 0.886
```


## Comparison methods
The source code download links for the comparison methods in the scPEDSSC methodology are listed here, and the detailed steps and code for using each method can be found in the detailed descriptions on the website. The NMF and SIMLR methods are given together in scBAKP.

scCCL:https://github.com/LuckyxiaoLin/ScCCL.

scBKAP:https://github.com/YuBinLab-QUST/scBKAP.

scMCKC:https://github.com/leaf233/scMCKC.

scDCC:https://github.com/ttgump/scDCC.

scDSSC:https://github.com/WHY-17/scDSSC.

SSRE:https://github.com/CSUBioGroup/SSRE.
