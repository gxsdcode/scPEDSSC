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


## Example 
Take the Darmanis dataset as an example. 

Step1:Darmanis.csv and Darmanis_label.csv as input, Run the preprocess.py file to get with the processed dataset Darmanis_select1.csv.

input:
```python
data = pd.read_csv('Darmanis.csv', header=None)
data = np.array(data)
```
output:
```python
with open('Darmanis_select1.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in data_select:
        writer.writerow(row)
```
Step2:The processed data Darmanis_select1.csv along with Darmanis_label.csv as input to main.py and run it to get the clustering results for the corresponding dataset.
input:
```python
x_hat = pd.read_csv('Darmanis_select1.csv', header=None)
x_hat = np.array(x_hat)
y = pd.read_csv('Darmanis_label.csv', header=None)
y = np.array(y)
y = y.ravel()
```
output:
```python
eva(y, pred_label_hat)
```
## Comparison methods
The source code download links for the comparison methods in the scPEDSSC methodology are listed here, and the detailed steps and code for using each method can be found in the detailed descriptions on the website. The NMF and SIMLR methods are given together in scBAKP

scCCL:https://github.com/LuckyxiaoLin/ScCCL.

scBKAP:https://github.com/YuBinLab-QUST/scBKAP.

scMCKC:https://github.com/leaf233/scMCKC.

scDCC:https://github.com/ttgump/scDCC.

scDSSC:https://github.com/WHY-17/scDSSC.

SSRE:https://github.com/CSUBioGroup/SSRE.
