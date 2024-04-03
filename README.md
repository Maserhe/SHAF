	
## SHAF for CIR


## Setting up

First, clone the repository to a desired location.

<details>
  <summary><b>Conda Environment</b></summary>
&emsp; 

&emsp; 
	
The following commands will create a local Anaconda environment with the necessary packages installed.

```bash
conda create -n shaf -y python=3.8
conda activate shaf
pip install -r requirements.txt
```
&emsp; 
</details>

<details>
  <summary><b>Datasets</b></summary>
&emsp; 
	
Experiments are conducted on two standard datasets -- [Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq) and [SHOES](http://tamaraberg.com/attributesDataset/index.html), please see their repositories for download instructions. 


&emsp; 
</details>

<details>
  <summary><b>Training</b></summary>
&emsp; 

model for training

```bash
# Optional: comet experiment logging --api-key and --workspace
python src/combiner_train.py --dataset
dataset_name
--projection-dim
2048
--hidden-dim
4096
--num-epochs
200
--clip-model-name
RN50x4
--combiner-lr
2e-5
--batch-size
512
--clip-bs
32
--transform
targetpad
--target-ratio
1.25
--validation-frequency
1
```
</details>