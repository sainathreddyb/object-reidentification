# Amur Tiger Reidentification

### **Dataset Download**

Dataset was downloaded from [cvwc2019challenge.](https://cvwc2019.github.io/challenge.html)

### **Dataset Preparation to run:**

Run below command to preprocess the data. The following preprocessing function, flips the images and keypoints and saves the results to image folder, csv and json files. And then extracts the parts for 
all the images and stores in the designated parts folder. Finally it creates 4-folds of the dataset.

> python preprocessing.py

All the paths to save and retrieve the files can be changed in **config.ini** file

### **Run the training script on gaivi**

We submitted our training jobs on gaivi server. Bash script is available in **trainer.sh**.
We used 4-fold cross validation for training. To run for a respective fold, change the --fold argument in trainer.sh as shown below.

> srun python -u ppbm_fold.py  --fold=0 > fold0.out

> srun python -u ppbm_fold.py  --fold=1 > fold1.out

> srun python -u ppbm_fold.py  --fold=2 > fold2.out

> srun python -u ppbm_fold.py  --fold=3 > fold3.out


Submit the job by running below command:

> #sbatch trainer.sh

### **Download weights**

Weights can be downloaded from the following link [weights.](https://drive.google.com/drive/folders/1aq__6ja3hJiqGl5eH5bQL-_9QE0w9Dxe?usp=sharing)

### **Evaluate the model**

Run the below command to evaluate the model
>  python testing.py

> python evaluation_script.py
