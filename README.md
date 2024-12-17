# Running Python Code 


## 1. Unzip the file
The folder consists of following structure
```
├── common_imports.py
├── Data
│   ├── cmnist
│   ├── ColorizeMNIST
│   └── MNIST
├── data_loader.py
├── main.py
├── model_bed.py
├── Notebooks
│   ├── colorize.ipynb
│   ├── model_performance_poster.png
│   ├── Predict.ipynb
│   ├── Screenshot 2024-12-15 at 21-15-14 Visualize.png
│   ├── Screenshot 2024-12-15 at 21-26-33 Visualize.png
│   └── Visualize.ipynb
├── output
│   ├── logs
│   └── models
├── Poster.pdf
├── predict.py
├── random_selector.py
├── README.md
├── requirements.txt
├── resnet.py
├── splitter.py
├── test.py
├── trainer.py
└── trainer_v2.py
```    
Unzip the zip file
```
unzip term_project-gopi.zip
```

## 2. Install all the dependencies

Ensure you are within `DistributionShift` folder.

Note: The training and inference should run on standard ml libraries. So running requirements.txt is completely optional!

```
pip install -r requirements.txt
```

This may take a minute or two.

## 3. Implementation

### 3.1 Change Directory to DistributionShift
```
cd DistributionShift
```

### 3.2 Generate Dataset (optional)

Sample generated sets are provided in the Data Folder. To create more variants of the data or a different seed, the notebook colorize.ipynb can be used. After the generation, move the dataset to Data folder using respective split names i.e train_fixed / train_random inside the cmnist folder. There are two different function modules provided to generate fixed vs random color sets on the mnist dataset for creating appropriate flavor of the dataset.

### 3.3 Train Experiments 

To train s.o.t.a model bed, use the training module as is, 
```
CUDA_VISIBLE_DEVICES=2 python3 main.py
```

To train a simple cnn architecture on the same datasets, use the `get_simple_cnn` module instead of `get_models` module and run the same command

```
CUDA_VISIBLE_DEVICES=2 python3 main.py
```

### 3.4 Train Results & Saved Models 

After successful completion of training on different model architecture, the results will be stored in the output folder, `output/logs` containing the training logs and `output/models` should contain the trained model variants. 

Note: s.o.t.a architecture input sizes `224 * 224`, simple_cnn `28*28`. Pytorch Transforms module need to be changed accordingly during train & inference times. 

### 3.5 Train Naming Convention

All the trained models follow the following naming convention,

```
train model falvor = model name + train data flavor

model names : [vgg16, efficientnet_b0, simple_cnn, resnet18]
train data flavor : ['fixed color', 'random color']
```

### 3.6 Infer Results & Naming Convention

The following script takes all the trained model variants in the `output/models/` folder and stores the results in `output/logs/` folder using the naming convention explained in subsequent section.

```
CUDA_VISIBLE_DEVICES=2 python3 predict.py
```

Naming Convention Infer Results,

```
test data results = train model flavor + test data flavor

train model flavor : [best_resnet18_fixed_color.pth, best_simple_cnn_rand_color.pth, ...]

test data flavor : ['fixed color', 'random color']

```

### 3.7 Inference Results Visualizer / Softmax Score Visualizer

Use the `visualize.ipynb` to look at the generated results for appropriante sections accordingly.
