# Binary classification with sklearn and keras models

A library for binary classification based on sklearn and keras.


## Requirements
- Keras
- Sklearn 

## Usage
1. ` models ` are implemented in `models.py`.
2. Models selection and evaluation functions are implemented in `model_selection.py`.
3. To clean data Run preprocessing_ckd.py on a terminal by specifying a path :
`python preprocessing_ckd.py --path "ckd.csv" --name "final_ckd" --target_name "Class"`
4. For training with default hyperparameters or for finetuning use classify.py. For exemple :
- training : `python classify.py --path  './data/processed.csv' --train True --model_name 'Decision_tree'` 
- finetunig : `python classify.py --path  './data/processed.csv' --finetune True --model_name 'Decision_tree'`
5. Models names used in classify.py are : 
```
 - Decision tree : 'Decision_tree'
 - Multilayer perceptron : 'MLP'
 - Random forest : 'Random_forest'
 - svm : 'svm'

```



