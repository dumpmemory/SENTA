# SENTA

This repository is the official implementation of the paper [Interventional Aspect-Based Sentiment Analysis](https://arxiv.org/pdf/2104.11681.pdf). 

## Requirements
python3 / pytorch 1.7 / transformers 4.5.1

## Dataset

Dataset is included in file. Pre-trained models such as [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main) can be download from [Hugging Face models](https://huggingface.co/models)

The expected structure of files is:
```
ATLOP
 |-- dataset
 |    |-- asc
 |    |    |-- laptop 
 |    |    |    |-- train.json
 |    |    |    |-- dev.json
 |    |    |    |-- test.json
 |    |    |    |-- test_enriched.json
 |    |    |    |-- build_data.py
 |    |    |-- rest 
 |    |    |    |-- train.json
 |    |    |    |-- dev.json
 |    |    |    |-- test.json
 |    |    |    |-- test_enriched.json
 |    |    |    |-- build_data.py
 |-- run_scm.py
 |-- model.py
 |-- utils.py
 |-- README.md
```

## Training
You can set the `--num_train_epochs` and `--model_name_or_path` argument in training scripts.
### STEP1: Train a Confounding Model

Train the confounding model on Laptop domain with the following command:
 
```bash
# example for bert-base-uncased
>>  python run_scm.py 
\--model_type bert  --data_dir dataset/asc 
\--per_gpu_train_batch_size 128 
\--per_gpu_eval_batch_size 128 
\--learning_rate 1e-5 
\--num_train_epochs ${num_train_epochs}
\--max_seq_length 64
\--doc_stride 128 
\--threads 10 
\--do_lower_case 
\--evaluate_during_training 
\--logging_steps 500 
\--save_steps 500 
\--dataset eaptop 
\--output_dir ./output_laptop_base 
\--model_name_or_path ${pre_trained_models_path}/bert-base-uncased 
\--do_train 
\--do_eval

```

### STEP2: Train a Interventional Model
Train the interventional model on Laptop domain with the following command:
```bash
# example for bert-base-uncased
>> python run_scm.py 
\--model_type bert  --data_dir dataset/asc 
\--per_gpu_train_batch_size 128 
\--per_gpu_eval_batch_size 128 
\--learning_rate 1e-5 
\--num_train_epochs ${num_train_epochs}
\--max_seq_length 64
\--doc_stride 128 
\--threads 10 
\--do_lower_case 
\--evaluate_during_training 
\--logging_steps 500 
\--save_steps 500 
\--dataset laptop 
\--output_dir ./output_laptop_base 
\--model_name_or_path ${pre_trained_models_path}/bert-base-uncased 
\--do_train 
\--do_eval 
\--do_intervention  
```

## Evaluation

To evaluate the trained interventional model in Ori test set, 
```bash
# example for bert-base-uncased
>>  python run_scm.py 
\--model_type bert  
\--data_dir dataset/asc 
\--do_lower_case 
\--dataset laptop 
\--output_dir ./output_laptop_base 
\--model_name_or_path ${pre_trained_models_path}/bert-base-uncased 
\--do_test  
\--do_intervention
```

To evaluate the trained interventional model in Change test set(ARTS), 
```bash
# example for bert-base-uncased
>>  python run_scm.py 
\--model_type bert  --data_dir dataset/asc 
\--do_lower_case 
\--evaluate_during_training 
\--dataset laptop 
\--output_dir ./output_laptop_base 
\--model_name_or_path ${pre_trained_models_path}/bert-base-uncased 
\--do_test 
\--do_test_enriched  
\--do_intervention
```

## How to Cite

```bibtex                
@article{bi2021interventional,
  title={Interventional Aspect-Based Sentiment Analysis},
  author={Bi, Zhen and Zhang, Ningyu and Ye, Ganqiang and Yu, Haiyang and Chen, Xi and Chen, Huajun},
  journal={arXiv preprint arXiv:2104.11681},
  year={2021}
}
```