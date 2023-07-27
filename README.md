# CPLL
This is the code for our paper: [A Confidence-based Partial Label Learning Model for Crowd-Annotated Named Entity Recognition](https://arxiv.org/pdf/2305.12485) (ACL 2023)

## Dependencies
```
pip install transformers
```

## Reproduction
### Preprocess Data
First put the data (train.txt, dev.txt, test.txt, labels.txt) in a folder. The data format refers to the CoNLL 2003 dataset.

Then to simulate a crowd-annotated situation, you can run our perturb script.
```
python preprocess.py --raw_data_dir raw_data_dir --output_dir output_dir
```

### Prepare a pre-trained model 
We used Chinese-roberta-wwm-ext downloaded from [here](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main)

### Training
Here is an example
```
python run.py --bert_dir pre-trained_model_dir --data_dir data_dir --ent2id_dir data_dir
```

# Citation
If you use our results or scripts in your research, please cite our paper.
```
@article{xiong2023confidence,
  title={A Confidence-based Partial Label Learning Model for Crowd-Annotated Named Entity Recognition},
  author={Xiong, Limao and Zhou, Jie and Zhu, Qunxi and Wang, Xiao and Wu, Yuanbin and Zhang, Qi and Gui, Tao and Huang, Xuanjing and Ma, Jin and Shan, Ying},
  journal={arXiv preprint arXiv:2305.12485},
  year={2023}
}
```