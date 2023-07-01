## Model
1. 到[这里](https://huggingface.co/facebook/m2m100_418M/tree/main)下载
- config.json
- generation_config.json
- pytorch_model.bin
- sentencepiece.bpe.model
- tokenizer_config.json
- vocab.json
2. 文件放到`./model/`下


## Environment
```
conda create -n translator python=3.8
conda activate translator
pip install -r requirements.txt
```


### Run
```
main.ipynb
```