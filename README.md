# <img src="img/talkdown.png" height=75 alt="TalkDown"> TalkDown: A Corpus for Condescension Detection in Context


## Introduction

This is the code release for the paper [TalkDown: A Corpus for Condescension Detection in Context](https://www.aclweb.org/anthology/D19-1385) by Zijian Wang and Christopher Potts in proceedings of EMNLP-IJCNLP 2019.

## Dependencies
### Python dependencies
Run `pip install -r requirements.txt`. This codebase requires Python version >= 3.6.
### Data
Run `bash download_data.bash` to download and uncompress the TalkDown dataset. Or you could use this [link](https://nlp.stanford.edu/~zijwang/talkdown/talkdown.tar.gz). 
### Pretrained model (optional)
Run `bash download_model.bash` to download our best pretrained model to reproduce the result. It is not required if you want to train your model from scratch.


## Sample commands for training and evaluation

### Train
You could train a BERT model using the following command.
```
python -m src.bert --do_train --use_quoted --use_context --output_dir test
```
### Evaluate
You could evaluate your model using the following command. This command also reproduces our best result in the paper (make sure you have downloaded the pretrained model).
```
python -m src.bert --do_eval --use_quoted --use_context --eval_on_test --output_dir pretrained_full
```
which should return `Model's F1 is 0.6835111677776263`
## Citation

    @inproceedings{wang2019talkdown,
      author = {Wang, Zijian  and  Potts, Christopher}
      title = {{TalkDown}: A Corpus for Condescension Detection in Context},
      booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
      url = {https://www.aclweb.org/anthology/D19-1385},
      year = {2019}
    }

## Contact

You may reach out us at zijwang@stanford.edu and cgpotts@stanford.edu.
