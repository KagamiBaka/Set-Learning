## Set Learning for Generative Information Extraction

Created by Jiangnan Li, Yice Zhang, Bin Liang, Kam-Fai Wong, and Ruifeng Xu, Harbin Insitute of Technology, Shenzhen.

This repository contains the official PyTorch implementation of our EMNLP 2023 paper [Set Learning for Generative Information Extraction](https://aclanthology.org/2023.emnlp-main.806.pdf). We use [BART-ABSA](https://github.com/yhcc/BARTABSA) as an example to illustrate how set learning can be implemented on current generative information extraction frameworks.

### Environment Setup and Data Preprocessing
Our code is built on [BART-ABSA](https://github.com/yhcc/BARTABSA). You can set up an environment and preprocess data for our code according to their configurations.

### Training and Evaluation
Enter the peng/ folder, and then run the following command.
```text
python train.py --dataset pengb/14lap
```
The code will automatically run the evaluation during and after training.

### Implemention of Set Learning
Compared to the original code, our modification concentrated on peng/trainer.py (to add permutation sampling) and peng/model/losses.py (to add set learning loss function).

### Citation
If you find this code helpful for your research, please consider citing
```text
@inproceedings{li2023set,
  title={Set learning for generative information extraction},
  author={Li, Jiangnan and Zhang, Yice and Liang, Bin and Wong, Kam-Fai and Xu, Ruifeng},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={13043--13052},
  year={2023}
}
```