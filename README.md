# [ECCV2022] Acknowledging the Unknown for Multi-label Learning with Single Positive Labels

<<<<<<< HEAD

> **Authors**: Donghao Zhou, Pengfei Chen, Qiong Wang, Guangyong Chen, Pheng-Ann Heng
>
> **Affiliations**: SIAT-CAS, UCAS, Tencent, Zhejiang Lab, CUHK

**Abstract**

Due to the difficulty of collecting exhaustive multi-label annotations, multi-label datasets often contain partial labels. We consider an extreme of this weakly supervised learning problem, called single positive multi-label learning (SPML), where each multi-label training image has only one positive label. Traditionally, all unannotated labels are assumed as negative labels in SPML, which introduces false negative labels and causes model training to be dominated by assumed negative labels. In this work, we choose to treat all unannotated labels from an alternative perspective, i.e. acknowledging they are unknown. Hence, we propose entropy-maximization (EM) loss to attain a special gradient regime for providing proper supervision signals. Moreover, we propose asymmetric pseudo-labeling (APL), which adopts asymmetric-tolerance strategies and a self-paced procedure, to cooperate with EM loss and then provide more precise supervision. Experiments show that our method significantly improves performance and achieves state-of-the-art results on all four benchmarks.

<div align="center">
<img src="Images/overview.jpg" title="EM loss and APL" width="50%">
</div>
<div align="center">
<img src="Images/viz.jpg" title="EM loss and APL" width="50%">
</div>


=======
>>>>>>> 14fbf23f1d15c2016bee7a8bc0439978c0a5583a
## To-do List
- [ ] Polish `README.md`.
- [ ] Further clean our code.


## Quick Start

1. Refer to [this docs](https://github.com/elijahcole/single-positive-multi-label/blob/main/data/README.md) to prepare the datasets.

2. Run this to train and evaluate a model trianed with EM loss+APL:
```
python train.py
```

## Cite this paper
```latex
@misc{zhou2022acknowledging,
      title={Acknowledging the Unknown for Multi-label Learning with Single Positive Labels},
      author={Donghao Zhou and Pengfei Chen and Qiong Wang and Guangyong Chen and Pheng-Ann Heng},
      year={2022},
      eprint={2203.16219},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
Our repository is built upon the code of [single-positive-multi-label](https://github.com/elijahcole/single-positive-multi-label). We would like to thank its authors for their great work.
