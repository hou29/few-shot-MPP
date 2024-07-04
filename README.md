# Attribute-guided prototype network (APN)
Official PyTorch-based implementation in the paper Attribute-guided prototype network for few-shot molecular property prediction.

# News
[2024/7/4] Repository installation completed.

# Abstract
The molecular property prediction (MPP) plays a crucial role in the drug discovery process, providing valuable insights for molecule evaluation and screening. Although deep learning has achieved numerous advances in this area, its success often depends on the availability of substantial labeled data. The few-shot MPP is a more challenging scenario, which aims to identify unseen property with only few available molecules. In this paper, we propose an attribute-guided prototype network (APN) to address the challenge. APN first introduces an molecular attribute extractor, which can not only extracts three different types of fingerprint attributes (single fingerprint attributes, dual fingerprint attributes, triplet fingerprint attributes) by considering 7 circular-based, 5 path-based and 2 substructure-based fingerprints, but also can use self-supervised learning to automatically extract high-level attributes from molecular graph. Further, APN designs the Attribute-Guided Dual-channel Attention module (AGDA) to learn the relationship between the molecular graphs and attributes and refine the local and global representation of the molecules. Compared with existing work, APN leverages high-level human-defined attributes and helps the model to explicitly generalize knowledge in molecular graphs. Experiments on benchmark datasets show that APN consistently outperforms existing methods. In addition, we conduct extensive and comprehensive experiments on the fingerprint attributes and self-supervised attributes, demonstrating that the attributes are effective for improving few-shot MPP. We evaluate the performance of APN on three benchmark datasets, and the results show that APN can achieve state-of-the-art performance in most cases and demonstrate that the attributes are effective for improving few-shot MPP performance. In addition, the strong generalization ability and rationality of APN is verified by conducting experiments on data from different domains and ablation study.
![image]()

# Environments
## GPU environment

## create conda environment
PyTorch = 1.7.0
Numpy = 1.21.6
Rdkit = 2023.3.1
scikit-learn = 1.0.2
scipy = 1.7.3

```

# Usage
```
python main.py
```

