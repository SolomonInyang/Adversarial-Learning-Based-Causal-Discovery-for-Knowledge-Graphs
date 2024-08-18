# Overview

This repository contains a **preliminary** implementation of the Structural Agnostic Modelling (SAM) causal discovery method proposed by Kalainathan et al. (citation below) and applies it to the task of causal learning for knowledge graphs (KG). 

The SAM method generates causal graphs, i.e., graphical descriptions of the underlying causal structure for a set of data points, by framing the causal discovery problem as an adversarial learning task between different nodes that 
each estimate variable distributions conditionally and then compare their estimates to other nodes in the neural network as well as the discriminator.

A learning criterion that combines distribution estimation, sparsity and acyclicity constraints is used to enforce the optimization of the graph structure and parameters using the Stochastic Gradient Descent (SGD) algorithm. 

The SAM method is applied to causal discovery for KGs by first obtaining a low-dimensional tensor decomposition of each knowledge graph which is then passed as an input matrix to the SAM model for causal graph prediction.

# Datasets
The SAM model performs causal discovery on two knowledge graphs: the FB1k-237 dataset and the WN18RR dataset.

FB15k-237 is a subset of the FB15k dataset in which many of the inverse relations between entities have been removed. The dataset contains 310,116 triples with 14,541 entities and 237 relation types and is typically used to evaluate KG link prediction models. 

WN18RR is a link prediction dataset created from WN18, a subset of the WordNet knowledge graph, with the inverse relations removed. The WN18RR dataset contains 93,003 triples with 40,943 entities and 11 relation types.

# Running the Model
To run the SAM model, simply navigate to the main.py file, pass one of the datasets as an input into the generate_matrix() function (will be set to fb15k-237 by default) and then run the file. The output causal graphs for the FB15k-237 and WN18RR datasets are shown in Figure 1
and Figure 2 respectively.

![Alt text](https://github.com/user-attachments/assets/292de346-f1ba-40b3-b34d-71b8a7a2caed)

<p align="left">
  <em>Figure 1: A causal graph generated using the SAM method for a subset of the FB15K-237 dataset.</em>
</p>

![Alt text](https://github.com/user-attachments/assets/47ca27a4-8d54-4a27-8c67-51e81c9bfc0d)

<p align="left">
  <em>Figure 2: A causal graph generated using the SAM method for a subset of the WN18RR dataset.</em>
</p>

# Citations
```bibtex
@article{kalainathan2022structural,
  title={Structural agnostic modeling: Adversarial learning of causal graphs},
  author={Kalainathan, Diviyan and Goudet, Olivier and Guyon, Isabelle and Lopez-Paz, David and Sebag, Mich{\`e}le},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={219},
  pages={1--62},
  year={2022}
}

@article{kalainathan2020causal,
  title={Causal discovery toolbox: Uncovering causal relationships in python},
  author={Kalainathan, Diviyan and Goudet, Olivier and Dutta, Ritik},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={37},
  pages={1--5},
  year={2020}
}

@article{giriraj2021causal,
  title={Causal Discovery in Knowledge Graphs by Exploiting Asymmetric Properties of Non-Gaussian Distributions},
  author={Giriraj, Rohan and Thomas, Sinnu Susan},
  journal={arXiv preprint arXiv:2106.01043},
  year={2021}
}
