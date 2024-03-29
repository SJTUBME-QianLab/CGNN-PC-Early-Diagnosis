# CGNN-PC-Early-Diagnosis

This repository holds the codes of the paper: 

*Causality-Driven Graph Neural Network for Early Diagnosis of Pancreatic Cancer in Non-Contrast Computerized Tomography*

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

## Authorship

Xinyue Li, Rui Guo, Jing Lu, Tao Chen, Xiaohua Qian

## **Abstract**

Pancreatic cancer is the emperor of all cancer maladies, mainly because there are no characteristic symptoms in the early stages, resulting in the absence of effective screening and early diagnosis methods in clinical practice. Non-contrast computerized tomography (CT) is widely used in routine check-ups and clinical examinations. Therefore, based on the accessibility of non-contrast CT, an automated early diagnosis method for pancreatic cancer is proposed. Among this, we develop a novel causality-driven graph neural network to solve the challenges of stability and generalization of early diagnosis, that is, the proposed method achieves stable performance for datasets from different hospitals, which highlights its clinical significance. Specifically, a multiple-instance-learning framework is designed to extract fine-grained pancreatic tumor features. Afterwards, to ensure the integrity and stability of the tumor features, we construct an adaptive-metric graph neural network that effectively encodes prior relationships of spatial proximity and feature similarity for multiple instances, and hence adaptively fuses the tumor features. Besides, a causal contrastive mechanism is developed to decouple the causality-driven and non-causal components of the discriminative features, suppress the non-causal ones, and hence improve the model stability and generalization. Extensive experiments demonstrated that the proposed method achieved the promising early diagnosis performance, and its stability and generalizability were independently verified on a multi-center dataset. Thus, the proposed method provides a valuable clinical tool for the early diagnosis of pancreatic cancer. 
