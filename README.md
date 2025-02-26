# MSIFT
Experimental codes for paper "MSIFT: A Novel End-to-End Mechanical Fault Diagnosis Framework under Limited & Imbalanced Data Using Multi-Source Information Fusion".


论文正在Under Review！
代码即将公布，敬请关注，谢谢！


<div align=center>
<img src="https://github.com/Polimi-YuYue/MSIFT/blob/main/Framework.jpg" width="500px">
</div>

# Abstract

Data-driven intelligent fault diagnosis methods have emerged as powerful tools for monitoring and maintaining the operating conditions of mechanical equipment. However, in real-world engineering scenarios, mechanical equipment typically operates under normal conditions, resulting in limited and imbalanced (L&I) data. This situation gives rise to label bias and biased training. Meanwhile, the current multi-source information fault diagnosis research to date has tended to focus on fault identification rather than effective feature fusion strategies. To solve these issues, a novel end-to-end mechanical fault diagnosis framework under limited & imbalanced data using multi-source information fusion is proposed to model data-level and algorithm-level ideas in a unified deep network for achieving effective multi-source information fusion under the L&I working conditions. From a data-level perspective, a data preprocessing operation is first employed to capture time-frequency information simultaneously. Subsequently, multi-source time-frequency information is fed into feature extractors with information discriminators to construct local and information-invariant feature maps with different scales to eliminate multi-source information domain shift. Then, the multi-source feature vectors are modeled by a multi-source information transformer-based neural network to achieve effective multi-source information fusion through cross-attention mechanism. Next, the global max pooling and global average pooling layers are leveraged to obtain the more representative features. Finally, from an algorithm-level perspective, a dual-stream diagnosis predictor with a binary diagnosis predictor and a multi-class diagnosis predictor is designed to synthesize the diagnostic results through a reweighing activation mechanism for addressing the L&I problems. Extensive experiments on four different multi-source information datasets show the superiority and promising performance of our method compared to the state-of-the-art methods, as evidenced by indicators from various aspects.


# Paper

A two-stage importance-aware subgraph convolutional network based on multi-source sensors for cross-domain fault diagnosis

a. Yue Yu, a. Youqian He, a. Hamid Reza Karimi, b. Len Gelman, c. Ahmet Enis Cetin

a Department of Mechanical Engineering, Politecnico di Milano, via La Masa 1, Milan 20156, Italy

b School of Computing and Engineering, University of Huddersfield, Queensgate, Huddersfield, HD1 3DH, UK

c Department of Electrical and Computer Engineering, University of Illinois Chicago, Chicago, USA

https://www.sciencedirect.com/science/article/pii/S0893608024004428

# If this code is helpful to you, please cite this paper as follows, thank you!
# Citation

@article{YU2024106518,
title = {A two-stage importance-aware subgraph convolutional network based on multi-source sensors for cross-domain fault diagnosis},
journal = {Neural Networks},
pages = {106518},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106518},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024004428},
author = {Yue Yu and Youqian He and Hamid Reza Karimi and Len Gelman and Ahmet Enis Cetin},}
