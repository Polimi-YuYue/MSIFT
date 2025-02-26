# MSIFT
Experimental codes for paper "MSIFT: A Novel End-to-End Mechanical Fault Diagnosis Framework under Limited & Imbalanced Data Using Multi-Source Information Fusion".

<div align=center>
<img src="https://github.com/Polimi-YuYue/MSIFT/blob/main/Framework.png" width="500px">
</div>

# Abstract

Data-driven intelligent fault diagnosis methods have emerged as powerful tools for monitoring and maintaining the operating conditions of mechanical equipment. However, in real-world engineering scenarios, mechanical equipment typically operates under normal conditions, resulting in limited and imbalanced (L&I) data. This situation gives rise to label bias and biased training. Meanwhile, the current multi-source information fault diagnosis research to date has tended to focus on fault identification rather than effective feature fusion strategies. To solve these issues, a novel end-to-end mechanical fault diagnosis framework under limited & imbalanced data using multi-source information fusion is proposed to model data-level and algorithm-level ideas in a unified deep network for achieving effective multi-source information fusion under the L&I working conditions. From a data-level perspective, a data preprocessing operation is first employed to capture time-frequency information simultaneously. Subsequently, multi-source time-frequency information is fed into feature extractors with information discriminators to construct local and information-invariant feature maps with different scales to eliminate multi-source information domain shift. Then, the multi-source feature vectors are modeled by a multi-source information transformer-based neural network to achieve effective multi-source information fusion through cross-attention mechanism. Next, the global max pooling and global average pooling layers are leveraged to obtain the more representative features. Finally, from an algorithm-level perspective, a dual-stream diagnosis predictor with a binary diagnosis predictor and a multi-class diagnosis predictor is designed to synthesize the diagnostic results through a reweighing activation mechanism for addressing the L&I problems. Extensive experiments on four different multi-source information datasets show the superiority and promising performance of our method compared to the state-of-the-art methods, as evidenced by indicators from various aspects.

# Paper

# MSIFT: A novel end-to-end mechanical fault diagnosis framework under limited & imbalanced data using multi-source information fusion

a. Yue Yu, a. Hamid Reza Karimi, b. Len Gelman, c. Ahmet Enis Cetin

a Department of Mechanical Engineering, Politecnico di Milano, via La Masa 1, Milan 20156, Italy

b School of Computing and Engineering, University of Huddersfield, Queensgate, Huddersfield, HD1 3DH, UK

c Department of Electrical and Computer Engineering, University of Illinois Chicago, Chicago, USA

https://www.sciencedirect.com/science/article/pii/S095741742500569X#:~:text=To%20solve%20these%20issues%2C%20a%20novel%20end-to-end%20mechanical,fusion%20is%20proposed%20to%20model%20data-level%20and%20algorithm-le

# If this code is helpful to you, please cite this paper as follows, thank you!
# Citation

@article{YU2025126947,
title = {MSIFT: A novel end-to-end mechanical fault diagnosis framework under limited & imbalanced data using multi-source information fusion},
journal = {Expert Systems with Applications},
volume = {274},
pages = {126947},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.126947},
url = {https://www.sciencedirect.com/science/article/pii/S095741742500569X},
author = {Yue Yu and Hamid Reza Karimi and Len Gelman and Ahmet Enis Cetin},}
