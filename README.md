# MSIFT
Experimental codes for paper "Class-imbalanced Multi-source Information Fusion Transformer-based Neural Networks for Mechanical Fault Diagnosis with Limited Data".

论文正在Under Review！
代码即将公布，敬请关注，谢谢！


<div align=center>
<img src="https://github.com/Polimi-YuYue/MSIFT/blob/main/Framework.jpg" width="500px">
</div>

Abstract:Data-driven intelligent fault diagnosis methods have emerged as powerful tools for monitoring and maintaining the operating conditions of mechanical equipment. However, in real-world engineering scenarios, mechanical equipment typically operates under normal conditions, resulting in class-imbalanced and limited data. This situation gives rise to label bias and biased training. Meanwhile, the current multi-source information fault diagnosis research to date has tended to focus on fault identification rather than effective feature fusion strategies. To solve these issues, a class-imbalanced multi-source information fusion transformer-based neural network is proposed to model data-level and algorithm-level ideas in a unified deep network for achieving effective multi-source information fusion under the class-imbalanced and limited (L&I) working conditions. From a data-level perspective, a preprocessing operation is first employed to capture time-frequency information simultaneously. Subsequently, multi-source time-frequency information is fed into a feature extractor with an information discriminator to construct local and information-invariant feature maps with different scales to eliminate multi-source information domain shift. Then, the multi-source feature maps are modeled by a multi-source information transformer-based neural network to achieve effective feature fusion through cross-attention mechanism. Next, the global max pooling and global average pooling layers are leveraged to obtain the more representative features. Finally, from an algorithm-level perspective, a dual-stream diagnosis predictor with a binary diagnosis predictor and a multi-class diagnosis predictor is designed to synthesize the diagnostic results through a reweighting activation mechanism for addressing the L&I problems. Extensive experiments on four different multi-source information datasets show the superiority and promising performance of our method compared to the state-of-the-art methods.
