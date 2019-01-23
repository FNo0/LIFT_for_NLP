# LIFT_for_NLP
2018BDCI汽车行业用户观点主题及情感识别rank27
# 问题描述：
本赛题提供一部分网络中公开的用户对汽车的相关内容文本数据作为训练集，训练集数据已由人工进行分类并进行标记，
参赛队伍需要对文本内容中的讨论主题和情感信息来分析评论用户对所讨论主题的偏好。
讨论主题可以从文本中匹配，也可能需要根据上下文提炼。
# 问题分析：
主要分为两步，第一步预测主题，第二步预测情感。
因为一个评论可能对应多个主题，因此使用到多标签分类。在这里采用的是2011年发表在IJCAI的LIFT算法。其主要思想是对每个标签分正负样本分别聚类，以原始特征与聚类中心点的距离作为标签特定特征。预测情感采用的是传统的xgboost算法。特征用的是onehot分词后的词语。
最后再通过情感词典对情感进行修正。
![](LIFT.png)
# 总结
亮点在于对多标签算法LIFT的使用。但最后依然不敌各种网络。情感词典修正算是一个人工的过程，最后答辩的前5支队伍有4支都使用过，但是这里严格控制情感词来源于训练集，且都是比较短的词语。这个题目使用bert效果比较好。
