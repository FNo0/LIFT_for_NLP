# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:26:56 2018

@author: FNo0
"""

import pandas as pd
import numpy as np
import jieba
jieba.initialize()
import re
from collections import Counter
import scipy.io as sio
import LIFT
import sentiment_model
import os
import pickle
import itertools
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

def load_data():
    train = pd.read_csv(r'../train/train.csv')
    test = pd.read_csv(r'../test_public/test_public.csv')
    return train,test

def split_words(dataset):
    data = dataset.copy()
    jieba.add_word('森林人')
    words = list(map(lambda x : jieba.cut(''.join(re.findall(u'[\u4e00-\u9fff]+',x)),cut_all = True,HMM = True),data['content'])) # jieba分词,只切中文
#    words = list(map(lambda x : jieba.cut(re.sub('\s+','',x),cut_all = False,HMM = True),data['content'])) # jieba分词
    words = [list(word) for word in words] # 分词结果转换为list
    data['words'] = words
    return data

def one_hot_improve(train,test,threshold):
    tr = train.copy()
    te = test.copy()
    tr['tr_or_te'] = 'tr'
    te['tr_or_te'] = 'te'
    data = pd.concat([tr,te],axis = 0)
    data.index = range(len(data))
    words = data['words'].tolist()
    words = [n for a in words for n in a ]# 转一维列表
    f = open(r'stopwords.txt')
    stopwords = []
    for row in f.readlines():
        stopwords.append(row)
    f.close()
    stopwords = [word.replace('\n','').strip() for word in stopwords]
    words = [word for word in words if word not in stopwords]
    words = dict(Counter(words))
    words = {k: v for k, v in words.items() if v > threshold}
    words = list(words.keys())
    data['words'] = data['words'].map(lambda x : list(set(x) & set(words)))
    data['words'] = data['words'].map(lambda x : dict([[y,1] for y in x]))
    feat = pd.DataFrame(data['words'].tolist())
    data = pd.concat([data,feat],axis = 1)
    data.fillna(0,downcast = 'infer',inplace = True)
    tr = data[data['tr_or_te'] == 'tr']
    te = data[data['tr_or_te'] == 'te']
    tr.drop(['tr_or_te'],axis = 1,inplace = True)
    te.drop(['tr_or_te'],axis = 1,inplace = True)
    tr.index = range(len(tr))
    te.index = range(len(te))
    return tr,te

def tfidf_improve(train,test):
    tr = train.copy()
    te = test.copy()
    tr['tr_or_te'] = 'tr'
    te['tr_or_te'] = 'te'
    data = pd.concat([tr,te],axis = 0)
    data.index = range(len(data))
    # 训练集词语
    words_tr = tr['words'].tolist()
    words_tr = [n for a in words_tr for n in a] # 转一维列表
    # 测试集词语
    words_te = te['words'].tolist()
    words_te = [n for a in words_te for n in a] # 转一维列表
    # 训练集和测试集同时出现的词语
    words_set_tr = list(set(words_tr))
    words_set_te = list(set(words_te))
    words = [word for word in words_tr if word in words_set_te]
    words2 = [word for word in words_te if word in words_set_tr]
    words.extend(words2)
    words = dict(Counter(words))
    # 分词结果
    words = list(words.keys())
    data['words'] = data['words'].map(lambda x : list(set(x) & set(words)))
    data = data[data['words'].map(lambda x : x != [])]
    data['words'] = data['words'].map(lambda x : ' '.join(x))
    # tfidf
    vectorizer = CountVectorizer(token_pattern = r'\b\w+\b',min_df = 1)#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data['words'].tolist()))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    weight = pd.DataFrame(weight,columns = words)
    data = pd.concat([data,weight],axis = 1)
    data.fillna(0,downcast = 'infer',inplace = True)
    tr = data[data['tr_or_te'] == 'tr']
    te = data[data['tr_or_te'] == 'te']
    tr.drop(['tr_or_te'],axis = 1,inplace = True)
    te.drop(['tr_or_te'],axis = 1,inplace = True)
    tr.index = range(len(tr))
    te.index = range(len(te))
    # 返回
    return tr,te

def trans_ml(train,test):
    tr = train.copy()
    te = test.copy()
    # 多标签
    tr = pd.concat([tr,pd.get_dummies(tr['subject'],prefix = 'subject')],axis = 1)
    subject_cols = ['subject_' + i for i in sorted(list(set(tr['subject'])))]
    subject = pd.pivot_table(tr,index = 'content_id',values = subject_cols,aggfunc = np.sum)
    tr.drop_duplicates('content_id',keep = 'first',inplace = True)
    tr.drop(subject_cols,axis = 1,inplace = True)
    tr = pd.merge(tr,subject,on = 'content_id',how = 'left')
    # for matlab
    train_x = tr.drop(['content_id','content','sentiment_value','sentiment_word','subject','words'],axis=1)
    train_x.drop(subject_cols,axis = 1,inplace = True)
    test_x = te.drop(['content_id','content','sentiment_value','sentiment_word','subject','words'],axis=1)
    train_y = tr[subject_cols]
    for col in subject_cols:
        train_y[col] = train_y[col].map(lambda x : 1 if x == 1 else -1)
    test_y = te[['content_id']]
    for col in subject_cols:
        test_y[col] = -1
    test_y.drop(['content_id'],axis = 1,inplace = True)
    # for lift
    train_data = train_x.applymap(float).values
    train_target = train_y.applymap(float).values.T
    test_data = test_x.applymap(float).values
    test_target = test_y.applymap(float).values.T
    # 输出
    sio.savemat('../tmp/subject_lift.mat',{'train_data' : train_data,'train_target' : train_target,'test_data' : test_data,'test_target' : test_target})
    
def keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds):
    # 原始训练集
    tr = train.copy()
    # 原始预测结果
    res = result.copy()
    
    ### 匹配关键词
    ## 训练集
    tr = tr[tr['subject'] == subject]
    tr['neg'] = tr['content'].map(lambda x : [key for key in words_neg if key in x])
    tr['pos'] = tr['content'].map(lambda x : [key for key in words_pos if key in x])
    # 筛掉负向正向否定词包含的词语
    tr['neg_pos'] = list(map(lambda x,y : [words for words in itertools.product(x,[y])],tr['neg'],tr['pos']))
    tr['neg'] = tr['neg_pos'].map(lambda x : [x[i][0] for i in range(len(x)) if all([x[i][0] not in x[i][1][j] for j in range(len(x[i][1]))]) == True])
    tr['pos_neg'] = list(map(lambda x,y : [words for words in itertools.product(x,[y])],tr['pos'],tr['neg']))
    tr['pos'] = tr['pos_neg'].map(lambda x : [x[i][0] for i in range(len(x)) if all([x[i][0] not in x[i][1][j] for j in range(len(x[i][1]))]) == True])
    tr.drop(['neg_pos','pos_neg'],axis = 1,inplace = True)
    # 包含负向词和正向词
    tr_neg = tr[tr['neg'].map(lambda x : x != [])]
    tr_pos = tr[tr['pos'].map(lambda x : x != [])]
    tr = pd.concat([tr_neg,tr_pos],axis = 0)
    tr.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    ## 测试集
    res = res[res['subject'] == subject]
    res['neg'] = res['content'].map(lambda x : [key for key in words_neg if key in x])
    res['pos'] = res['content'].map(lambda x : [key for key in words_pos if key in x])
    # 筛掉负向正向否定词包含的词语
    res['neg_pos'] = list(map(lambda x,y : [words for words in itertools.product(x,[y])],res['neg'],res['pos']))
    res['neg'] = res['neg_pos'].map(lambda x : [x[i][0] for i in range(len(x)) if all([x[i][0] not in x[i][1][j] for j in range(len(x[i][1]))]) == True])
    res['pos_neg'] = list(map(lambda x,y : [words for words in itertools.product(x,[y])],res['pos'],res['neg']))
    res['pos'] = res['pos_neg'].map(lambda x : [x[i][0] for i in range(len(x)) if all([x[i][0] not in x[i][1][j] for j in range(len(x[i][1]))]) == True])
    res.drop(['neg_pos','pos_neg'],axis = 1,inplace = True)
    # 包含负向词和正向词
    res_neg = res[res['neg'].map(lambda x : x != [])]
    res_pos = res[res['pos'].map(lambda x : x != [])]
    te = pd.concat([res_neg,res_pos],axis = 0)
    te.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    
    ### 再次训练
    tr = split_words(tr)
    te = split_words(te)
    tr,te = tfidf_improve(tr,te)    
    tr.drop(['neg','pos'],axis = 1,inplace = True)
    te.drop(['neg','pos'],axis = 1,inplace = True)
    pre = sentiment_model.model_xgb(tr,te,'sentiment_value',params,rounds)
    r = pre.drop_duplicates('content_id',keep='first')
    r['subject'] = subject
    
    ### 整合
    r['sentiment_word'] = np.nan
    res.drop(['neg','pos'],axis = 1,inplace = True)
    r = r[res.columns.tolist()]
    res = pd.merge(res,r.rename(columns = {'sentiment_value' : 'sentiment_value_change'}),how = 'left',on = ['content_id','content','subject'])
    res['sentiment_value'] = list(map(lambda x,y : y if (str(y) != 'nan' and x == 0) else x,res['sentiment_value'],res['sentiment_value_change']))
    res.drop(['sentiment_value_change','sentiment_word_y'],axis = 1,inplace = True)
    res.rename(columns = {'sentiment_word_x' : 'sentiment_word'},inplace = True)
    
    ### 返回
    return res,r

if __name__ == '__main__':
    ##### subject
    train,test = load_data()
    train = split_words(train)
    test = split_words(test)
    train,test = one_hot_improve(train,test,15)
    
    #### 转为多标签
    trans_ml(train,test)
    
    print('LIFT:')
    #### LIFT
    train_data,train_target,test_data,test_target = LIFT.load_data(r'../tmp/subject_lift.mat')
    ratio = 0.6
    # 聚类
    if os.path.exists(r'../tmp/P_Centers.pkl') and os.path.exists(r'../tmp/N_Centers.pkl'):
        P_Centers = pickle.load(open(r'../tmp/P_Centers.pkl','rb'))
        N_Centers = pickle.load(open(r'../tmp/N_Centers.pkl','rb'))
    else:
        P_Centers,N_Centers = LIFT.find_key_instances(train_data,train_target,ratio)
    # 保存P_Centers和N_Centers
    if os.path.exists(r'../tmp/P_Centers.pkl') and os.path.exists(r'../tmp/N_Centers.pkl'):
        pass
    else:
        pickle.dump(P_Centers,open(r'../tmp/P_Centers.pkl','wb'),protocol = 4)
        pickle.dump(N_Centers,open(r'../tmp/N_Centers.pkl','wb'),protocol = 4)
    # 训练
    Models = LIFT.train(train_data,train_target,P_Centers,N_Centers)
    # 预测
    Pre_Probs = LIFT.test(train_data,train_target,test_data,P_Centers,N_Centers,Models)
    # 保存LIFT的中间结果
    Pre_Probs.to_pickle(r'../tmp/Pre_Probs.gz',compression = 'gzip')
    
    #### 解析LIFT结果
    Pre_Probs = pd.read_pickle(r'../tmp/Pre_Probs.gz',compression = 'gzip')
    res_subject = Pre_Probs.copy()
    # 取1
    res_subject['max'] = list(map(lambda x0,x1,x2,x3,x4,x5,x6,x7,x8,x9 : \
              [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9].index(max([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9])),\
              res_subject['0'],res_subject['1'],res_subject['2'],res_subject['3'],res_subject['4'],\
              res_subject['5'],res_subject['6'],res_subject['7'],res_subject['8'],res_subject['9']))
    for i in res_subject.index:
        res_subject.loc[i,str(res_subject.loc[i,'max'])] = 1
    res_subject = res_subject.applymap(lambda x : 1 if x >= 0.7 else 0)
    res_subject.drop(['max'],axis = 1,inplace = True)
    # 竖起来
    subject_cols = ['subject_' + i for i in sorted(list(set(train['subject'])))]    
    res_subject.columns = subject_cols
    res_subject = pd.concat([test[['content_id','content']],res_subject],axis = 1)
    res_subject.set_index(['content_id','content'],inplace = True)
    res_subject = res_subject.stack()
    res_subject = pd.DataFrame(res_subject)
    res_subject.reset_index(inplace = True)
    res_subject['subject'] = res_subject['level_2']
    res_subject.drop(['level_2'],axis = 1,inplace = True)
    res_subject['subject'] = res_subject['subject'].map(lambda x : x.split('_')[1])
    res_subject = res_subject[res_subject[0] == 1]
    
    print('sentiment:')
    ##### sentiment
    train,test = load_data()
    train = split_words(train)
    test = split_words(test)
    train,test = one_hot_improve(train,test,10)
    ## 分subject建模
    subjects = sorted(list(set(train['subject'])))
    # 10模型参数
    params,rounds = sentiment_model.get_params_rounds(train)
    predict = pd.DataFrame()
    for i in range(len(subjects)):
        tr = train[train['subject'] == subjects[i]]
        te = pd.merge(res_subject[res_subject['subject'] == subjects[i]][['content_id']],test,how = 'left',on = 'content_id')
        pre_sentiment_value = sentiment_model.model_xgb(tr,te,'sentiment_value',params[i],rounds[i])
        pre_sentiment_value['subject'] = subjects[i]
        predict = pd.concat([predict,pre_sentiment_value],axis = 0)
        print(str(i) + ' ' + subjects[i] + '预测完毕!')
    predict.rename(columns = {0 : 'split_prob'},inplace = True)
    ### 单模型建模
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric' : 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,#0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 0,  # 2 3
              'num_class':len(set(train['sentiment_value'])),
              'lambda' : 2
              }
    rounds = 400
    result = sentiment_model.model_xgb(train,test,'sentiment_value',params,rounds)
    result = pd.merge(res_subject.drop([0],axis = 1),result,on = ['content_id','content'],how = 'left')
    result.rename(columns = {0 : 'all_prob'},inplace = True)
    ## 融合
    result = pd.merge(result,predict,on = ['content_id','content','sentiment_value','subject'],how = 'left')
    result['prob'] = result['all_prob'] * 0.7 + result['split_prob'] * 0.3
    result.sort_values('prob',ascending = False,inplace = True)
    result = result.drop_duplicates(['content_id','content','subject'],keep = 'first')
    
    ##### 处理结果
    result['sentiment_word'] = np.nan
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    ##### 关键词匹配
    #### 价格
    subject = '价格'
    words_neg = ['高','贵','不低','不便宜','不会便宜','便宜货','低档','无优惠','没有优惠','没优惠','没啥优惠','优惠太小','才优惠','只优惠','不值','不保值','保值率低',\
              '性价比不高','性价比不算高','性价比不算很高','性价比低','性价比很低','性价比超级低','性价比非常低','不合适','不合算','不划算','不靠谱','宰','黑',\
              '尴尬','惊人','不考虑','不用考虑','不良心','不实惠','不省心','廉价','套路','抢钱','不菲','不合理','不实在','不放心']
    words_pos = ['不高','不虚高','不贵','不是很贵','低','便宜','有优惠','优惠不错','优惠可以','优惠幅度不小','优惠不小','超级优惠','优惠大','优惠挺大','最优惠',\
              '值','性价比高','性价比蛮高','性价比比较高','性价比很高','合适','合算','划算','靠谱','良心','实惠','省心','不错','合理','实在','放心']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,  # 0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 14,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              }
    rounds = 400
    result_jg,jg = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    ### 重新合并
    result_jg = result_jg[result.columns.tolist()]
    result = pd.concat([result_jg,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    #### 内饰
    subject = '内饰'
    words_neg = ['普通','落后','松垮','旧','差','挫','破','糙','塑料','简陋','败','水','过时','毛病','脱皮','坏','老气','低档','异味','粗','味道',\
                 '扯','渣','辣鸡','就那样','偷工减料','难受','单调','硬','噪音','吐','寒酸','异响','没得比','不咋地','不敢恭维','裂','薄','惨','丑','臭','锈',\
                 '廉价','LOW','low','chou','磕碜','土','污染','问题','懒得','平淡','不细致','不如','不是一个档次','不在一个档次','不一个档次','不上档次',
                 '不够','没用','不满意','不好','不喜欢','看不上','不能接受','没戏','不豪华','不齐','不漂亮','不舒坦','不符合','不值']
    words_pos = ['细致','满意','好','喜欢','豪华','漂亮','舒坦','上档次','不差','没有味道','无味道','没有异味','无异味','没有毛病','无毛病','没有脱皮','不脱皮',\
                 '没有噪音','无噪音','没有问题','无问题','不错','放松','优点','认可度','皮实','耐用','符合','一致','手感','不逊','强','没得说',\
                 '没的说','值','豪华','口味','可以','简洁','干练','提升','升级','时尚','真皮','酷炫','精致','带感','高出一个档次','科技','质感','舒适','细',\
                 '骚','高大上','秒杀','胜','帅','新','吊打','接受','高贵','气场','高级']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,  # 0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 14,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              }
    rounds = 600
    result_ns,ns = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    ### 重新合并
    result_ns = result_ns[result.columns.tolist()]
    result = pd.concat([result_ns,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]

    #### 外观
    subject = '外观'
    words_neg = ['差','丑','薄','难看','小','中庸','老气','驼背','硬伤','畸形','抖','脱落','平庸','偷工减料','败','獠牙','软','土','平淡','毁','尴尬',\
                 '低级','锈','low','LOW','chou','变色','纠结','挫','倾斜大','不好','不喜欢','接受度不高','接受不了','不年轻','不够','不符合','不协调','没有质感',\
                 '没有丝毫质感','不敢恭维','不行','跟不上时尚','不中看','不咋','吐槽','无爱','无下限','撑不起','不怎么']
    words_pos = ['不差','不丑','不薄','不难看','不小','不中庸','不老气','不驼背','不畸形','不脱落','不平庸','不软','不土','不平淡','不挫','好','喜欢','接受',\
                 '年轻','协调','质感','时尚','中看','爱','逼格','迷恋','支持','认可度高','尊贵','炫','一致','大','沉稳','猛','对口味','看中','不错','厉害','秒杀',\
                 '漂亮','眼缘','合','优势','气质','强','颠覆','硬','颜值','档次','秀','厚','赞','可以','精致','吊打','有面子','帅','拉风','完整','抢眼']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.01,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.9,  # 0.8
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'min_child_weight': 2,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              'lambda' : 2,
#              'gamma' : 0.5
              }
    rounds = 800
    result_wg,wg = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    ### 重新合并
    result_wg = result_wg[result.columns.tolist()]
    result = pd.concat([result_wg,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    #### 操控
    subject = '操控'
    words_neg = ['就那样','就这样','那么回事','病','差','渗油','多余','颠','晃','单薄','败','问题','抖','响','爆震','迟疑','掉','偏','脱皮','烂','延迟',\
                 '破','漏','无语','磨','软','味道','呕','重','滑','廉价','能比','辣鸡','打脸','傻','缺','pass','弱','慢','费','一般','轻','飘','迟钝',\
                 '飞叉叉','有毛用','摆动','歪','异常','危险','冻','威','滞','吓人','玩意','逗比','屁','号称','可怕','怄气','晕','废','呵','迷信','沉',\
                 '比不上','比不了','并没有','并非','没办法','没法','无法','不如','不好','不行','不怎么样','不佳','不及','不能','不适','不咋地','完全不是',\
                 '没有性能','不够','不放心','不敢恭维','不存在','不一样','承受不起','没优势','无解','没用','出大事']
    words_pos = ['不差','一','追求','好','名副其实','灵','看中','值','棒','漂移','专业','不错','满意','优势','兼顾','喜欢','符合','牛','舒服','厉害',\
                 '没问题','不会有问题','没有问题','韧','操控性','没得说','没的说','稳','提高','准','讲究','操控感','优点','强','玩','宝马','档次','完爆',\
                 '通过性','可以','轻快','不发飘','完美','有操控','没话说','超级无敌','闻名天下','毋庸置疑','可靠','乐趣','胜','提升','逼格','范']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.01,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 1,  # 0.8
              'subsample': 1,
              'scale_pos_weight': 1,
              'min_child_weight': 15,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              'lambda' : 3
              }
    rounds = 400
    result_ck,ck = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    # ?规则
    result_ck['sentiment_value'] = list(map(lambda x,y : 0 if x == 1 and '？' in y else x,result_ck['sentiment_value'],result_ck['content']))
    ck['sentiment_value'] = list(map(lambda x,y : 0 if x == 1 and '？' in y else x,ck['sentiment_value'],ck['content']))
    ### 重新合并
    result_ck = result_ck[result.columns.tolist()]
    result = pd.concat([result_ck,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    #### 配置
    subject = '配置'
    words_neg = ['差','醉','奇葩','坑','吐槽','鸡肋','山寨','手机导航','垃圾','费电','刺眼','弱','低','强奸','凉','批判','硬伤','简',\
                 '索赔','故障','气死','病','辣鸡','伤心','哭','挫','臭','惨','屎','配置不高','不太爽','没意义','没用','华而不实',\
                 '没多大优势','没有','性价比不高','不足','不好','不喜欢','不能接受','不行','不要买','不厚道','看不上','tmd','无用',\
                 '不能忍','无语','不咋样','不方便','后悔','不够']
    words_pos = ['不差','不坑','不费电','不刺眼','强','不弱','高','没问题','没有问题','没毛病','没有毛病','好','厚道','方便','不挫','不臭',\
                 '喜欢','秒杀','羡慕','完美','提升','不错','良心','安静','舒适','不后悔','够','多了','出色','很全','实用','人性','心动']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.7,  # 0.8
              'subsample': 1,
              'scale_pos_weight': 1,
              'min_child_weight': 30,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              'lambda' : 2,
              }
    rounds = 400
    result_pz,pz = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    # ?规则
    result_pz['sentiment_value'] = list(map(lambda x,y : 0 if x == 1 and '？' in y or '?' in y else x,result_pz['sentiment_value'],result_pz['content']))
    pz['sentiment_value'] = list(map(lambda x,y : 0 if x == 1 and '？' in y or '?' in y else x,pz['sentiment_value'],pz['content']))
    ### 重新合并
    result_pz = result_pz[result.columns.tolist()]
    result = pd.concat([result_pz,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]

    #### 舒适性
    subject = '舒适性'
    words_neg = ['噪','反光','咯','叽','大','热死','晒','疼','差','漏','响','颠','问题','破','折腾','卡顿','硬','格芝','震动',\
                 '担心','妈的','当当当','呛','抖','吱','失望','间隙','味','滋','糊','爆震','病','TMD','喀','嗒','进水','毒',\
                 '座椅小','塑料','翁','轰鸣','累','薄','短','烦','臭','超标','凑合','忽悠','晕','哒哒','频繁','凉','滴','一般',\
                 '刺鼻','啸叫','不小','缺陷','软','降低','廉价','太次','挫','不太爽','不舒服','不是很舒适','受不了','不怎么样',\
                 '毫无','不是真皮','不行','不咋地','不咋样','不好','不足','不畅','不敢恭维','不消停','不方便','不严','不制冷','无语']
    words_pos = ['噪音小','噪音低','噪音不大','无噪音','没有风噪','低','小','减少','不差','没问题','没有问题','没啥问题','没有毛病',\
                 '没毛病','不臭','爽','舒服','舒适','好','方便','给力','强劲','足','橡胶软','座椅加热','座椅记忆','凉爽','制冷快','恒温',\
                 '厉害','隔音','满意','电动调节','静音','提升','没的说','安静','人性','实用','秒杀','平顺','平滑','有力','提高','可以','宽敞',\
                 '不错','不大','不吵']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.7,  # 0.8
              'subsample': 1,
              'scale_pos_weight': 1,
              'min_child_weight': 0,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              'lambda' : 2,
              }
    rounds = 200
    result_ssx,ssx = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    ### 重新合并
    result_ssx = result_ssx[result.columns.tolist()]
    result = pd.concat([result_ssx,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    #### 油耗
    subject = '油耗'
    words_neg = ['高','费油','牛b','牛逼','服了','厉害','大','耗油','假','增加','上涨','蹭蹭','不得了','多','上升','吓人',\
                 '不喜欢','不低','低不了','不太省','不省','不好','不满意']
    words_pos = ['不高','不费油','不大','不多','喜欢','低','省','可以','无变化','稳定','给力','节油','满意','好','不会高','下降',\
                 '不错','不是特别高','棒','经济','不会很']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 1,  # 0.8
              'subsample': 1,
              'scale_pos_weight': 1,
              'min_child_weight': 12,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              'lambda' : 5,
              }
    rounds = 400
    result_yh,yh = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    ### 重新合并
    result_yh = result_yh[result.columns.tolist()]
    result = pd.concat([result_yh,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    #### 空间
    subject = '空间'
    words_neg = ['小','硬伤','限制','挤','败','差','不大','不太大','不喜欢','不咋地','不好','不行','不满意','不够','不敢恭维','不舒服',\
                 '不漂亮','不宽','不高','不多','不舒适','紧张','不放松']
    words_pos = ['不小','不挤','不差','大','喜欢','好','行','满意','够','舒服','可以','漂亮','理想','宽','高','多','舒适','优势','不错',\
                 '出色','放松','保障','满足','杠杠','提升']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.03,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 1,  # 0.8
              'subsample': 1,
              'scale_pos_weight': 1,
              'min_child_weight': 12,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              'lambda' : 5,
              }
    rounds = 400
    result_kj,kj = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    # 规则
    result_kj = result_kj[result_kj['sentiment_value'] != 1] # 只保留其中负的
    ### 重新合并
    result_kj = result_kj[result.columns.tolist()]
    result = pd.concat([result_kj,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    #### 安全性
    subject = '安全性'
    words_neg = ['软','差','心寒','问题','醉','恼人','追尾','异响','无奈','垃圾','辣鸡','磨耗','抖','松动','滋滋','妈蛋','故障',\
                 '不强','不耐磨','刹不住','不给力','不抱死','不安全','不好','不扎实','不安全','不稳']
    words_pos = ['不软','不差','没问题','没有问题','不错','不输','不溜坡','不逊色','不弱','强','耐磨','给力','安全','一致','兼顾',\
                 '提升','适合','好','有效果','皮实','扎实','安全感','硬','正名','稳','优越','心动','优秀','合适']
    train,test = load_data();del test
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'merror',
              'eta': 0.01,
              'max_depth': 4,  # 4 3
              'colsample_bytree': 0.7,  # 0.8
              'subsample': 0.7,
              'scale_pos_weight': 1,
              'min_child_weight': 30,
              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
              'lambda' : 5,
              }
    rounds = 400
    result_aqx,aqx = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
    # ?规则
    result_aqx['sentiment_value'] = list(map(lambda x,y : 0 if x == 1 and '？' in y or '?' in y else x,result_aqx['sentiment_value'],result_aqx['content']))
    aqx['sentiment_value'] = list(map(lambda x,y : 0 if x == 1 and '？' in y or '?' in y else x,aqx['sentiment_value'],aqx['content']))
    ### 重新合并
    result_aqx = result_aqx[result.columns.tolist()]
    result = pd.concat([result_aqx,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
#    #### 动力
#    subject = '动力'
#    words_neg = ['爆震','差','担心','抖','坑','垃圾','慢','缺陷','肉','弱','烧','渗油','熄火','降','严重','异响','呵呵','衰减','缓慢','无力'
#                 '不够','不行','不好','不给力','不快','不灵','不满意','问题','不猛','不强','不舒服','不爽','没动力','不高','不跟脚','不佳','不足',]
#    words_pos = ['不差','不抖','不坑','不慢','不弱','不烧机油','没有烧机油','不错','不后悔','够','行','好','足','超车','窜','杠杠','给力','快','灵',\
#                 '满意','没问题','猛','强','全时四驱','舒服','爽','有动力','充沛','高','跟脚','可以','尚可','王','出色','轻轻松松']
#    train,test = load_data();del test
#    params = {'booster': 'gbtree',
#              'objective': 'multi:softprob',
#              'eval_metric': 'merror',
#              'eta': 0.03,
#              'max_depth': 4,  # 4 3
#              'colsample_bytree': 1,  # 0.8
#              'subsample': 1,
#              'scale_pos_weight': 1,
#              'min_child_weight': 20,
#              'num_class': len(set(train[train['subject'] == subject]['sentiment_value'])),
#              'lambda' : 5,
#              }
#    rounds = 800
#    result_dl,dl = keywords_sentiment(subject,words_neg,words_pos,train,result,params,rounds) 
#    # ?规则
#    result_dl = result_dl[result_dl['sentiment_value'] != 1] # 只保留其中负的
#    ### 重新合并
#    result_dl = result_dl[result.columns.tolist()]
#    result = pd.concat([result_dl,result],axis = 0)
#    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
#    result['sentiment_value'] = result['sentiment_value'].map(int)
#    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    # 动力
    result_dl = result[result['subject'] == '动力']
    result_dl['sentiment_value'] = 0
    ### 重新合并
    result_dl = result_dl[result.columns.tolist()]
    result = pd.concat([result_dl,result],axis = 0)
    result.drop_duplicates(['content_id','content','subject'],keep = 'first',inplace = True)
    result['sentiment_value'] = result['sentiment_value'].map(int)
    result = result[['content_id','content','subject','sentiment_value','sentiment_word']]
    
    
    
    ##### 最终结果
    result.drop(['content'],axis = 1,inplace = True)
    result = result.sample(frac = 1)
    result.to_csv(r'../result/6585.csv',index = False,encoding = 'utf-8') 
    
    
    
    
    
    
    
