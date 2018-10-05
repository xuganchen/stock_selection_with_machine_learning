* 时间

  * 训练集：20050701-20121231
  * 测试集：20121231-20180629

* 数据来源：

  * 主要来自tushare(https://tushare.pro, 使用jaqs包(https://github.com/quantOS-org/JAQS)进行整合下载)，和国泰安数据服务中心(CSMAR， http://cn.gtadata.com)
  * 其中，训练ML模型所用的HS300相关信息、股票日行情数据（OHLC）和股票公司金融指标来源于Tushare；利用BL模型进行计算的股票日收益率、HS300日收益率、无风险利率等数据来源于CAMAR

* 采用的因子：

  * 采用123个金融指标因子和106个技术指标因子，前者由Tushare直接下载，后者自己写代码进行计算，一共229个因子

* 训练机器学习模型的数据格式：

  * 对于每天每只股票，我们有

    $$
    X_i(t) \in R^n, n为因子数目
    $$
    计算未来5天的
    $$
    Return-to-Std\ Ratio = \frac{Return\ of\ 5\ days}{Std\ of\ 5\ days'\ price}=\frac{price_4 / price_0}{Std(price_i, i = 0,...,4)}
    $$
    进行排序得到Rank值，选取Rank值前15的股票，记
    $$
    Y_i(t) = 1
    $$
    选取Rank值后15的股票，记
    $$
    Y_i(t) = 0
    $$
    忽略中间的股票。

    这样一来，我们就得到了
    $$
    (X_i(t), Y_i(t)), X_i(t) \in R^n, Y_i(t) \in \{0, 1\}
    $$
    作为我们的训练模型的基本数据格式

    

* 遗传算法GA

  * 最开始我们采用了229个因子，但是由于因子数量太多，其中必然存在某些因子是存在线性相关或者有噪声的，这些因子是多余的，会增加我们的计算压力

  * 因此我们采用遗传算法GA，在299个因子中寻找全集最优子集，以减少因子间的线性性，从而减低我们的计算压力

  * 在算法实现过程中，主要分成这么几步：

    * 生成染色体(newPopulation)：

      初始染色体个数为100，每个染色体服从
      $$
      N_i \sim Binomial(229, 0.5), i = 1,...100
      $$
      并且对染色体添加约束条件
      $$
      染色体中1的个数必须在（100/2, 229 - 100 /2) = (50, 179)之间
      $$
      以使得最后GA得到的结果不会有太多或者太少的因子

    * 适应性函数(Fitness):

      定义为：采用Logistic Regression在对应的因子集合子集上训练并测试得到的AUC值作为对应染色体的适应度

    * 选择(Selection):

      根据计算出来的适应度构建一个关于当前染色体集合的分布，其中更大适应度的人更有概率被选中

    * 交叉(Crossover):

      对于每一个染色体，有0.2的可能性会进行交叉作用

      有三种随机化的交叉方式：①相邻染色体一个位点交换，②相邻染色体前面一部分进行交叉交换，③相邻染色体多个位点交换

    * 变异(Mutation):

      对于每一个染色体，有0.1的可能性会进行变异作用，并且变异的位点数是随机化的

  * 对于遗传算法的有效性，我们可以根据遗传算法前后的策略效果进行比较

* 机器学习训练模型

  * DNN

  ```python
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, BatchNormalization
  from keras.optimizers import SGD
  from keras import regularizers
  
  input = self.X_train.shape[1]
  model = Sequential()
  reg = regularizers.L1L2(l2=0.01)
  model.add(Dense(input // 2, activation='relu',input_dim=input, kernel_regularizer=reg))
  model.add(Dropout(0.5))
  model.add(Dense(input // 4, activation='relu', kernel_regularizer=reg))
  model.add(BatchNormalization())
  model.add(Dense(2, activation='softmax', kernel_regularizer=reg))
  sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  ```
  ```python
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  dense_2 (Dense)              (None, 114)               26220     
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 114)               0         
  _________________________________________________________________
  dense_3 (Dense)              (None, 57)                6555      
  _________________________________________________________________
  batch_normalization_1 (Batch (None, 57)                228       
  _________________________________________________________________
  dense_4 (Dense)              (None, 2)                 116       
  =================================================================
  Total params: 33,119
  Trainable params: 33,005
  Non-trainable params: 114
  _________________________________________________________________
  ```

  * Logistic Regression

  ```python
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.regularizers import L1L2
  
  input = self.X_train.shape[1]
  model = Sequential()
  reg = L1L2(l1=0.01, l2=0.01)
  model.add(Dense(1, input_dim=input, activation="sigmoid", kernel_regularizer=reg))
  model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
  ```
  ```python
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  dense_1 (Dense)              (None, 1)                 230       
  =================================================================
  Total params: 230
  Trainable params: 230
  Non-trainable params: 0
  _________________________________________________________________
  ```

  * Support Vector Machine

  ```python
  from sklearn import svm
  
  model = svm.SVC(kernel=kernel, C=C, probability=True, verbose=0)
  ```

  ```python
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=0)
  ```

  * Naive Beyes

  ```python
  from sklearn.naive_bayes import BernoulliNB
  
  model = BernoulliNB(alpha=0.6)
  ```

  ```python
  BernoulliNB(alpha=0.6, binarize=0.0, class_prior=None, fit_prior=True)
  ```

  * Random Forest

  ```python
  from sklearn.ensemble import RandomForestClassifier
  
  model = RandomForestClassifier(n_estimators=100, max_depth=4, verbose=0)
  ```

  ```python
  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
              max_depth=4, max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
              oob_score=False, random_state=None, verbose=0,
              warm_start=False)
  ```

  * Gradient Boosting Machine

  ```python
  from sklearn.ensemble import GradientBoostingClassifier
  
  model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=4, verbose=0)
  ```

  ```python
  GradientBoostingClassifier(criterion='friedman_mse', init=None,
                learning_rate=0.1, loss='deviance', max_depth=4,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100,
                presort='auto', random_state=None, subsample=1.0, verbose=0,
                warm_start=False)
  ```

  * Bagging

  ```python
  from sklearn.ensemble import BaggingClassifier
  
  model = BaggingClassifier(n_estimators=100, verbose=0)
  ```

  ```python
  BaggingClassifier(base_estimator=None, bootstrap=True,
           bootstrap_features=False, max_features=1.0, max_samples=1.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
  ```

  * Extra Trees

  ```python
  from sklearn.ensemble import ExtraTreesClassifier
  
  model = ExtraTreesClassifier(n_estimators=100, max_depth=4, verbose=0)
  ```

  ```python
  ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
             max_depth=4, max_features='auto', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
             oob_score=False, random_state=None, verbose=0, warm_start=False)
  ```

  * Ada Boost

  ```python
  from sklearn.ensemble import AdaBoostClassifier
  
  model = AdaBoostClassifier(n_estimators=100)
  ```

  ```python
  AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
            learning_rate=1.0, n_estimators=100, random_state=None)
  ```

  * Ensemble Xgboost Stack

  ```python
  from sklearn.ensemble import *
  import xgboost as xgb
  
  ada = AdaBoostClassifier(n_estimators=100)
  bag = BaggingClassifier(n_estimators=100, verbose=0)
  et = ExtraTreesClassifier(n_estimators=100, max_depth=4, verbose=0)
  gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=4, verbose=0)
  rf = RandomForestClassifier(n_estimators=100, max_depth=4, verbose=0)
  model = xgb.XGBRegressor(max_depth=7, objective="reg:logistic", learning_rate=0.5, n_estimators=500)
  ```

  ```python
  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
         colsample_bytree=1, gamma=0, learning_rate=0.5, max_delta_step=0,
         max_depth=7, min_child_weight=1, missing=nan, n_estimators=500,
         n_jobs=1, nthread=None, objective='reg:logistic', random_state=0,
         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
         silent=True, subsample=1)
  ```

  ​	此模型将sklearn.emsemble中的5个模型和xgboost中的模型利用堆栈的方式进行模型合成，以期望能提高模型效果。

  ​	首先将训练集的2/3数据分别训练ada, bag, et, gb, rf 5个模型，将其得到的预测结果Y = (Y1, Y2, Y3, Y4, Y5)作为xgboost模型的输入数据，从而得到Stack模型


