# stock_selection_with_machine_learning
Stock selection with machine learning 

### report

http://xuganchen.com/download/20181008StockML.pdf

### code:
the code of the project

* run.py 

  running all models in model dictionary

* factors:

  all indicators is get from [tushare](https://tushare.pro/) and calculated with [talib](https://github.com/mrjbq7/ta-lib)

  * finIndicatormd.md: finincial indicators lists

  * technical_index.py: technical indicators lists

* model:

  all model class dictionary

  * base.py: abstract model class

  * \_\_init\_\_.py: import model

* portfolio.py

  calculate the equity when we have the weight of portfolio and plot the equity

* genetic_algorithm.py

  Genatic Algorithm using in factors selection

* dataview.py

  load and process data from [jaqs](https://github.com/quantOS-org/JAQS)

* data_generate.py

  generate dataview data into the format using in training model

* blacklitterman.py

  the Black Litterman Model 
