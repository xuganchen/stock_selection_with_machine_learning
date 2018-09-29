# stock_selection_with_machine_learning
Stock selection with machine learning 

* data/sample: data sample using in training and testing model
  * Training Model: the data using in "Training Model" in run.py
  * Calculate Portfolio: the data using in "Calculate Portfolio" in run.py

* results:
  the results of average weight portfolio

* code:
  the code of the project

  * run.py and run_sample.ipynb
    running all four models in model dictionary, and you can open run_sample.ipynb(or run_sample.html) to see how to using the model

  * factors:
    all indicators is get from [tushare](https://tushare.pro/) and calculated with [talib](https://github.com/mrjbq7/ta-lib)
    * finIndicatormd.md: finincial indicators lists
    * technical_index.py: technical indicators lists

  * model:
    all model class dictionary
    * base.py: abstract model class
    * logistic_regression.py
    * random_forest.py
    * support_vector_machine.py
    * deep_neural_network.py

  * portfolio.py
    calculate the equity when we have the weight of portfolio and plot

  * genetic_algorithm.py
    Genatic Algorithm using in factors selection

  * dataview.py
    load and process data from [jaqs](https://github.com/quantOS-org/JAQS)

  * data_generate.py
    generate dataview data into the format using in training model