# Advanced Regression Techniques

#### Practicing regression in a Kaggle competition

A "knowledge" competition hosted by Kaggle to test advanced regression techniques. The aim of participating in this competition was to practice tackling a typical regression ML problem. The notebook includes data cleaning, EDA, building and interpreting the data and the model that I found to perform the best. The data cleaning methods I followed for this particular dataset was inspired by [this](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) popular notebook.

I purposely stuck to Linear models (Lasso, OLSR, GLMs etc) and avoided producing multi-model ensembles to boost my leaderboard score as model simplicity and interpretability was a priority of mine (see the interpretation section of the notebook). 

My final model was a simple OLSR with feature selection performed sequentially using [mlxtend](http://rasbt.github.io/mlxtend/) (0.12090 RMSLE, 1051/4887 on LB as of 24/08/2019).

From a brief perusal of the notebooks on Kaggle that scored better, here are some things that I could implement in the future to improve my LB score without resorting to ensembling:

- boxcox1p transforms for skewed features
- log1p transform the label instead of log
- more FE by creating indicator features that flag for categorical feature levels that spike the sale price (see step 3 of [this](https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force))
- MICE to impute missing values
- Sequential Feature Selection using a regularized linear model (e.g. Lasso) where the transformed features from above are included in the selection process

Any further improvements made to my solution will be mentioned here and updated in the notebook
