from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=estimator,
                       scoring='neg_mean_squared_error',
                       X=X,
                       y=y,
                       train_sizes=np.linspace(0.1, 1.0, 11),
                       cv=cv,
                       n_jobs=1,
                       )

    train_mean = np.mean(np.sqrt(-train_scores), axis=1)
    train_std = np.std(np.sqrt(-train_scores), axis=1)
    test_mean = np.mean(np.sqrt(-test_scores), axis=1)
    test_std = np.std(np.sqrt(-test_scores), axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training RMSE')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation RMSE')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
