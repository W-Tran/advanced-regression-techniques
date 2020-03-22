from sklearn.model_selection import learning_curve
import numpy as np


def compute_learning_curve(estimator, X, y, cv=5):
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

    learning_curve_params = {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std
    }

    return learning_curve_params


def plot_learning_curve(ax, lc_params):
    ax.plot(lc_params['train_sizes'], lc_params['train_mean'],
            color='blue', marker='o',
            markersize=5, label='training RMSE')

    ax.fill_between(lc_params['train_sizes'],
                    lc_params['train_mean'] + lc_params['train_std'],
                    lc_params['train_mean'] - lc_params['train_std'],
                    alpha=0.15, color='blue')

    ax.plot(lc_params['train_sizes'], lc_params['test_mean'],
            color='green', linestyle='--',
            marker='s', markersize=5,
            label='validation RMSE')

    ax.fill_between(lc_params['train_sizes'],
                    lc_params['test_mean'] + lc_params['test_std'],
                    lc_params['test_mean'] - lc_params['test_std'],
                    alpha=0.15, color='green')

    ax.grid()
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('RMSE')
    ax.legend()
