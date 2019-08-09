from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class MissingDataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_all=False, impute_lotfrontage=False):
        self.impute_lotfrontage = impute_lotfrontage
        self.impute_all = impute_all
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        if self.impute_lotfrontage:
            imputer = SimpleImputer(strategy='median')
            lotfrontage = X_copy['LotFrontage'].values.reshape(-1, 1)
            lotfrontage = imputer.fit_transform(lotfrontage)
            X_copy['LotFrontage'] = lotfrontage
            if not self.impute_all:
                return X_copy

        X_copy["PoolQC"].fillna("None", inplace=True)
        X_copy["MiscFeature"].fillna("None", inplace=True)
        X_copy["Alley"].fillna("None", inplace=True)
        X_copy["Fence"].fillna("None", inplace=True)
        X_copy["FireplaceQu"].fillna("None", inplace=True)

        # Numerical garage features
        for col in ["GarageYrBlt", "GarageCars", "GarageArea"]:
            X_copy[col].fillna(0, inplace=True)
        # Categorical garage features
        for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
            X_copy[col].fillna("None", inplace=True)

        # Numerical basement features
        for col in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]:
            X_copy[col].fillna(0, inplace=True)

            # Categorical basement features
        for col in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
            X_copy[col].fillna("None", inplace=True)

        # Numerical Masonry Veneer
        X_copy["MasVnrArea"].fillna(0, inplace=True)
        # Categorical Masonry Veneer
        X_copy["MasVnrType"].fillna("None", inplace=True)

        for col in ["BsmtHalfBath", "BsmtFullBath"]:
            X_copy[col].fillna(0, inplace=True)

        # Fill in the rest of the categorical features using mode
        X_copy["Electrical"].fillna('SBrkr', inplace=True)
        X_copy["MSZoning"].fillna("RL", inplace=True)
        X_copy["Functional"].fillna("Typ", inplace=True)
        X_copy["Utilities"].fillna("Allpub", inplace=True)
        X_copy["SaleType"].fillna("WD", inplace=True)
        X_copy["KitchenQual"].fillna("TA", inplace=True)
        X_copy["Exterior1st"].fillna("VinylSd", inplace=True)
        X_copy["Exterior2nd"].fillna("VinylSd", inplace=True)

        # self.all_feature_names = X_copy.columns.tolist()

        return X_copy