import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CatTransformer(BaseEstimator, TransformerMixin):
    # Ordinal mappings
    _alley_mapping = {'none': 1, 'grvl': 2, 'pave': 3}
    _utilities_mapping = {'elo': 1, 'nosewa': 2, 'nosewr': 3, 'allpub': 4}
    _landslope_mapping = {'gtl': 1, 'mod': 2, 'sev': 3}
    _lotshape_mapping = {'reg': 1, 'ir1': 2, 'ir2': 3, 'ir3': 4}
    _landcontour_mapping = {'lvl': 1, 'bnk': 2, 'hls': 3, 'low': 4}
    _exterqual_mapping = {'po': 1, 'fa': 2, 'ta': 3, 'gd': 4, 'ex': 5}
    _extercond_mapping = {'po': 1, 'fa': 2, 'ta': 3, 'gd': 4, 'ex': 5}
    _bsmtqual_mapping = {'none': 1, 'po': 2, 'fa': 3, 'ta': 4, 'gd': 5, 'ex': 6}
    _bsmtcond_mapping = {'none': 1, 'po': 2, 'fa': 3, 'ta': 4, 'gd': 5, 'ex': 6}
    _bsmtexposure_mapping = {'none': 1, 'no': 2, 'mn': 3, 'av': 4, 'gd': 5}
    _bsmtfintype1_mapping = {'none': 1, 'unf': 2, 'lwq': 3, 'rec': 4, 'blq': 5, 'alq': 6, 'glq': 7}
    _bsmtfintype2_mapping = {'none': 1, 'unf': 2, 'lwq': 3, 'rec': 4, 'blq': 5, 'alq': 6, 'glq': 7}
    _heatingqc_mapping = {'po': 1, 'fa': 2, 'ta': 3, 'gd': 4, 'ex': 5}
    _kitchenqual_mapping = {'po': 1, 'fa': 2, 'ta': 3, 'gd': 4, 'ex': 5}
    _functional_mapping = {'sal': 8, 'sev': 7, 'maj2': 6, 'maj1': 5, 'mod': 4, 'min2': 3, 'min1': 2, 'typ': 1}
    _fireplacequ_mapping = {'none': 1, 'po': 2, 'fa': 3, 'ta': 4, 'gd': 5, 'ex': 6}
    _garagefinish_mapping = {'none': 1, 'unf': 2, 'rfn': 3, 'fin': 4}
    _garagequal_mapping = {'none': 1, 'po': 2, 'fa': 3, 'ta': 4, 'gd': 5, 'ex': 6}
    _garagecond_mapping = {'none': 1, 'po': 2, 'fa': 3, 'ta': 4, 'gd': 5, 'ex': 6}
    _paveddrive_mapping = {'n': 1, 'p': 2, 'y': 3}
    _poolqc_mapping = {'none': 1, 'fa': 2, 'ta': 3, 'gd': 4, 'ex': 5}

    # mapping list (can add additional mappings via init)
    _mappings_list = [_alley_mapping, _utilities_mapping, _landslope_mapping, _lotshape_mapping,
                      _landcontour_mapping, _exterqual_mapping, _extercond_mapping, _bsmtqual_mapping,
                      _bsmtcond_mapping, _bsmtexposure_mapping, _bsmtfintype1_mapping, _bsmtfintype2_mapping,
                      _heatingqc_mapping, _kitchenqual_mapping, _functional_mapping, _fireplacequ_mapping,
                      _garagefinish_mapping, _garagequal_mapping, _garagecond_mapping, _paveddrive_mapping,
                      _poolqc_mapping]

    # list of ordinal features (can add additional features via init)
    _ordinal_feats = ["Alley", "Utilities", "LandSlope", "LotShape", "LandContour",
                      "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
                      "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional",
                      "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC"]

    # list of nominal features (can add additional features via init)
    _nominal_feats = ["HouseStyle", "Street", "MSZoning", "LotConfig", "Neighborhood", "Condition1",
                      "Condition2", "BldgType", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                      "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical", "GarageType",
                      "Fence", 'MiscFeature', 'SaleType', 'SaleCondition', 'MSSubClass', 'MoSold', 'YrSold']

    def __init__(self, extra_feats=None):
        self.ordinal_feats = CatTransformer._ordinal_feats
        self.nominal_feats = CatTransformer._nominal_feats
        self.mappings_list = CatTransformer._mappings_list

        if extra_feats is not None:
            if type(extra_feats) != list:
                raise TypeError('extra_feats must be a list, got {} instead'.format(type(extra_feats)))
            # Add extra categories to be mapped

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Make copy of dataframe
        X_copy = X.copy()

        # Convert the extra categorical features to string
        for col in ['MSSubClass', 'MoSold', 'YrSold']:
            X_copy[col] = X_copy[col].astype(str)

        # Get dataframe of ordinal and nominal categorical features
        ordinal = X_copy[self.ordinal_feats].copy()
        nominal = X_copy[self.nominal_feats].copy()

        # Keep numerical features separate from categorical features
        X_copy.drop(columns=self.ordinal_feats, inplace=True)
        X_copy.drop(columns=self.nominal_feats, inplace=True)

        # One hot encode nominal categorical features
        nominal = pd.get_dummies(nominal, drop_first=True)

        # map ordinal categorical features
        for feat, mapping in zip(self.ordinal_feats, self.mappings_list):
            ordinal[feat] = ordinal[feat].apply(lambda x: x.lower())
            ordinal[feat] = ordinal[feat].map(mapping)

        if 'OverallQual' not in self.ordinal_feats:
            self.ordinal_feats.append('OverallQual')
        if 'OverallCond' not in self.ordinal_feats:
            self.ordinal_feats.append('OverallCond')

        X_copy = pd.concat([X_copy, ordinal, nominal], axis=1, join='inner')
        self.all_feature_names = X_copy.columns.tolist()

        return X_copy
