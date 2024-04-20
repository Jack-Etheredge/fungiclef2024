"""
make a model from metadata only
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

metadata_dir = Path('__file__').parent.absolute().parent / 'metadata'
metadata_file = metadata_dir / "FungiCLEF2023_train_metadata_PRODUCTION.csv"
df = pd.read_csv(metadata_file, dtype={"class_id": "int64"})
y = df['class_id'].values
print(len(y), "training examples")
# test metadata cols (in addition to image_path and observation_id)
# month,day,countryCode,hasCoordinate,Substrate,Latitude,Longitude,Habitat,MetaSubstrate
# TODO: check the val and test for different possible values in the string cols
df = df.drop(columns=['year', 'class_id'])
df = df[['month', 'day', 'countryCode', 'Substrate', 'Latitude', 'Longitude', 'Habitat', 'MetaSubstrate']]
categorical_columns = ['countryCode', 'Substrate', 'Habitat', 'MetaSubstrate']
numerical_columns = ['month', 'day', 'Latitude', 'Longitude']
for col in numerical_columns:
    df[col] = df[col].fillna(-1)
for col in ['month', 'day']:
    df[col] = df[col].astype(int)
column_trans = make_column_transformer((OrdinalEncoder(), categorical_columns), remainder='passthrough')
df = column_trans.fit_transform(df)
# TODO: not sure what to do with hasCoordinate. Do I need to handle cases where coordinates are missing?
X_train, X_test, y_train, y_test = train_test_split(df, y, stratify=y, test_size=0.2)

# model = LogisticRegression()
model = HistGradientBoostingClassifier()
# model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1_score = f1_score(y_test, y_pred, average='macro')
print(f1_score)

# ordinalencoder + lgbm = 0.0004998159976922541
# ordinalencoder + fillna + lgbm = 0.00011526844452806728
# onehotencoder + lgbm = 0.0005765268823302054
