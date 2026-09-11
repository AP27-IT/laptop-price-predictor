import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('/home/claude/proj/laptop_data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# Ram / Weight cleanup
df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')

# Touchscreen / IPS
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# PPI
new = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
df['X_res'] = new[0].astype('int')
df['Y_res'] = new[1].astype('int')
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype('float')

df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# Cpu brand
def fetch_processor(text):
    text = ' '.join(text.split()[0:3])
    if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
        return text
    elif text.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'

df['Cpu Name'] = df['Cpu'].apply(fetch_processor)
df['Cpu brand'] = df['Cpu Name']
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

# Memory -> HDD / SSD
df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)
new = df['Memory'].str.split('+', n=1, expand=True)

df['first'] = new[0].str.strip()
df['second'] = new[1]

df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)
df['first'] = df['first'].str.replace(r'\D', '', regex=True)

df['second'] = df['second'].fillna('0')
df['Layer2HDD'] = df['second'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer2SSD'] = df['second'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)
df['second'] = df['second'].str.replace(r'\D', '', regex=True)

df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

df['HDD'] = (df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD'])
df['SSD'] = (df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD'])
df['Hybrid'] = (df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid'])
df['Flash_Storage'] = (df['first'] * df['Layer1Flash_Storage'] + df['second'] * df['Layer2Flash_Storage'])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                  'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                  'Layer2Flash_Storage'], inplace=True)
df.drop(columns=['Memory'], inplace=True)
df.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True)  # negligible / near-zero variance

# Gpu brand
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df = df[df['Gpu brand'] != 'ARM']  # drop the single ARM outlier row
df.drop(columns=['Gpu'], inplace=True)

# OS simplification
def cat_os(inp):
    if inp in ('Windows 10', 'Windows 7', 'Windows 10 S'):
        return 'Windows'
    elif inp in ('macOS', 'Mac OS X'):
        return 'Mac'
    else:
        return 'Other/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'], inplace=True)

# ---- Save the reference dataframe used to populate the UI dropdowns ----
# (this is the pre-encoding frame the Streamlit app reads unique() values from)
df_for_ui = df.drop(columns=['Price']).copy()

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

cat_cols_idx = [0, 1, 7, 10, 11]  # Company, TypeName, Cpu brand, Gpu brand, os
print("Columns:", list(X.columns))

step1 = ColumnTransformer(
    transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), cat_cols_idx)
    ],
    remainder='passthrough'
)

estimators = [
    ('rf', RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5,
                                  max_features=0.75, max_depth=15)),
    ('dt', DecisionTreeRegressor(max_depth=8, random_state=3)),
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=10))

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("R2 score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))

with open('/home/claude/build/df.pkl', 'wb') as f:
    pickle.dump(df_for_ui, f)

with open('/home/claude/build/pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Saved df.pkl and pipe.pkl")
print(df_for_ui.dtypes)
print(df_for_ui.head(2))
