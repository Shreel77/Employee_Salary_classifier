from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['workclass'] = encoder.fit_transform(df['workclass'])
df['occupation'] = encoder.fit_transform(df['occupation'])
df['age'] = encoder.fit_transform(df['age'])
df['education'] = encoder.fit_transform(df['education'])
df['income'] = encoder.fit_transform(df['income'])
#income encoded first as 0 and 1 then used condition for getting less than or more than 50k
df['hours-per-week'] = encoder.fit_transform(df['hours-per-week'])
df
