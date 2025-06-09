def prepare_features(df):
  features = df[['GrLivArea','BedroomAbvGr','FullBath']]
  target = df['SalePrice']

  features = features.fillna(features.mean())
  return features, target