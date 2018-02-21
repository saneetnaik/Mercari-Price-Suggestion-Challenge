import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

print('\n Reading Training data from train.tsv file')
print('\n ......................................')
train = pd.read_table('./inputCopy/train.tsv')
test = pd.read_table('./inputCopy/test.tsv')

df = pd.concat([train, test], 0)
nrow_train = train.shape[0]

print('\n Splitting Category name into general category, sub category 1 and 2')
print('\n ......................................')
df['general_cat'], df['subcat_1'], df['subcat_2'] = \
zip(*df['category_name'].apply(lambda x: split_cat(x)))
#print(train.head())
# print("There are %d unique categories." % train['general_cat'].nunique())
# print("There are %d unique first sub-categories." % train['subcat_1'].nunique())
# print("There are %d unique second sub-categories." % train['subcat_2'].nunique())
print('\n Applying log transformation on the price')
print('\n ......................................')
y_train = np.log1p(train['price'])

print('\n Filling empty values of brand_name with missing, item_description with None, converting shipping and item_condition_id to strings to handle them with a count vectorizer too.')
print('\n ......................................')
#train['category_name'] = train['category_name'].fillna('Other').astype(str)
df['brand_name'] = df['brand_name'].fillna('missing').astype(str)
df['shipping'] = df['shipping'].astype(str)
df['item_condition_id'] = df['item_condition_id'].astype(str)
df['item_description'] = df['item_description'].fillna('None')
# we need a custom pre-processor to extract correct field,
# but want to also use default scikit-learn preprocessing (e.g. lowercasing)
default_preprocessor = CountVectorizer().build_preprocessor()

def build_preprocessor(field):
    field_idx = list(df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

print('\n Using feature union to combine all the features from all the columns')
print('\n ......................................')
vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        lowercase=True,
        preprocessor=build_preprocessor('name'))),
    ('general_cat', CountVectorizer(
        token_pattern='.+',
        lowercase=True,
        max_features=11,
        preprocessor=build_preprocessor('general_cat'))),
    ('subcat_1', CountVectorizer(
            token_pattern='.+',
            max_features=112,
            preprocessor=build_preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(
            token_pattern='.+',
            max_features=697,
            preprocessor=build_preprocessor('subcat_2'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        stop_words="english",
        preprocessor=build_preprocessor('item_description'))),
])
X_train = vectorizer.fit_transform(df.values)
print('\n (rows, feature size)', X_train.shape)
#print(vectorizer.get_feature_names())

def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

# print('\n we have used k fold cross validator, to check how good our model is. We have 10 splits of data, so 9 parts will be used for building the model and 1 part for validation,this will be executed k times')
# print('\n ......................................')
# cv = KFold(n_splits=10, shuffle=True, random_state=42)
# for train_ids, valid_ids in cv.split(X_train[:nrow_train]):
#     #model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#     model = Ridge(
#         solver='auto',
#         fit_intercept=True,
#         alpha=0.5,
#         max_iter=1000,
#         normalize=False,
#         tol=0.05)
#     model.fit(X_train[train_ids], y_train[train_ids])
#     y_pred_valid = model.predict(X_train[valid_ids])
#     rmsle = get_rmsle(y_pred_valid, y_train[valid_ids])
#     print('Valid Root Mean Squared Logarithmic Error:', rmsle)
#     break

print("Using Ridge Regression model")
model = Ridge(
        solver='auto',
        fit_intercept=True,
        alpha=0.5,
        max_iter=1000,
        normalize=False,
        tol=0.05)
print("Fitting Model")
model.fit(X_train[:nrow_train], y_train)
X_test = X_train[nrow_train:]
print("Predicting Price")
preds = model.predict(X_test)
test["price"] = np.expm1(preds)
print("Saving data in submission_mps_ridgeregression.csv file")
test[["test_id", "price"]].to_csv("submission_mps_ridgeregression.csv", index = False)