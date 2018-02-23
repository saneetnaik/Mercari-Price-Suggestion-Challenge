# Steps
1.	Analysed the data
2.	Cleaned the data
3.	Extracted the features
4.	Identified ML model
5.	Trained and validated the model 
6.	Tuned the model and feature extraction
7.	Predict the price for test.tsv data using the trained model

# Potential shortcomings
1. Feature extraction is not up to the mark.
2. Compared only linear vs Ridge Regression. 

# Possible improvements
1. We could do more word processing and cleaning up of data. This will be helpful for the feature extraction.
2. We can try other algorithms like RNN(Recurrent neural networks)
3. We could use domain specific stopwords to derive better results, in addition to english stopwords.

## Data analysis/cleaning up the data
A.	Numerical features
•	Price – This is our response that we need to predict
•	Shipping cost
B.	Categorical features
•	Shipping cost: A binary indicator, 1 if shipping fee is paid by the seller and 0 if paid by buyer.
•	item_condition_id: The condition of the items provided by the seller
•	name: The item's name
•	brand_name: The item's producer brand name
•	category_name: The item's single or multiple categories that are separated by "\"
•	item_description: A short description on the item that may include removed words, flagged by [rm]

Price(Target Variable)
The median price of all the items in the training is about \$267 but given the existence of some extreme values of over \$100 and the maximum at \$2,009, the distribution of the variables is heavily skewed to the left. So let's make log-transformation on the price (we added +1 to the value before the transformation to avoid zero and negative values).

Shipping
The shipping cost burden is decently split between sellers and the sellers (55%) pay buyers with more than half of the items’ shipping fees.

Item Category
There are about 1,287 unique categories but among each of them, we will always see a main/general category firstly, followed by two more particular subcategories (e.g. Beauty/Makeup/Face or Lips). In addition, about 6,327 items do not have a category labels. 

Brand name
There are 4809 unique brand names in the training dataset.

Item Description
It will be more challenging to parse through this particular item since it's unstructured data.  We will strip out all punctuations, remove some english stop words (i.e. redundant words such as "a", "the", etc.) and any other words with a length less than 3:


