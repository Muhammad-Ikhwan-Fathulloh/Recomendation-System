from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertModel

# Data contoh
X_train = [["ulasan yang sangat bagus"], ["ulasan yang sangat buruk"]]
y_train = [1, 0]

# SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenisasi dengan BERT
inputs = tokenizer(X_resampled, return_tensors="pt", padding=True, truncation=True)
print(inputs)

# Model BERT
bert_model = BertModel.from_pretrained('bert-base-uncased')
outputs = bert_model(**inputs)
print(outputs)
