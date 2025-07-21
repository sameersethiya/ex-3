from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "apple banana apple",
    "apple fruit"
    ]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
features = vectorizer.get_feature_names_out()
matrix_array = tfidf_matrix.toarray()
print("Features:", features)
print("TF-IDF Matrix:")
for i, row in enumerate(matrix_array):
    print(f"Doc{i+1}:", row)