from classifier import CreditCardIntentClassifier

# Test the classifier
classifier = CreditCardIntentClassifier()
classifier.setup()  # This will load Excel data and create embeddings

# Test a sample query
result = classifier.classify_query("My card was declined at the store")
print(result)