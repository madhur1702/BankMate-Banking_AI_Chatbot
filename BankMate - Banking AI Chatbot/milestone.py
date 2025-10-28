# milestone.py
import pandas as pd
import ast
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ======================
# Load & Prepare Data
# ======================
print("📥 Loading dataset...")
data = pd.read_csv("training_and_responses_expanded.csv")

print(f"✅ Dataset loaded successfully! Total records: {len(data)}")
print("📄 Columns in CSV:", data.columns.tolist())

# --- Helper function to safely parse 'entities' ---
def parse_entities(entity_str):
    """Parse entities from either JSON-like or custom key:value|key:value strings."""
    if pd.isnull(entity_str):
        return {}
    try:
        # Try normal literal_eval first (e.g. {'PERSON': 'Teja'})
        return ast.literal_eval(entity_str)
    except (SyntaxError, ValueError):
        # Fallback for custom formats like PERSON:Teja|MONEY:500
        entities = {}
        for part in str(entity_str).split("|"):
            if ":" in part:
                key, value = part.split(":", 1)
                entities[key.strip()] = value.strip()
        return entities

# Apply parsing
print("🔍 Parsing entities...")
data["entities"] = data["entities"].apply(parse_entities)

# Split into features & labels
X = data["text"]
y = data["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Build Intent Classifier
# ======================
print("⚙️  Training intent classifier...")
clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logreg", LogisticRegression(max_iter=1000))
])

clf.fit(X_train, y_train)

# Evaluate model
print("\n=== 📊 Classification Report ===")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))  # <-- Warning fixed
print(f"✅ Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ======================
# Entity Extraction
# ======================
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """Extract named entities using spaCy and custom rules."""
    doc = nlp(text)
    spacy_entities = {ent.label_.lower(): ent.text for ent in doc.ents}

    # Custom rule: Extract account numbers (sequences of digits)
    account_numbers = re.findall(r'\b\d{5,}\b', text)  # Finds 5+ digit numbers
    if account_numbers:
        spacy_entities["account_number"] = account_numbers[0]
    
    # Custom rule: Extract money amounts
    money_match = re.search(r'\b(\d+)\s*(rupees?|dollars?|rs\.?|inr)?\b', text.lower())
    if money_match:
        spacy_entities["money"] = money_match.group(1)

    return spacy_entities

# ======================
# Test Sample
# ======================
if __name__ == "__main__":
    print("\n Running sample query test...")
    sample = "Show me balance of my savings account 12345"
    intent = clf.predict([sample])[0]
    entities = extract_entities(sample)

    print("\n Query:", sample)
    print(" Predicted Intent:", intent)
    print(" Extracted Entities:", entities)
    
    # Additional test cases
    print("\n--- Additional Tests ---")
    test_cases = [
        "send 500 rupees to Teja",
        "my account number is 96182240",
        "transfer 1000 to Sri",
        "how much money do i have"
    ]
    
    for test in test_cases:
        intent = clf.predict([test])[0]
        entities = extract_entities(test)
        print(f"\n Query: {test}")
        print(f" Intent: {intent}")
        print(f" Entities: {entities}")
