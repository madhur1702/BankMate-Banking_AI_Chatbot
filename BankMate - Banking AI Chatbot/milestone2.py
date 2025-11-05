#!/usr/bin/env python3
# ======================
# milestone2.py
# Milestone 2: Response Handling & Dialogue Flow
# Builds on Milestone 1 (milestone.py)
# ======================

import pandas as pd
import ast
import re
import spacy
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random

# ======================
# PART 1: LOAD MILESTONE 1 MODELS
# ======================
print("=" * 60)
print("📦 MILESTONE 2: Response Handling & Dialogue Flow")
print("=" * 60)

# Load data (same as Milestone 1)
print("\n📥 Loading dataset...")
data = pd.read_csv("bank_chatbot_dataset_large.csv")
print(f"✅ Dataset loaded: {len(data)} records")

# Helper function to parse entities (from Milestone 1)
def parse_entities(entity_str):
    """Parse entities from JSON-like or custom key:value strings."""
    if pd.isnull(entity_str):
        return {}
    try:
        return ast.literal_eval(entity_str)
    except (SyntaxError, ValueError):
        entities = {}
        for part in str(entity_str).split("|"):
            if ":" in part:
                key, value = part.split(":", 1)
                entities[key.strip()] = value.strip()
        return entities

# Prepare data (same as Milestone 1)
data["entities"] = data["entities"].apply(parse_entities)
X = data["query"]
y = data["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train intent classifier (from Milestone 1)
print("⚙️  Training intent classifier...")
clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logreg", LogisticRegression(max_iter=1000))
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Intent Classifier Accuracy: {accuracy * 100:.2f}%")

# Load spaCy NER model (from Milestone 1)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """Extract entities using spaCy and custom rules (from Milestone 1)."""
    doc = nlp(text)
    spacy_entities = {ent.label_.lower(): ent.text for ent in doc.ents}
    
    account_numbers = re.findall(r'\b\d{5,}\b', text)
    if account_numbers:
        spacy_entities["account_number"] = account_numbers[0]
    
    money_match = re.search(r'\b(\d+)\s*(rupees?|dollars?|rs\.?|inr)?\b', text.lower())
    if money_match:
        spacy_entities["money"] = money_match.group(1)
    
    return spacy_entities

# ======================
# PART 2: RESPONSE TEMPLATES (Rule-Based Responses)
# ======================
print("\n" + "=" * 60)
print("📝 Initializing Response Templates...")
print("=" * 60)

RESPONSE_TEMPLATES = {
    # Account Operations
    "check_balance": [
        "Your current account balance is ₹{amount}.",
        "Account balance: ₹{amount}",
        "You have ₹{amount} in your account.",
    ],
    "statement_request": [
        "Your account statement for {date} has been generated. Please check your email.",
        "Statement requested for {date}. It will be sent to your registered email shortly.",
        "I'm fetching your statement for {date}.",
    ],
    "transfer_money": [
        "Transferring ₹{amount} to {person}. Transaction ID: TXN{txn_id}",
        "✅ ₹{amount} transferred to {person} successfully.",
        "Transfer of ₹{amount} to {person} is in progress.",
    ],
    "withdraw_money": [
        "Withdrawing ₹{amount} from your account. Please visit the nearest ATM.",
        "✅ Withdrawal request for ₹{amount} processed.",
        "Your withdrawal of ₹{amount} has been initiated.",
    ],
    "deposit_money": [
        "Deposit of ₹{amount} recorded in your account.",
        "✅ ₹{amount} deposited successfully.",
        "Your deposit of ₹{amount} is now in your account.",
    ],
    
    # Card Operations
    "block_card": [
        "Your debit card ending with {card_number} has been blocked successfully.",
        "✅ Card blocked. You'll receive a new card in 7-10 business days.",
        "Your card has been blocked as requested.",
    ],
    "card_inquiry": [
        "Your card status: Active. Credit limit: ₹{limit}",
        "Recent transactions on your card are available.",
        "Card is working fine. No issues detected.",
    ],
    "change_pin": [
        "Your PIN has been changed successfully. ✅",
        "✅ New PIN activated. Please remember it securely.",
        "PIN change request processed.",
    ],
    
    # Bill Payments
    "pay_bill": [
        "Bill payment of ₹{amount} for {bill_type} processed successfully. Reference: {ref_id}",
        "✅ ₹{amount} paid for {bill_type}. Payment confirmed.",
        "Your {bill_type} bill of ₹{amount} has been paid.",
    ],
    
    # Loan & Interest
    "loan_inquiry": [
        "Current loan types: Personal, Home, Car, Education. Interest rates range from 7-12%.",
        "For {loan_type} loan information, please visit our website or visit a branch.",
        "Loan details: Interest rate {rate}%, Tenure options available.",
    ],
    "interest_rate": [
        "Current FD rates: 5.5-6.5% depending on tenure.",
        "Savings account interest rate: 3.5% per annum.",
        "Interest rates vary based on account type.",
    ],
    
    # Customer Service
    "close_account": [
        "To close your account, please visit a branch with required documents.",
        "Account closure typically takes 5-7 business days.",
        "We'd like to know why you want to close your account. Please visit us.",
    ],
    "kyc_update": [
        "KYC update request received. Please provide required documents.",
        "✅ Your KYC is complete and verified.",
        "KYC documents uploaded successfully.",
    ],
    "open_account": [
        "You can open an account online or visit your nearest branch.",
        "Account opening takes 10-15 minutes online.",
        "Different account types available: Savings, Salary, Joint, NRI.",
    ],
    "branch_locator": [
        "Nearest branch to {location}: XYZ Branch at ABC Street, {location}.",
        "Branches available in {location}. Operating hours: 9 AM - 5 PM.",
        "Visit our website for complete branch locator.",
    ],
    "request_chequebook": [
        "Chequebook request received. It will be delivered in 3-5 business days.",
        "✅ Chequebook ordered. You'll receive it shortly.",
        "New chequebook is on its way to your registered address.",
    ],
    
    # Default
    "greeting": [
        "Hello! Welcome to Banking AI Chatbot. How can I help you today?",
        "Hi there! I'm here to assist with your banking needs.",
        "Welcome! What can I do for you?",
    ],
    "goodbye": [
        "Thank you for banking with us. Goodbye!",
        "Have a great day! Thank you for using our service.",
        "Goodbye! Feel free to reach out anytime.",
    ],
}

# ======================
# PART 3: DIALOGUE STATE MANAGEMENT
# ======================
class DialogueState:
    """Manages conversation context and state."""
    
    def __init__(self):
        self.conversation_history = []
        self.current_intent = None
        self.extracted_entities = {}
        self.user_profile = {}
        self.confidence_score = 0.0
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self):
        """Generate unique session ID."""
        return f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def update_context(self, query, intent, entities, confidence):
        """Update dialogue context."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": query,
            "intent": intent,
            "entities": entities,
            "confidence": confidence
        })
        self.current_intent = intent
        self.extracted_entities = entities
        self.confidence_score = confidence
    
    def get_history(self):
        """Return conversation history."""
        return self.conversation_history
    
    def reset(self):
        """Reset conversation state."""
        self.conversation_history = []
        self.current_intent = None
        self.extracted_entities = {}
        self.confidence_score = 0.0

# ======================
# PART 4: FALLBACK & CHITCHAT HANDLER
# ======================
class FallbackChitchatHandler:
    """Handles fallback responses and chitchat."""
    
    def __init__(self):
        self.chitchat_keywords = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            "goodbye": ["bye", "goodbye", "see you", "thank you", "thanks", "end chat"],
            "joke": ["joke", "funny", "laugh", "humor"],
            "weather": ["weather", "rain", "sunny", "temperature"],
            "help": ["help", "assistance", "support", "issue"],
            "out_of_scope": ["play music", "open youtube", "translate", "what's the weather"],
        }
        
        self.chitchat_responses = {
            "greeting": [
                "Hello! Welcome to our Banking Chatbot. How can I assist you with your banking needs?",
                "Hi! I'm here to help. What banking service do you need?",
                "Good day! What can I do for you today?",
            ],
            "goodbye": [
                "Thank you for using our service. Have a great day!",
                "Goodbye! Feel free to reach out anytime.",
                "See you soon! Thanks for banking with us.",
            ],
            "joke": [
                "Why did the banker go to the river? To check the current account! 😄",
                "What's a banker's favorite type of music? Loan Ranger theme! 🎵",
                "Why don't banks ever get lonely? Because they always have interest! 💰",
            ],
            "weather": [
                "I'm a banking chatbot, so weather forecasting isn't my expertise! 😊",
                "You might want to check a weather app for that. I'm here for banking help!",
                "That's outside my scope, but I can help with your banking queries!",
            ],
            "help": [
                "Of course! I can help you with account balance, transfers, bill payments, and more.",
                "I'm here to assist! What banking issue can I help resolve?",
                "I'm ready to help. What's your concern?",
            ],
            "out_of_scope": [
                "I appreciate your interest, but that's outside my capabilities. I'm a banking chatbot!",
                "That's not something I can do. Is there anything banking-related I can help with?",
                "I specialize in banking services. How can I assist you with your accounts?",
            ],
        }
    
    def detect_chitchat(self, query: str) -> Optional[str]:
        """Detect if query is chitchat."""
        query_lower = query.lower()
        
        for category, keywords in self.chitchat_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return category
        
        return None
    
    def get_chitchat_response(self, category: str) -> str:
        """Get chitchat response."""
        if category in self.chitchat_responses:
            return random.choice(self.chitchat_responses[category])
        return "I'm here to help with banking services. How can I assist?"
    
    def get_fallback_response(self, confidence: float) -> str:
        """Generate fallback response based on confidence."""
        if confidence < 0.3:
            return "I didn't quite understand that. Could you rephrase your question? I'm here to help with banking services."
        elif confidence < 0.6:
            return "I'm not fully confident in my understanding. Could you provide more details about what you need?"
        else:
            return "I'm having trouble with that request. Please contact our support team or visit a branch for assistance."

# ======================
# PART 5: RESPONSE GENERATOR
# ======================
class BankingResponseGenerator:
    """Generates contextual responses based on intent and entities."""
    
    def __init__(self, templates: Dict):
        self.templates = templates
        self.fallback_handler = FallbackChitchatHandler()
        self.txn_counter = 1000
    
    def generate_txn_id(self):
        """Generate transaction ID."""
        self.txn_counter += 1
        return self.txn_counter
    
    def generate_response(self, intent: str, entities: Dict, confidence: float, query: str = "") -> Tuple[str, bool]:
        """Generate response for given intent and entities."""
        
        # Check for chitchat first
        chitchat_category = self.fallback_handler.detect_chitchat(query)
        if chitchat_category:
            return self.fallback_handler.get_chitchat_response(chitchat_category), True
        
        # Check confidence threshold
        if confidence < 0.5:
            return self.fallback_handler.get_fallback_response(confidence), False
        
        # Generate template-based response
        if intent in self.templates:
            template = random.choice(self.templates[intent])
            
            # Fill in entity placeholders
            response_params = {
                "amount": entities.get("money", "₹0"),
                "person": entities.get("person", "account"),
                "date": entities.get("date", "today"),
                "bill_type": entities.get("bill_type", "bill"),
                "account_number": entities.get("account_number", "****"),
                "location": entities.get("location", "your area"),
                "card_number": entities.get("card_number", "****"),
                "loan_type": entities.get("loan_type", "personal"),
                "rate": "7-12%",
                "limit": "₹100,000",
                "txn_id": self.generate_txn_id(),
                "ref_id": f"REF{self.generate_txn_id()}",
            }
            
            try:
                response = template.format(**response_params)
                return response, True
            except KeyError:
                return template, True
        
        # Default fallback
        return "I can help you with that. Please provide more details.", False

# ======================
# PART 6: CONVERSATION MANAGER
# ======================
class ConversationManager:
    """Manages complete conversation flow."""
    
    def __init__(self, intent_model, entity_extractor, response_generator):
        self.intent_model = intent_model
        self.entity_extractor = entity_extractor
        self.response_generator = response_generator
        self.dialogue_state = DialogueState()
        self.conversation_log = []
    
    def process_query(self, user_query: str) -> Dict:
        """Process user query and generate response."""
        
        # Get intent prediction
        intent = self.intent_model.predict([user_query])[0]
        confidence = self.intent_model.predict_proba([user_query]).max()
        
        # Extract entities
        entities = self.entity_extractor(user_query)
        
        # Update dialogue state
        self.dialogue_state.update_context(user_query, intent, entities, confidence)
        
        # Generate response
        response, success = self.response_generator.generate_response(intent, entities, confidence, user_query)
        
        # Log conversation
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "intent": intent,
            "confidence": f"{confidence:.2%}",
            "entities": entities,
            "response": response,
            "success": success,
        }
        self.conversation_log.append(log_entry)
        
        return {
            "user_query": user_query,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "response": response,
            "success": success,
        }
    
    def get_conversation_log(self) -> List[Dict]:
        """Return conversation log."""
        return self.conversation_log
    
    def export_log(self, filename: str = "conversation_log.json"):
        """Export conversation log to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)
        print(f"✅ Conversation log exported to {filename}")

# ======================
# PART 7: TEST CONVERSATIONS
# ======================
def run_test_conversations():
    """Run sample conversations to test the system."""
    
    print("\n" + "=" * 60)
    print("🧪 Running Test Conversations...")
    print("=" * 60)
    
    # Initialize components
    response_gen = BankingResponseGenerator(RESPONSE_TEMPLATES)
    conv_manager = ConversationManager(clf, extract_entities, response_gen)
    
    # Sample test conversations
    test_conversations = [
        # Account Operations
        {
            "category": "Account Operations",
            "queries": [
                "Show my account balance",
                "Send 500 rupees to Teja",
                "I want to withdraw 1000",
                "Get my bank statement",
            ]
        },
        # Card Operations
        {
            "category": "Card Operations",
            "queries": [
                "Block my debit card",
                "Change my PIN",
                "Card not working",
            ]
        },
        # Bill Payments
        {
            "category": "Bill Payments",
            "queries": [
                "Pay electricity bill 500",
                "Pay my water bill",
                "Mobile bill payment 999",
            ]
        },
        # Customer Service
        {
            "category": "Customer Service",
            "queries": [
                "How do I complete KYC?",
                "Where is nearest branch?",
                "How can I open an account?",
                "Request new chequebook",
            ]
        },
        # Chitchat & Fallback
        {
            "category": "Chitchat & Fallback",
            "queries": [
                "Hello",
                "Tell me a joke",
                "What's the weather?",
                "Thank you",
                "Goodbye",
            ]
        },
    ]
    
    # Process conversations
    all_results = []
    for conv_category in test_conversations:
        print(f"\n📌 {conv_category['category']}")
        print("-" * 60)
        
        for query in conv_category['queries']:
            result = conv_manager.process_query(query)
            all_results.append(result)
            
            print(f"\n👤 User: {result['user_query']}")
            print(f"🤖 Intent: {result['intent']} (Confidence: {result['confidence']:.2%})")
            print(f"🏷️  Entities: {result['entities']}")
            print(f"💬 Response: {result['response']}")
            print(f"✓ Success: {'✅' if result['success'] else '❌'}")
    
    # Export logs
    print("\n" + "=" * 60)
    conv_manager.export_log("milestone2_conversation_log.json")
    
    # Summary Statistics
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    total_queries = len(all_results)
    successful = sum(1 for r in all_results if r['success'])
    avg_confidence = sum(r['confidence'] for r in all_results) / total_queries
    
    print(f"Total Conversations: {total_queries}")
    print(f"Successful Responses: {successful}/{total_queries} ({successful/total_queries*100:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.2%}")
    
    return conv_manager

# ======================
# PART 8: INTERACTIVE CHATBOT
# ======================
def run_interactive_chatbot():
    """Run interactive chatbot session."""
    
    print("\n" + "=" * 60)
    print("💬 Interactive Banking Chatbot")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end conversation")
    print("Type 'history' to see conversation history")
    print("Type 'export' to export logs\n")
    
    # Initialize components
    response_gen = BankingResponseGenerator(RESPONSE_TEMPLATES)
    conv_manager = ConversationManager(clf, extract_entities, response_gen)
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("🤖 Thank you for banking with us. Goodbye!")
            break
        elif user_input.lower() == 'history':
            print("\n📜 Conversation History:")
            for entry in conv_manager.conversation_log:
                print(f"  - {entry['user_query']} → {entry['intent']}")
        elif user_input.lower() == 'export':
            conv_manager.export_log()
        elif user_input:
            result = conv_manager.process_query(user_input)
            print(f"\n🤖 Bot: {result['response']}")
            print(f"   [Intent: {result['intent']}, Confidence: {result['confidence']:.2%}]")

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("\n")
    
    # Run automated test conversations
    conv_manager = run_test_conversations()
    
    # Optionally run interactive mode
    print("\n" + "=" * 60)
    response = input("\n🎯 Would you like to start interactive chatbot? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        run_interactive_chatbot()
    
    print("\n✅ Milestone 2 Completed Successfully!")
    print("📁 Logs saved to: milestone2_conversation_log.json")
    print("=" * 60)