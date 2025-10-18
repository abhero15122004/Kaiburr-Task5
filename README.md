# Kaiburr-Task5
Code for the Data Science Task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data (safe-guard: wrap in try so repeated runs don't error)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ComplaintClassifier:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def load_and_prepare_data(self, file_path, sample_size=100000):
        """Load data with optional sampling for faster processing"""
        print("Loading data...")
        try:
            self.df = pd.read_csv(file_path)
            print(f"Full dataset loaded: {len(self.df)} records")
        except MemoryError:
            print("Memory error - using sampling...")
            self.df = pd.read_csv(file_path, nrows=sample_size)
            print(f"Sampled dataset loaded: {len(self.df)} records")
        except FileNotFoundError:
            raise
        except Exception as e:
            # fallback: try reading a small sample to at least continue
            print(f"Error reading file ({e}). Attempting to read first {sample_size} rows.")
            self.df = pd.read_csv(file_path, nrows=sample_size)
            print(f"Sampled dataset loaded: {len(self.df)} records")

        print(f"Dataset shape: {self.df.shape}")
        print("Columns:", self.df.columns.tolist())

    def explore_data(self):
        """Perform initial data exploration"""
        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        if self.df is None:
            print("Dataframe is empty. Load data first.")
            return False

        print("Dataset Info:")
        print(f"Total records: {len(self.df)}")
        print(f"Columns: {self.df.columns.tolist()}")

        required_cols = ['Consumer complaint narrative', 'Product']
        if not all(col in self.df.columns for col in required_cols):
            print("Warning: Required columns not found. Available columns:", self.df.columns.tolist())
            return False

        return True

    def map_categories(self):
        """Map product categories to our target classes"""
        print("\n=== MAPPING CATEGORIES ===\n")
        print("Original Product Distribution (top 10):")
        print(self.df['Product'].value_counts().head(10))

        category_mapping = {
            # Category 0: Credit reporting, repair, or other personal consumer reports
            'Credit reporting, credit repair services, or other personal consumer reports': 0,
            'Credit reporting': 0,

            # Category 1: Debt collection
            'Debt collection': 1,

            # Category 2: Consumer Loan (including cards/loans)
            'Payday loan': 2,
            'Vehicle loan or lease': 2,
            'Student loan': 2,
            'Consumer Loan': 2,
            'Credit card': 2,
            'Prepaid card': 2,

            # Category 3: Mortgage
            'Mortgage': 3,
            'Home equity loan or line of credit (HELOC)': 3
        }

        # Map using .map then drop unmapped
        self.df['target'] = self.df['Product'].map(category_mapping)
        original_size = len(self.df)
        self.df = self.df.dropna(subset=['target'])
        self.df['target'] = self.df['target'].astype(int)
        print(f"Removed {original_size - len(self.df)} unmapped records")
        print("Final category distribution:")
        print(self.df['target'].value_counts())

    def handle_missing_data(self):
        """Handle missing values in the narrative column"""
        print("\n=== HANDLING MISSING DATA ===\n")
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Consumer complaint narrative'])
        print(f"Removed {initial_count - len(self.df)} records with missing narratives")
        print(f"Remaining records: {len(self.df)}")

    def create_text_features(self):
        """Create basic text features for EDA"""
        print("\n=== CREATING TEXT FEATURES ===\n")
        # Make sure narratives are strings
        self.df['narrative_str'] = self.df['Consumer complaint narrative'].astype(str)
        self.df['word_count'] = self.df['narrative_str'].apply(lambda x: len(x.split()))
        self.df['char_count'] = self.df['narrative_str'].apply(len)
        # avoid division by zero
        self.df['avg_word_length'] = self.df.apply(
            lambda row: (row['char_count'] / row['word_count']) if row['word_count'] > 0 else 0.0, axis=1
        )

        print("Text feature statistics:")
        print(self.df[['word_count', 'char_count', 'avg_word_length']].describe())

    def visualize_data(self):
        """Create visualizations for EDA"""
        print("\n=== CREATING VISUALIZATIONS ===\n")
        category_names = ['Credit Reporting', 'Debt Collection', 'Consumer Loan', 'Mortgage']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Category distribution
        category_counts = self.df['target'].value_counts().reindex(range(4), fill_value=0)
        axes[0, 0].bar(category_names, category_counts.values)
        axes[0, 0].set_title('Distribution of Complaint Categories')
        axes[0, 0].set_ylabel('Number of Complaints')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Word count distribution
        axes[0, 1].hist(self.df['word_count'], bins=50, alpha=0.7)
        axes[0, 1].set_title('Distribution of Word Counts')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')

        # 3. Word count by category (boxplot)
        category_data = [self.df[self.df['target'] == i]['word_count'] for i in range(4)]
        axes[1, 0].boxplot(category_data, labels=category_names)
        axes[1, 0].set_title('Word Count by Category')
        axes[1, 0].set_ylabel('Word Count')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Text length vs category (violin)
        sns.violinplot(x='target', y='word_count', data=self.df, ax=axes[1, 1])
        axes[1, 1].set_title('Text Length Distribution by Category')
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Word Count')
        axes[1, 1].set_xticklabels(category_names)

        plt.tight_layout()
        plt.show()

        # Print some statistics
        print("Category distribution:")
        for i, name in enumerate(category_names):
            count = len(self.df[self.df['target'] == i])
            percentage = (count / len(self.df)) * 100 if len(self.df) > 0 else 0
            print(f"{name}: {count} complaints ({percentage:.2f}%)")

    def preprocess_text(self, text):
        """Preprocess a single narrative"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        # remove explicit placeholder sequences like "xxxx" or "XXXX" (case-insensitive)
        text = re.sub(r'x{2,}', ' ', text, flags=re.IGNORECASE)
        # remove dates like mm/dd/yyyy or dd/mm/yyyy
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', ' ', text)
        # remove other numbers
        text = re.sub(r'\d+', ' ', text)
        # remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # collapse whitespace
        text = ' '.join(text.split())

        # tokenize (simple split is fine here)
        tokens = text.split()
        # remove stopwords and lemmatize; keep tokens length > 2
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def apply_preprocessing(self, sample_limit=50000):
        """Apply preprocessing to all narratives (or a sample)"""
        print("\n=== PREPROCESSING TEXT ===\n")
        print("Preprocessing narratives...")

        if len(self.df) > sample_limit:
            print(f"Sampling {sample_limit} records for faster processing...")
            sample_df = self.df.sample(n=sample_limit, random_state=42).copy()
        else:
            sample_df = self.df.copy()

        sample_df['processed_text'] = sample_df['Consumer complaint narrative'].apply(self.preprocess_text)
        # Drop empty processed texts
        sample_df = sample_df[sample_df['processed_text'].str.len() > 0].copy()
        print(f"After preprocessing: {len(sample_df)} records")

        # show a few examples
        print("\nSample of processed text:")
        n_show = min(3, len(sample_df))
        for i in range(n_show):
            original = sample_df['Consumer complaint narrative'].iloc[i][:200]
            processed = sample_df['processed_text'].iloc[i][:200]
            print(f"\nOriginal: {original}")
            print(f"Processed: {processed}")

        return sample_df

    def train_models(self, df_processed):
        """Train and compare multiple models"""
        print("\n=== MODEL TRAINING ===\n")

        X = df_processed['processed_text']
        y = df_processed['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

        models = {
            'Multinomial Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(random_state=42, max_iter=5000)
        }

        results = {}

        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            try:
                cv_scores = cross_val_score(model, X_train_tfidf, y_train,
                                            cv=cv, scoring='f1_macro', n_jobs=None)
                print(f"Cross-validation F1 scores: {cv_scores}")
                print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"CV failed for {name} ({e}) - continuing to train on full set.")

            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'predictions': y_pred
            }

            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1 Macro: {f1_macro:.4f}")
            print(f"Test F1 Weighted: {f1_weighted:.4f}")

        return results, X_test, y_test, X_train_tfidf, y_train

    def compare_models(self, results):
        """Compare model performance"""
        print("\n=== MODEL COMPARISON ===\n")
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'F1 Macro': result['f1_macro'],
                'F1 Weighted': result['f1_weighted']
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))

        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_macro'])
        best_model = results[best_model_name]['model']

        print(f"\nBest model: {best_model_name}")
        self.model = best_model

        return best_model_name, best_model

    def evaluate_model(self, results, best_model_name, X_test, y_test):
        """Comprehensive model evaluation"""
        print(f"\n=== EVALUATING {best_model_name.upper()} ===\n")

        best_result = results[best_model_name]
        y_pred = best_result['predictions']

        category_names = ['Credit Reporting', 'Debt Collection', 'Consumer Loan', 'Mortgage']
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=category_names))

        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=category_names,
                    yticklabels=category_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

        # Feature importance / top features for linear models
        if hasattr(self.model, 'coef_'):
            print("\nTop features per category:")
            feature_names = self.vectorizer.get_feature_names_out()
            # Some linear models have shape (n_classes, n_features)
            coefs = self.model.coef_
            # If binary, ensure shape compatibility
            if coefs.ndim == 1:
                coefs = np.vstack([-coefs, coefs])
            for i, category in enumerate(category_names):
                idx = i if i < coefs.shape[0] else 0
                indices = np.argsort(coefs[idx])[-10:][::-1]
                top_features = [(feature_names[j], coefs[idx][j]) for j in indices]
                print(f"\n{category}:")
                for feature, score in top_features:
                    print(f"  {feature}: {score:.4f}")
        elif hasattr(self.model, 'feature_log_prob_'):
            # MultinomialNB alternative
            print("\nTop features per category (MultinomialNB):")
            feature_names = self.vectorizer.get_feature_names_out()
            for i, category in enumerate(category_names):
                indices = np.argsort(self.model.feature_log_prob_[i])[-10:][::-1]
                top_features = [(feature_names[j], self.model.feature_log_prob_[i][j]) for j in indices]
                print(f"\n{category}:")
                for feature, score in top_features:
                    print(f"  {feature}: {score:.4f}")

    def predict_new_complaints(self, new_complaints):
        """Make predictions on new complaint narratives"""
        print("\n=== MAKING PREDICTIONS ON NEW COMPLAINTS ===\n")

        if self.model is None or self.vectorizer is None:
            print("Please train the model first!")
            return

        category_names = ['Credit Reporting', 'Debt Collection', 'Consumer Loan', 'Mortgage']

        predictions = []
        for i, complaint in enumerate(new_complaints):
            processed = self.preprocess_text(complaint)
            vectorized = self.vectorizer.transform([processed])

            prediction = self.model.predict(vectorized)[0]

            # Probabilities or decision scores
            top_2 = []
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(vectorized)[0]
                top_2_indices = np.argsort(probabilities)[-2:][::-1]
                top_2 = [(category_names[idx], float(probabilities[idx])) for idx in top_2_indices]
            elif hasattr(self.model, 'decision_function'):
                decisions = self.model.decision_function(vectorized)
                # shape may be (1, n_classes) or (n_classes,)
                decisions = np.array(decisions).flatten()
                top_2_indices = np.argsort(decisions)[-2:][::-1]
                top_2 = [(category_names[int(idx)], float(decisions[int(idx)])) for idx in top_2_indices]

            predictions.append({
                'complaint': (complaint[:100] + "...") if len(complaint) > 100 else complaint,
                'predicted_category': category_names[int(prediction)],
                'category_code': int(prediction),
                'top_2_predictions': top_2
            })

            print(f"Complaint {i+1}:")
            print(f"  Text: {complaint[:200]}...")
            print(f"  Predicted: {category_names[int(prediction)]} (Code: {int(prediction)})")
            print(f"  Top 2 predictions: {top_2}")
            print()

        return predictions


def main():
    classifier = ComplaintClassifier()

    file_path = "complaints.csv"  # Update to your dataset path
    classifier.load_and_prepare_data(file_path, sample_size=100000)

    if not classifier.explore_data():
        return

    classifier.map_categories()
    classifier.handle_missing_data()
    classifier.create_text_features()
    classifier.visualize_data()

    processed_df = classifier.apply_preprocessing(sample_limit=50000)
    if len(processed_df) == 0:
        print("No processed records available after preprocessing. Exiting.")
        return

    results, X_test, y_test, X_train_tfidf, y_train = classifier.train_models(processed_df)
    best_name, best_model = classifier.compare_models(results)
    classifier.evaluate_model(results, best_name, X_test, y_test)

    new_complaints = [
        "I have been trying to resolve an error on my credit report for months. The credit bureau keeps ignoring my disputes and my score is suffering because of their negligence.",
        "A debt collector has been calling me multiple times a day about a debt I don't even owe. They're using abusive language and threatening to sue me if I don't pay immediately.",
        "I applied for a car loan last week and the interest rate they offered me was much higher than advertised. The terms were completely different from what was promised.",
        "My mortgage company lost my payment and is now charging me late fees. I've sent them proof of payment but they refuse to remove the fees from my account."
    ]

    classifier.predict_new_complaints(new_complaints)
    print("=== TASK COMPLETED ===")
    print(f"Best model: {best_name}")

if __name__ == "__main__":
    main()
