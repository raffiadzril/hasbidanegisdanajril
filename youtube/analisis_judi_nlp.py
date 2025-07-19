import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

class JudiOnlineDetector:
    def __init__(self):
        self.pipeline = None
        self.gambling_keywords = [
            # Kata kunci judi online dalam bahasa Indonesia
            'slot', 'gacor', 'maxwin', 'scatter', 'bonus', 'deposit', 
            'withdraw', 'saldo', 'jackpot', 'rtp', 'pragmatic', 'pg soft',
            'gates of olympus', 'sweet bonanza', 'starlight princess',
            'situs judi', 'bandar', 'togel', 'bola online', 'casino',
            'poker online', 'domino', 'capsa', 'blackjack', 'roulette',
            'betting', 'taruhan', 'menang besar', 'profit', 'untung',
            'modal', 'invest', 'prediksi', 'bocoran', 'pola',
            'admin', 'cs', 'customer service', 'daftar', 'register',
            'link alternatif', 'promo', 'event', 'turnover',
            'slot gacor', 'slot online', 'judi online', 'situs slot'
        ]
        
        # Download NLTK requirements
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """Preprocessing teks untuk analisis NLP"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
        
        return text
    
    def extract_features(self, text):
        """Extract features untuk deteksi judi online"""
        features = {}
        
        # Feature 1: Jumlah kata kunci judi yang ditemukan
        gambling_count = sum(1 for keyword in self.gambling_keywords if keyword in text.lower())
        features['gambling_keywords_count'] = gambling_count
        
        # Feature 2: Apakah ada kata kunci judi
        features['has_gambling_keywords'] = gambling_count > 0
        
        # Feature 3: Sentiment analysis
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Feature 4: Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Feature 5: Presence of common promotional words
        promo_words = ['gratis', 'free', 'bonus', 'promo', 'diskon', 'cashback']
        features['promo_words_count'] = sum(1 for word in promo_words if word in text.lower())
        
        # Feature 6: Presence of contact info patterns
        contact_patterns = [
            r'\b\d{10,}\b',  # Phone numbers
            r'wa\s*\d+',     # WhatsApp
            r'dm\s*me',      # Direct message
            r'contact\s*me'  # Contact me
        ]
        features['has_contact_info'] = any(re.search(pattern, text.lower()) for pattern in contact_patterns)
        
        return features
    
    def create_training_data(self, comments_df):
        """Membuat data training berdasarkan kata kunci dan analisis manual"""
        training_data = []
        gambling_count = 0
        
        for _, row in comments_df.iterrows():
            comment = str(row['Comment']) if 'Comment' in row else str(row['comment'])
            processed_text = self.preprocess_text(comment)
            
            # Rule-based labeling untuk training data
            features = self.extract_features(processed_text)
            
            # Label sebagai judi online jika:
            # 1. Ada kata kunci judi + kondisi tambahan
            # 2. Ada kombinasi kata kunci tertentu
            # 3. Ada pola khas promosi judi
            
            is_gambling = False
            
            # Lebih permissive untuk mendapatkan lebih banyak contoh gambling
            if features['gambling_keywords_count'] >= 1:
                # Check for strong indicators
                strong_indicators = ['slot gacor', 'situs judi', 'judi online', 'deposit pulsa',
                                   'gates olympus', 'sweet bonanza', 'pragmatic play', 'maxwin',
                                   'scatter', 'bonus new member', 'rtp tinggi']
                
                if any(indicator in processed_text for indicator in strong_indicators):
                    is_gambling = True
                elif features['gambling_keywords_count'] >= 2:
                    is_gambling = True
                elif features['has_contact_info'] and features['gambling_keywords_count'] >= 1:
                    is_gambling = True
                elif features['promo_words_count'] >= 2 and features['gambling_keywords_count'] >= 1:
                    is_gambling = True
            
            # Additional patterns for gambling detection
            gambling_patterns = [
                r'slot\s+gacor', r'deposit\s+\d+', r'withdraw\s+\d+', 
                r'bonus\s+\d+', r'rtp\s+\d+', r'maxwin\s+\d+',
                r'wa\s*\d+', r'dm\s*untuk', r'link\s*alternatif'
            ]
            
            if any(re.search(pattern, processed_text.lower()) for pattern in gambling_patterns):
                is_gambling = True
            
            if is_gambling:
                gambling_count += 1
            
            training_data.append({
                'text': processed_text,
                'original_text': comment,
                'is_gambling': is_gambling,
                **features
            })
        
        df = pd.DataFrame(training_data)
        
        # Ensure we have enough examples of both classes
        gambling_samples = df[df['is_gambling'] == True]
        non_gambling_samples = df[df['is_gambling'] == False]
        
        print(f"Initial distribution - Gambling: {len(gambling_samples)}, Non-gambling: {len(non_gambling_samples)}")
        
        # If we have very few gambling samples, create synthetic ones or adjust threshold
        if len(gambling_samples) < 5:
            print("⚠️ Terlalu sedikit contoh gambling, menurunkan threshold deteksi...")
            # Lower the threshold to get more gambling examples
            for idx, row in df.iterrows():
                if not row['is_gambling'] and row['gambling_keywords_count'] >= 1:
                    df.at[idx, 'is_gambling'] = True
                    gambling_count += 1
                    if gambling_count >= 5:  # Stop when we have enough
                        break
        
        # If still not enough, use any comment with gambling keywords
        if len(df[df['is_gambling'] == True]) < 3:
            print("⚠️ Masih kurang contoh gambling, menggunakan semua komentar dengan kata kunci...")
            for idx, row in df.iterrows():
                if row['gambling_keywords_count'] > 0:
                    df.at[idx, 'is_gambling'] = True
        
        return df
    
    def train_model(self, training_df):
        """Train model NLP untuk deteksi judi online"""
        # Prepare features and labels
        X = training_df['text']
        y = training_df['is_gambling']
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,  # Reduced min_df to handle small datasets
                max_df=0.95
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Check if we have enough samples for train-test split
        min_class_size = class_counts.min()
        
        if min_class_size < 2 or len(training_df) < 10:
            # If we have very few samples, train on all data
            print("⚠️ Dataset terlalu kecil untuk train-test split. Training pada semua data.")
            self.pipeline.fit(X, y)
            
            # Evaluate on training data (not ideal but necessary for small datasets)
            y_pred = self.pipeline.predict(X)
            
            print("=== Model Performance (on training data) ===")
            print(classification_report(y, y_pred, zero_division=0))
            
            return X, y, y_pred
        else:
            # Normal train-test split
            # Use stratify only if both classes have at least 2 samples
            if min_class_size >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Train model
            self.pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.pipeline.predict(X_test)
            
            print("=== Model Performance ===")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            return X_test, y_test, y_pred
    
    def predict_gambling_comments(self, comments_df):
        """Prediksi komentar judi online"""
        if self.pipeline is None:
            raise ValueError("Model belum di-train. Jalankan train_model() terlebih dahulu.")
        
        results = []
        
        for idx, row in comments_df.iterrows():
            comment = str(row['Comment']) if 'Comment' in row else str(row['comment'])
            processed_text = self.preprocess_text(comment)
            
            # Predict using trained model
            prediction = self.pipeline.predict([processed_text])[0]
            confidence = self.pipeline.predict_proba([processed_text])[0].max()
            
            # Extract additional features
            features = self.extract_features(processed_text)
            
            results.append({
                'index': idx,
                'author': row.get('Author', row.get('author', 'Unknown')),
                'original_comment': comment,
                'processed_comment': processed_text,
                'is_gambling_prediction': prediction,
                'confidence': confidence,
                'gambling_keywords_found': features['gambling_keywords_count'],
                'has_contact_info': features['has_contact_info'],
                'sentiment_polarity': features['sentiment_polarity']
            })
        
        return pd.DataFrame(results)
    
    def analyze_gambling_patterns(self, results_df):
        """Analisis pola komentar judi online"""
        gambling_comments = results_df[results_df['is_gambling_prediction'] == True]
        
        print(f"\n=== ANALISIS KOMENTAR JUDI ONLINE ===")
        print(f"Total komentar: {len(results_df)}")
        print(f"Komentar teridentifikasi judi online: {len(gambling_comments)}")
        print(f"Persentase: {len(gambling_comments)/len(results_df)*100:.2f}%")
        
        if len(gambling_comments) > 0:
            print(f"\nConfidence score rata-rata: {gambling_comments['confidence'].mean():.3f}")
            print(f"Rata-rata kata kunci judi per komentar: {gambling_comments['gambling_keywords_found'].mean():.2f}")
            print(f"Komentar dengan info kontak: {gambling_comments['has_contact_info'].sum()}")
        
        return gambling_comments
    
    def create_visualizations(self, results_df, gambling_comments):
        """Membuat visualisasi hasil analisis"""
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Distribution of gambling vs non-gambling comments
            gambling_counts = results_df['is_gambling_prediction'].value_counts()
            labels = ['Non-Gambling', 'Gambling'] if False in gambling_counts.index else ['Gambling']
            values = [gambling_counts.get(False, 0), gambling_counts.get(True, 0)]
            values = [v for v in values if v > 0]  # Remove zero values
            
            if len(values) > 0:
                axes[0, 0].pie(values, labels=labels[:len(values)], 
                              autopct='%1.1f%%', startangle=90)
            else:
                axes[0, 0].text(0.5, 0.5, 'No data to display', ha='center', va='center')
            axes[0, 0].set_title('Distribusi Komentar Judi vs Non-Judi')
            
            # 2. Confidence distribution
            if len(results_df) > 0:
                axes[0, 1].hist(results_df['confidence'], bins=min(20, len(results_df)), 
                               alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Distribusi Confidence Score Prediksi')
                axes[0, 1].set_xlabel('Confidence Score')
                axes[0, 1].set_ylabel('Frequency')
            else:
                axes[0, 1].text(0.5, 0.5, 'No data to display', ha='center', va='center')
            
            # 3. Gambling keywords count distribution
            if len(gambling_comments) > 0:
                axes[1, 0].hist(gambling_comments['gambling_keywords_found'], 
                               bins=min(10, len(gambling_comments)), alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Distribusi Jumlah Kata Kunci Judi dalam Komentar Judi')
                axes[1, 0].set_xlabel('Jumlah Kata Kunci')
                axes[1, 0].set_ylabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'No gambling comments found', ha='center', va='center')
            
            # 4. Sentiment analysis
            if len(gambling_comments) > 0:
                axes[1, 1].scatter(gambling_comments['sentiment_polarity'], 
                                 gambling_comments['confidence'], alpha=0.6)
                axes[1, 1].set_title('Sentiment vs Confidence (Komentar Judi)')
                axes[1, 1].set_xlabel('Sentiment Polarity')
                axes[1, 1].set_ylabel('Confidence Score')
            else:
                axes[1, 1].text(0.5, 0.5, 'No gambling comments found', ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig('analisis_judi_online.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Word cloud for gambling comments
            if len(gambling_comments) > 0:
                gambling_text = ' '.join(gambling_comments['processed_comment'])
                if gambling_text.strip():
                    try:
                        wordcloud = WordCloud(width=800, height=400, 
                                            background_color='white',
                                            colormap='Reds').generate(gambling_text)
                        
                        plt.figure(figsize=(12, 6))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title('Word Cloud - Komentar Judi Online', fontsize=16)
                        plt.savefig('wordcloud_judi.png', dpi=300, bbox_inches='tight')
                        plt.show()
                    except Exception as e:
                        print(f"⚠️ Tidak bisa membuat word cloud: {e}")
            else:
                print("⚠️ Tidak ada komentar judi untuk membuat word cloud")
                
        except Exception as e:
            print(f"⚠️ Error dalam membuat visualisasi: {e}")
            print("Melanjutkan tanpa visualisasi...")

def main():
    """Fungsi utama untuk menjalankan analisis"""
    print("🎯 Memulai Analisis Komentar Judi Online dengan NLP...")
    
    # Load data
    try:
        df = pd.read_csv('komentar_youtube-adajudol.csv')
        print(f"✅ Berhasil load {len(df)} komentar dari CSV")
    except FileNotFoundError:
        print("❌ File komentar_youtube.csv tidak ditemukan!")
        return
    
    # Initialize detector
    detector = JudiOnlineDetector()
    
    # Create training data
    print("\n🔄 Membuat training data...")
    training_df = detector.create_training_data(df)
    
    print(f"Training data created:")
    print(f"- Total samples: {len(training_df)}")
    print(f"- Gambling samples: {training_df['is_gambling'].sum()}")
    print(f"- Non-gambling samples: {(~training_df['is_gambling']).sum()}")
    
    # Train model
    print("\n🤖 Training model NLP...")
    X_test, y_test, y_pred = detector.train_model(training_df)
    
    # Predict on all comments
    print("\n🔍 Melakukan prediksi pada semua komentar...")
    results = detector.predict_gambling_comments(df)
    
    # Analyze results
    gambling_comments = detector.analyze_gambling_patterns(results)
    
    # Save results
    results.to_csv('hasil_analisis_judi.csv', index=False, encoding='utf-8')
    gambling_comments.to_csv('komentar_judi_terdeteksi.csv', index=False, encoding='utf-8')
    
    print(f"\n💾 Hasil disimpan ke:")
    print(f"- hasil_analisis_judi.csv (semua hasil)")
    print(f"- komentar_judi_terdeteksi.csv (hanya komentar judi)")
    
    # Create visualizations
    print("\n📊 Membuat visualisasi...")
    detector.create_visualizations(results, gambling_comments)
    
    # Show sample gambling comments
    if len(gambling_comments) > 0:
        print("\n🎰 CONTOH KOMENTAR JUDI TERDETEKSI:")
        print("="*50)
        for _, row in gambling_comments.head(10).iterrows():
            print(f"Author: {row['author']}")
            print(f"Comment: {row['original_comment'][:100]}...")
            print(f"Confidence: {row['confidence']:.3f}")
            print(f"Keywords found: {row['gambling_keywords_found']}")
            print("-" * 30)
    
    print("\n✅ Analisis selesai!")

if __name__ == "__main__":
    main()
