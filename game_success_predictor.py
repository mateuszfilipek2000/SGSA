import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MIN_REVIEWS_FOR_SUCCESS = 100
SUCCESS_RATIO_THRESHOLD = 0.6

class SimpleLogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.theta = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        self.theta = np.zeros(n_features + 1)
        
        sample_weights = np.ones(n_samples)
        n_class0, n_class1 = np.sum(y == 0), np.sum(y == 1)
        sample_weights[y == 0] = n_samples / (2.0 * n_class0)
        sample_weights[y == 1] = n_samples / (2.0 * n_class1)

        for i in range(self.n_iter):
            h = self._sigmoid(X_bias.dot(self.theta))
            gradient = (X_bias.T.dot(sample_weights * (h - y))) / n_samples
            self.theta -= self.lr * gradient

    def predict_proba(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(X_bias.dot(self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# PREPROCESSING
df = pd.read_json("games.json", orient="index").reset_index()
df.rename(columns={'index': 'app_id'}, inplace=True)

# Dataset info
print("Rozmiar danych:", df.shape)
print("Kolumny danych:", df.columns)
print("Typy danych:\n", df.dtypes)
print("Brakujące wartości:\n", df.isnull().sum())
print("Przykładowe dane:\n", df.head())

df['positive'] = pd.to_numeric(df['positive'], errors='coerce').fillna(0)
df['negative'] = pd.to_numeric(df['negative'], errors='coerce').fillna(0)
df['total_reviews'] = df['positive'] + df['negative']
df['review_ratio'] = df['positive'] / df['total_reviews'].replace(0, 1)
df['is_successful'] = ((df['total_reviews'] >= 100) & (df['review_ratio'] > 0.6)).astype(int)

df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
df['dlc_count'] = pd.to_numeric(df['dlc_count'], errors='coerce').fillna(0)
df['achievements'] = pd.to_numeric(df['achievements'], errors='coerce').fillna(0)

df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(datetime.now().year)
df['game_age'] = datetime.now().year - df['release_year']
df['windows'] = df['windows'].astype(int)
df['mac'] = df['mac'].astype(int)
df['linux'] = df['linux'].astype(int)

top_genres = df['genres'].explode().value_counts().head(3).index.tolist()
top_categories = df['categories'].explode().value_counts().head(3).index.tolist()

for genre in top_genres:
    df[f'genre_{genre}'] = df['genres'].apply(lambda x: int(genre in x) if isinstance(x, list) else 0)

for cat in top_categories:
    df[f'cat_{cat}'] = df['categories'].apply(lambda x: int(cat in x) if isinstance(x, list) else 0)

# Dataset after preprocessing
print("\n--- Dane po przetworzeniu ---")
print("Rozmiar danych:", df.shape)
print("Brakujące wartości:\n", df.isnull().sum())
print("Przykładowe dane:\n", df.head())

# Drop unnecessary columns
features = [
    'price', 'dlc_count', 'achievements', 'game_age',
    'windows', 'mac', 'linux'
] + [f'genre_{g}' for g in top_genres] + [f'cat_{c}' for c in top_categories]

X = df[features]
y = df['is_successful']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sk_model = LogisticRegression(class_weight='balanced', max_iter=200)
sk_model.fit(X_train_scaled, y_train)
y_pred_sk = sk_model.predict(X_test_scaled)

print("--- Scikit-learn ---")
print(f"Dokładność: {accuracy_score(y_test, y_pred_sk):.4f}")
print(classification_report(y_test, y_pred_sk, target_names=["Nieudana", "Udana"]))

custom_model = SimpleLogisticRegression(lr=0.05, n_iter=2000)
custom_model.fit(X_train_scaled, y_train.values)
y_pred_custom = custom_model.predict(X_test_scaled)

print("\n--- Prosta regresja ---")
print(f"Dokładność: {accuracy_score(y_test, y_pred_custom):.4f}")
print(classification_report(y_test, y_pred_custom, target_names=["Nieudana", "Udana"]))

game = {
    "release_date": "Dec 15, 2024",
    "price": 49.99,
    "dlc_count": 1,
    "achievements": 50,
    "windows": True,
    "mac": True,
    "linux": False,
    "genres": ["Indie", "RPG", "Adventure"],
    "categories": ["Single-player", "RPG"]
}

df_game = pd.DataFrame([game])
df_game['release_year'] = pd.to_datetime(df_game['release_date'], errors='coerce').dt.year.fillna(datetime.now().year)
df_game['game_age'] = datetime.now().year - df_game['release_year']
df_game['price'] = pd.to_numeric(df_game['price'], errors='coerce').fillna(0)
df_game['dlc_count'] = pd.to_numeric(df_game['dlc_count'], errors='coerce').fillna(0)
df_game['achievements'] = pd.to_numeric(df_game['achievements'], errors='coerce').fillna(0)
df_game['windows'] = int(df_game['windows'].iloc[0])
df_game['mac'] = int(df_game['mac'].iloc[0])
df_game['linux'] = int(df_game['linux'].iloc[0])

for genre in top_genres:
    df_game[f'genre_{genre}'] = int(genre in game['genres'])

for cat in top_categories:
    df_game[f'cat_{cat}'] = int(cat in game['categories'])

for col in features:
    if col not in df_game.columns:
        df_game[col] = 0

X_game = scaler.transform(df_game[features])

sk_prob = sk_model.predict_proba(X_game)[0]
sk_pred = sk_model.predict(X_game)[0]

custom_prob_success = custom_model.predict_proba(X_game)[0]
custom_prob = [1 - custom_prob_success, custom_prob_success]
custom_pred = custom_model.predict(X_game)[0]


print("Scikit-learn")
print(f"  Przewidywanie: {'Udana' if sk_pred==1 else 'Nieudana'}")
print(f"  Prawdopodobieństwo sukcesu: {sk_prob[1]:.4f}")
print(f"  Prawdopodobieństwo porażki: {sk_prob[0]:.4f}")

print("\nProsta regresja")
print(f"  Przewidywanie: {'Udana' if custom_pred==1 else 'Nieudana'}")
print(f"  Prawdopodobieństwo sukcesu: {custom_prob[1]:.4f}")
print(f"  Prawdopodobieństwo porażki: {custom_prob[0]:.4f}")

# Wizualizacje
cm_sk = confusion_matrix(y_test, y_pred_sk)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_sk, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Nieudana', 'Udana'],
            yticklabels=['Nieudana', 'Udana'])
plt.title('Macierz pomyłek - Scikit-learn Logistic Regression')
plt.xlabel('Przewidywana etykieta')
plt.ylabel('Prawdziwa etykieta')
plt.savefig('confusion_matrix_sklearn.png')
plt.show()

# Rozkład współczynnika pozytywnych recenzji
plt.figure(figsize=(7, 4))
sns.histplot(df['review_ratio'], bins=30, kde=True, color='skyblue')
plt.title('Rozkład współczynnika pozytywnych recenzji')
plt.xlabel('Współczynnik recenzji (Pozytywne / Wszystkie)')
plt.ylabel('Liczba gier')
plt.savefig('review_ratio_distribution.png')
plt.show()

# Liczba udanych vs. nieudanych gier
plt.figure(figsize=(5, 4))
sns.countplot(x='is_successful', data=df, palette='Set2')
plt.title('Liczba udanych vs. nieudanych gier')
plt.xlabel('Czy udana (1=Tak, 0=Nie)')
plt.ylabel('Liczba gier')
plt.xticks([0, 1], ['Nieudana', 'Udana'])
plt.savefig('success_count.png')
plt.show()

# Macierz korelacji cech
plt.figure(figsize=(10, 8))
corr = df[features + ['is_successful']].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Macierz korelacji cech')
plt.savefig('correlation_matrix.png')
plt.show()

# Rozkład cen gier
plt.figure(figsize=(7, 4))
sns.histplot(df['price'], bins=30, kde=True, color='orange')
plt.title('Rozkład cen gier')
plt.xlabel('Cena (USD)')
plt.ylabel('Liczba gier')
plt.savefig('price_distribution.png')
plt.show()

# Najpopularniejsze gatunki i kategorie
# Top 10 gatunków
all_genres = df['genres'].explode().dropna()
top_genres = all_genres.value_counts().head(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='crest')
plt.title('10 najpopularniejszych gatunków')
plt.xlabel('Liczba gier')
plt.ylabel('Gatunek')
plt.savefig('top10_genres.png')
plt.show()
# Top 10 kategorii
all_categories = df['categories'].explode().dropna()
top_categories = all_categories.value_counts().head(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top_categories.values, y=top_categories.index, palette='flare')
plt.title('10 najpopularniejszych kategorii')
plt.xlabel('Liczba gier')
plt.ylabel('Kategoria')
plt.savefig('top10_categories.png')
plt.show()

# Zależność między ceną a sukcesem
plt.figure(figsize=(7, 4))
sns.boxplot(x='is_successful', y='price', data=df, palette='Set3')
plt.title('Cena vs. sukces gry')
plt.xlabel('Czy udana (0=Nie, 1=Tak)')
plt.ylabel('Cena (USD)')
plt.xticks([0, 1], ['Nieudana', 'Udana'])
plt.savefig('price_vs_success.png')
plt.show()

# Rozkład ocen Metacritic (jeśli dostępne)
if (df['metacritic_score'] > 0).sum() > 0:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[df['metacritic_score'] > 0]['metacritic_score'], bins=20, kde=True, color='purple')
    plt.title('Rozkład ocen Metacritic')
    plt.xlabel('Ocena Metacritic')
    plt.ylabel('Liczba gier')
    plt.savefig('metacritic_score_distribution.png')
    plt.show()

# Wsparcie platform
platform_counts = {'Windows': df['windows'].sum(), 'Mac': df['mac'].sum(), 'Linux': df['linux'].sum()}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(platform_counts.keys()), y=list(platform_counts.values()), palette='pastel')
plt.title('Wsparcie platform')
plt.xlabel('Platforma')
plt.ylabel('Liczba gier')
plt.savefig('platform_support.png')
plt.show()

# Korelacja między czasem gry a sukcesem
if 'average_playtime_forever' in df.columns:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x='average_playtime_forever', y='is_successful', data=df, alpha=0.3)
    plt.title('Średni czas gry vs. sukces')
    plt.xlabel('Średni czas gry (minuty)')
    plt.ylabel('Czy udana (0=Nie, 1=Tak)')
    plt.savefig('playtime_vs_success.png')
    plt.show()
