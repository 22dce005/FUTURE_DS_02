import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter

# NLTK Setup
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("customer_support_tickets.csv")  # Change to your file path

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# --- Data Overview ---
print("Basic Info:")
print(df.info())
print("\nHead of Data:")
print(df.head())

# --- Descriptive Statistics ---
print("\nDescriptive Stats:")
print(df.describe(include='all'))

# -------------------------------
# 1. CUSTOMER DEMOGRAPHICS
# -------------------------------

# Gender Distribution
if 'Customer Gender' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Customer Gender', hue='Customer Gender', palette='Set2', legend=False)
    plt.title("Customer Gender Distribution")
    plt.show()

# Age Distribution
if 'Customer Age' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['Customer Age'], kde=True, bins=20, color='teal')
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

# -------------------------------
# 2. TICKET TYPE & STATUS
# -------------------------------

if 'Ticket Status' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Ticket Status', hue='Ticket Status', palette='coolwarm', legend=False)
    plt.title("Ticket Status Distribution")
    plt.show()

if 'Ticket Type' in df.columns:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x='Ticket Type', hue='Ticket Type', palette='viridis', legend=False)
    plt.title("Ticket Type Distribution")
    plt.xticks(rotation=45)
    plt.show()

# -------------------------------
# 3. CUSTOMER SATISFACTION
# -------------------------------

if 'Customer Satisfaction Rating' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['Customer Satisfaction Rating'], kde=True, bins=10, color='orange')
    plt.title("Customer Satisfaction Ratings")
    plt.show()

# Boxplot: Satisfaction by Priority
if 'Ticket Priority' in df.columns and 'Customer Satisfaction Rating' in df.columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Ticket Priority', y='Customer Satisfaction Rating', data=df)
    plt.title("Satisfaction by Ticket Priority")
    plt.show()

# -------------------------------
# 4. TIME ANALYSIS
# -------------------------------

if 'First Response Time' in df.columns and 'Customer Satisfaction Rating' in df.columns and 'Ticket Priority' in df.columns:
    plt.figure(figsize=(10,5))
    sns.scatterplot(x='First Response Time', y='Customer Satisfaction Rating', hue='Ticket Priority', data=df)
    plt.title("Response Time vs Satisfaction")
    plt.show()

if 'Time to Resolution' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['Time to Resolution'], kde=True, bins=20, color='purple')
    plt.title("Time to Resolution Distribution")
    plt.show()

# -------------------------------
# 5. TEXT MINING – TICKET DESCRIPTIONS
# -------------------------------

if 'Ticket Description' in df.columns:
    text = ' '.join(df['Ticket Description'].dropna().astype(str))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    word_freq = Counter(tokens)
    common_words = word_freq.most_common(20)

    if common_words:
        words, counts = zip(*common_words)
        plt.figure(figsize=(10,5))
        sns.barplot(x=list(words), y=list(counts), palette='magma')
        plt.title("Most Common Words in Ticket Descriptions")
        plt.xticks(rotation=45)
        plt.show()

# -------------------------------
# 6. CHANNEL & PRIORITY ANALYSIS
# -------------------------------

if 'Ticket Channel' in df.columns:
    plt.figure(figsize=(7,4))
    sns.countplot(data=df, x='Ticket Channel', hue='Ticket Channel', palette='cool', legend=False)
    plt.title("Ticket Channel Usage")
    plt.xticks(rotation=45)
    plt.show()

if 'Ticket Priority' in df.columns:
    plt.figure(figsize=(7,4))
    sns.countplot(data=df, x='Ticket Priority', hue='Ticket Priority', palette='Reds', legend=False)
    plt.title("Ticket Priority Distribution")
    plt.show()

if 'Ticket Channel' in df.columns and 'Ticket Priority' in df.columns:
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='Ticket Channel', hue='Ticket Priority', palette='Accent')
    plt.title("Ticket Channel vs Priority")
    plt.xticks(rotation=45)
    plt.show()
