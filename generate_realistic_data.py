import pandas as pd
import numpy as np

np.random.seed(42)

# Generate complex realistic data with significant overlap and noise
data = []

# Total: 10000 samples with distribution: E-commerce(3200), Blog(3000), News(2400), Personal(1400)

# E-commerce (3200 samples) - varied characteristics with outliers
for i in range(3200):
    # Add some outliers and edge cases
    if i % 20 == 0:  # 5% outliers
        word_count = np.random.randint(500, 3000)  # Wide range
        bounce_rate = round(np.random.uniform(20, 60), 1)
    else:
        word_count = np.random.randint(800, 2200)
        bounce_rate = round(np.random.uniform(28, 48), 1)
    
    data.append({
        'url': f'ecommerce{i+1}.com',
        'word_count': word_count,
        'num_links': np.random.randint(30, 85),
        'has_contact_form': np.random.choice([0, 1], p=[0.25, 0.75]),
        'meta_description_length': np.random.randint(135, 165),
        'h1_count': np.random.randint(1, 4),
        'h2_count': np.random.randint(2, 10),
        'image_count': np.random.randint(6, 28),
        'script_count': np.random.randint(4, 15),
        'css_count': np.random.randint(2, 7),
        'page_load_time': round(np.random.uniform(1.5, 4.0), 1),
        'bounce_rate': bounce_rate,
        'avg_session_duration': round(np.random.uniform(140, 300), 1),
        'social_media_links': np.random.randint(2, 8),
        'external_links': np.random.randint(10, 32),
        'internal_links': np.random.randint(20, 60),
        'text_to_html_ratio': round(np.random.uniform(0.52, 0.75), 2),
        'category': 'E-commerce'
    })

# Blog (3000 samples) - significant overlap with E-commerce and News
for i in range(3000):
    # Add variability
    if i % 15 == 0:  # Some blogs are more like news
        word_count = np.random.randint(1200, 2200)
        num_links = np.random.randint(40, 70)
    else:
        word_count = np.random.randint(550, 1500)
        num_links = np.random.randint(18, 50)
    
    data.append({
        'url': f'blog{i+1}.com',
        'word_count': word_count,
        'num_links': num_links,
        'has_contact_form': np.random.choice([0, 1], p=[0.65, 0.35]),
        'meta_description_length': np.random.randint(130, 160),
        'h1_count': np.random.randint(1, 3),
        'h2_count': np.random.randint(2, 8),
        'image_count': np.random.randint(3, 15),
        'script_count': np.random.randint(2, 10),
        'css_count': np.random.randint(1, 5),
        'page_load_time': round(np.random.uniform(1.0, 3.0), 1),
        'bounce_rate': round(np.random.uniform(32, 58), 1),
        'avg_session_duration': round(np.random.uniform(100, 200), 1),
        'social_media_links': np.random.randint(2, 6),
        'external_links': np.random.randint(7, 22),
        'internal_links': np.random.randint(12, 35),
        'text_to_html_ratio': round(np.random.uniform(0.45, 0.65), 2),
        'category': 'Blog'
    })

# News (2400 samples) - highest content, overlaps with E-commerce and Blog
for i in range(2400):
    # Add variability - some news sites are lighter
    if i % 12 == 0:  # Lighter news articles
        word_count = np.random.randint(900, 1600)
        image_count = np.random.randint(8, 20)
    else:
        word_count = np.random.randint(1400, 2800)
        image_count = np.random.randint(12, 38)
    
    data.append({
        'url': f'news{i+1}.com',
        'word_count': word_count,
        'num_links': np.random.randint(55, 100),
        'has_contact_form': np.random.choice([0, 1], p=[0.35, 0.65]),
        'meta_description_length': np.random.randint(145, 170),
        'h1_count': np.random.randint(1, 4),
        'h2_count': np.random.randint(5, 14),
        'image_count': image_count,
        'script_count': np.random.randint(7, 20),
        'css_count': np.random.randint(3, 9),
        'page_load_time': round(np.random.uniform(2.2, 5.0), 1),
        'bounce_rate': round(np.random.uniform(18, 42), 1),
        'avg_session_duration': round(np.random.uniform(180, 350), 1),
        'social_media_links': np.random.randint(4, 10),
        'external_links': np.random.randint(18, 40),
        'internal_links': np.random.randint(35, 70),
        'text_to_html_ratio': round(np.random.uniform(0.58, 0.80), 2),
        'category': 'News'
    })

# Personal (1400 samples) - smallest but with high variance
for i in range(1400):
    # High variance - some personal sites are more developed
    if i % 10 == 0:  # More developed personal sites
        word_count = np.random.randint(700, 1300)
        num_links = np.random.randint(20, 45)
    else:
        word_count = np.random.randint(250, 900)
        num_links = np.random.randint(8, 35)
    
    data.append({
        'url': f'personal{i+1}.com',
        'word_count': word_count,
        'num_links': num_links,
        'has_contact_form': np.random.choice([0, 1], p=[0.55, 0.45]),
        'meta_description_length': np.random.randint(115, 150),
        'h1_count': np.random.randint(1, 3),
        'h2_count': np.random.randint(1, 5),
        'image_count': np.random.randint(1, 10),
        'script_count': np.random.randint(1, 6),
        'css_count': np.random.randint(1, 4),
        'page_load_time': round(np.random.uniform(0.7, 2.5), 1),
        'bounce_rate': round(np.random.uniform(40, 75), 1),
        'avg_session_duration': round(np.random.uniform(50, 150), 1),
        'social_media_links': np.random.randint(0, 5),
        'external_links': np.random.randint(4, 18),
        'internal_links': np.random.randint(4, 25),
        'text_to_html_ratio': round(np.random.uniform(0.38, 0.62), 2),
        'category': 'Personal'
    })

# Shuffle the data to mix categories
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('Web_Document_Dataset.csv', index=False)
print(f"Complex realistic dataset created with {len(df)} samples")
print("\nCategory distribution:")
print(df['category'].value_counts())
print("\nDataset characteristics:")
print(f"- Multiple overlapping feature ranges")
print(f"- Outliers and edge cases included (~5%)")
print(f"- High variance within categories")
print(f"- Shuffled for random distribution")
print("\nDataset saved to Web_Document_Dataset.csv")
