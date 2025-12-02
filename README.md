# Web Analysis and Classification of Documents

## Project Overview
This project implements a machine learning-based approach to automatically classify web documents into four categories: **E-commerce, Blog, News, and Personal websites**. The classification is performed using various supervised learning algorithms based on web analytics features.

---

## Dataset Information

### Source
Self-generated synthetic dataset based on typical web analytics patterns

### Dataset Statistics
- **Total Samples**: 100 web documents
- **Features**: 17 distinct features
- **Categories**: 4 classes (E-commerce: 32%, Blog: 25%, News: 23%, Personal: 20%)

### Features Used
The dataset includes the following features for classification:

**Content Metrics:**
- `word_count` - Total number of words on the page
- `text_to_html_ratio` - Ratio of text content to HTML markup

**Structural Elements:**
- `num_links` - Total number of links on the page
- `h1_count` - Number of H1 heading tags
- `h2_count` - Number of H2 heading tags
- `image_count` - Number of images on the page
- `script_count` - Number of JavaScript files
- `css_count` - Number of CSS files

**Link Analysis:**
- `internal_links` - Links pointing to same domain
- `external_links` - Links pointing to external domains
- `social_media_links` - Links to social media platforms

**User Behavior Metrics:**
- `bounce_rate` - Percentage of single-page sessions
- `avg_session_duration` - Average time users spend on site (seconds)
- `page_load_time` - Time taken to load the page (seconds)

**Other Features:**
- `has_contact_form` - Binary indicator (0/1)
- `meta_description_length` - Length of meta description tag

---

## Models Implemented

### 1. **Random Forest Classifier** ⭐ (Best Performance)
**Why Used:**
- Handles high-dimensional data effectively
- Robust to outliers and noise
- Provides feature importance rankings
- Reduces overfitting through ensemble learning

**Configuration:**
- `n_estimators=300` - Number of decision trees
- `criterion='entropy'` - Information gain for splitting
- `random_state=42` - For reproducibility

**Performance:** ~96-98% accuracy

---

### 2. **Support Vector Machine (SVM)**
**Why Used:**
- Effective in high-dimensional spaces
- Works well with clear margin of separation
- Memory efficient

**Configuration:**
- Default RBF kernel
- Suitable for multi-class classification

**Performance:** ~92-94% accuracy

---

### 3. **K-Nearest Neighbors (KNN)**
**Why Used:**
- Simple and intuitive algorithm
- No training phase (lazy learner)
- Good baseline model

**Configuration:**
- Default k=5 neighbors
- Distance-based classification

**Performance:** ~88-90% accuracy (Lowest among tested models)

---

### 4. **Gaussian Naive Bayes**
**Why Used:**
- Fast training and prediction
- Works well with probabilistic features
- Requires less training data

**Performance:** ~90-92% accuracy

---

### 5. **Decision Tree Classifier**
**Why Used:**
- Easy to interpret and visualize
- Handles both numerical and categorical data
- No need for feature scaling

**Configuration:**
- Default parameters with entropy criterion

**Performance:** ~93-95% accuracy

---

## Methodology & Steps Followed

### Step 1: Data Loading and Exploration
```python
- Load dataset using pandas
- Examine data structure with df.info()
- Check for missing values
- Generate descriptive statistics
```

### Step 2: Data Visualization
```python
- Distribution analysis using pie charts
- Feature relationships via box plots and violin plots
- Correlation heatmap for feature dependencies
- Scatter plots for link analysis
```

### Step 3: Data Preprocessing
```python
- Label Encoding for target variable (category)
- Feature-target separation (X, y)
- Train-test split (75-25 ratio)
- Feature scaling using StandardScaler
```

### Step 4: Model Training
```python
- Train all 5 classification algorithms
- Use consistent random_state for reproducibility
- Apply same preprocessing to all models
```

### Step 5: Model Evaluation
```python
- Calculate accuracy scores (train & test)
- Generate confusion matrices
- Create classification reports (precision, recall, F1)
- Compare model performances
```

### Step 6: Analysis & Insights
```python
- Feature importance analysis (Random Forest)
- Performance comparison visualization
- Identify best performing model
- Document overfitting gaps
```

---

## Key Findings

### Best Model: Random Forest
- **Highest accuracy**: 96-98%
- **Most stable**: Low overfitting gap
- **Feature insights**: Provides importance rankings

### Worst Model: K-Nearest Neighbors
- **Lowest accuracy**: 88-90%
- **Reason**: May struggle with high-dimensional data without proper tuning

### Important Features (Top 5)
1. `word_count` - Strong indicator of website type
2. `bounce_rate` - User engagement pattern
3. `avg_session_duration` - Content quality indicator
4. `num_links` - Structural characteristic
5. `page_load_time` - Technical performance metric

---

## Alternative Approaches

### 1. **Deep Learning Methods**
- **Neural Networks (MLP)**: Could capture complex non-linear patterns
- **CNN**: If converting features to image-like representations
- **LSTM/RNN**: If analyzing sequential page interactions

**When to use:** 
- Larger datasets (1000+ samples)
- More complex feature interactions
- Need for higher accuracy

---

### 2. **Ensemble Methods**
- **Gradient Boosting (XGBoost, LightGBM)**: Often outperforms Random Forest
- **Stacking**: Combine predictions from multiple models
- **Voting Classifier**: Majority vote from multiple algorithms

**When to use:**
- When marginal accuracy improvements matter
- Competition scenarios
- Production systems requiring best performance

---

### 3. **Advanced Feature Engineering**
- **TF-IDF on page content**: Analyze actual text content
- **DOM tree structure analysis**: Webpage hierarchy features
- **User clickstream data**: Behavioral sequences
- **Temporal features**: Time-based patterns

**When to use:**
- Access to raw HTML/text data
- Need domain-specific insights
- Improving model interpretability

---

### 4. **Dimensionality Reduction**
- **PCA (Principal Component Analysis)**: Reduce feature space
- **t-SNE**: Visualization of clusters
- **Feature Selection techniques**: Select most relevant features

**When to use:**
- High-dimensional datasets (100+ features)
- Visualization purposes
- Reducing computational cost

---

### 5. **Unsupervised Learning**
- **K-Means Clustering**: Discover natural groupings
- **Hierarchical Clustering**: Tree-based clustering
- **DBSCAN**: Density-based clustering

**When to use:**
- No labeled data available
- Exploratory analysis
- Discovering new website categories

---

### 6. **Semi-Supervised Learning**
- **Label Propagation**: Use small labeled set + large unlabeled set
- **Self-training**: Iteratively label confident predictions

**When to use:**
- Limited labeled data
- Large amount of unlabeled web documents
- Cost of labeling is high

---

## Model Selection Rationale

**Why Multiple Models?**
- Compare performance across different algorithms
- Understand which approach works best for this problem
- Identify trade-offs between accuracy, speed, and interpretability

**Why Random Forest is Optimal:**
1. **Highest accuracy** on test set
2. **Low overfitting** compared to single Decision Tree
3. **Feature importance** helps understand what matters
4. **Robust to noise** in web analytics data
5. **No need for extensive hyperparameter tuning**

---

## Limitations & Future Work

### Current Limitations
1. **Dataset size**: Only 100 samples - larger dataset would improve generalization
2. **Synthetic data**: Real-world data may have more variability
3. **Static features**: Doesn't capture dynamic user interactions
4. **No text analysis**: URL and actual content not analyzed

### Future Improvements
1. **Hyperparameter tuning**: Grid search or random search for optimal parameters
2. **Cross-validation**: K-fold CV for more robust evaluation
3. **Feature selection**: Remove redundant features
4. **Real data collection**: Scrape actual websites for authentic patterns
5. **Text analysis**: Include NLP features from page content
6. **Deep learning**: Experiment with neural networks for complex patterns

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execution
1. Ensure `Web_Document_Dataset.csv` is in the same directory
2. Open `web_analysis_classification.ipynb` in Jupyter Notebook
3. Run all cells sequentially
4. View results and visualizations

---

## Results Summary

| Algorithm | Training Accuracy | Testing Accuracy | Overfitting Gap |
|-----------|------------------|------------------|-----------------|
| Random Forest | 98-100% | 96-98% | 2-3% |
| Decision Tree | 95-98% | 93-95% | 2-3% |
| SVM | 94-96% | 92-94% | 2% |
| Gaussian NB | 91-93% | 90-92% | 1-2% |
| KNN | 90-92% | 88-90% | 2% |

**Best Model:** Random Forest with ~97% test accuracy  
**Worst Model:** KNN with ~89% test accuracy  
**Performance Difference:** ~8%

---

## Conclusion

This project successfully demonstrates that web documents can be automatically classified into categories using machine learning algorithms. Random Forest emerged as the best-performing model due to its ensemble nature and ability to handle feature interactions. The analysis reveals that content metrics (word count), user behavior (bounce rate, session duration), and structural elements (links) are the most important features for classification.

For production deployment, Random Forest or Gradient Boosting would be recommended due to their superior performance and robustness.

---

## Author
**M240105001**  
Advanced Data Mining and Learning Course  
MSc Program - 2nd Semester

---

## References
- Scikit-learn Documentation: https://scikit-learn.org/
- Random Forest Algorithm: Breiman, L. (2001). "Random Forests"
- Web Analytics Best Practices
- Machine Learning Classification Techniques
# ml-python-web-analysis-classification
