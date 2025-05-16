"""         F L I G H T   T I C K E T   P R I C E
                   ... vytvoril Bc.Lukas Patrnciak a Bc.Andrej Tomcik
                   Semestralny projekt na predmet "Metody klasifikacie a rozhodovania"
                   xpatrnciak@stuba.sk, xtomcik@stuba.sk
"""

# KNIZNICE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from collections import Counter
from dataclasses import dataclass


# TREE
@dataclass
class TreeNode:
    feature_index: int = None
    threshold: float = None
    left: 'TreeNode' = None
    right: 'TreeNode' = None
    value: any = None  # hodnota triedy v listovom uzle


# DECISION TREE
class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=None, random_state=None):
        self.max_depth = max_depth
        self.tree = None
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        self.tree = self._build_tree(x, y, depth=0)

    def _build_tree(self, x, y, depth):
        if len(set(y)) == 1:
            return TreeNode(value=y[0])

        if self.max_depth is not None and depth >= self.max_depth:
            most_common = Counter(y).most_common(1)[0][0]
            return TreeNode(value=most_common)

        best_feature, best_threshold, best_entropy, best_sets = None, None, float('inf'), None

        for feature_index in range(x.shape[1]):
            thresholds = np.unique(x[:, feature_index])
            self.rng.shuffle(thresholds)  # náhodné poradie thresholdov pre deterministické správanie

            for threshold in thresholds:
                left_mask = x[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                entropy = self._entropy_index(y[left_mask], y[right_mask])
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature_index
                    best_threshold = threshold
                    best_sets = (x[left_mask], y[left_mask], x[right_mask], y[right_mask])

        if best_sets is None:
            most_common = Counter(y).most_common(1)[0][0]
            return TreeNode(value=most_common)

        left_subtree = self._build_tree(best_sets[0], best_sets[1], depth + 1)
        right_subtree = self._build_tree(best_sets[2], best_sets[3], depth + 1)

        return TreeNode(feature_index=best_feature, threshold=best_threshold,
                        left=left_subtree, right=right_subtree)

    @staticmethod
    def _entropy_index(left_y, right_y):
        def entropy(y):
            count = Counter(y)
            total = len(y)
            return -sum((freq / total) * np.log2(freq / total) for freq in count.values() if freq > 0)

        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * entropy(left_y) + (len(right_y) / n) * entropy(right_y)

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        return [self._predict_row(row, self.tree) for row in x]

    def score(self, x, y):
        predictions = self.predict(x)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        correct = sum(pred == true for pred, true in zip(predictions, y))
        return correct / len(y)

    def visualize_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree

        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}Leaf: Predict {node.value}")
        else:
            print(f"{indent}Feature {node.feature_index} <= {node.threshold}")
            self.visualize_tree(node.left, depth + 1)
            self.visualize_tree(node.right, depth + 1)


# RANDOM FOREST
class CustomRandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.features_idx = []
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, x, y):
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Input x must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise ValueError("Input y must be a pandas Series.")

        self.trees = []
        self.features_idx = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(len(x), len(x), replace=True)
            n_features = int(np.sqrt(x.shape[1]))
            features = np.random.choice(x.columns, n_features, replace=False)

            x_sample = x.iloc[indices][features]
            y_sample = y.iloc[indices]

            tree = CustomDecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x_sample, y_sample)

            self.trees.append(tree)
            self.features_idx.append(features)

    def predict(self, x):
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Input x must be a pandas DataFrame.")

        predictions = []
        for tree, features in zip(self.trees, self.features_idx):
            preds = tree.predict(x[features])
            predictions.append(preds)

        predictions = np.array(predictions).T
        final_preds = [Counter(row).most_common(1)[0][0] for row in predictions]
        return final_preds


# FUNKCIE
def remove_outliers(data):
    cleaned_data = data.copy()
    for col in cleaned_data.select_dtypes(include=[np.number]).columns:
        Q1 = cleaned_data[col].quantile(0.25)
        Q3 = cleaned_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
    return cleaned_data

def median_replace(data, column):
    data[column] = pd.to_numeric(data[column], errors='coerce')
    median_value = data[column].median()
    data[column] = data[column].fillna(median_value)
    return data

def correlation_matrix(data, title_text):
    matrix = data.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title(title_text)
    plt.show()
    return matrix

def evaluate_classifier(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    print(f"\n{model_name} performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n\n", classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def decision_tree(x_train, x_test, y_train, y_test, depth):
    model = CustomDecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(x_train, y_train)
    evaluate_classifier(model, x_test, y_test, 'Decision Tree Classifier')
    return model

def random_forest(x_train, x_test, y_train, y_test, n_est):
    model = CustomRandomForestClassifier(n_estimators=n_est, random_state=42)
    model.fit(x_train, y_train)
    evaluate_classifier(model, x_test, y_test, 'Random Forest Classifier')
    return model

def export_predictions(model, x_test, y_test, output_path):
    predictions = model.predict(x_test)
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })
    results_df.to_csv(output_path, index=False)
    print(f"Random Forest predictions exported to {output_path}")


# SPRACOVANIE DAT
file_path = "dataset_flights.csv"
flight_data = pd.read_csv(file_path)

null_values = flight_data.isnull().sum().sum()
duplicates = flight_data.duplicated().sum()
samples = flight_data.shape[0]

print("ORIGINAL DATA STATS:\nmissing values:", null_values, "\nduplicates:", duplicates, "\nsamples:", samples, "\n")

numerical_set = ['ID', 'duration', 'days_left', 'price']

for i in numerical_set:
    flight_data = median_replace(flight_data, i)

flight_data = flight_data.drop(columns=['ID', 'flight'], errors='ignore')

flight_data = flight_data.drop_duplicates()
flight_data = flight_data.dropna()
flight_data = remove_outliers(flight_data)

null_values = flight_data.isnull().sum().sum()
duplicates = flight_data.duplicated().sum()
samples = flight_data.shape[0]

print("CLEANED DATA STATS:\nmissing values:", null_values, "\nduplicates:", duplicates, "\nsamples:", samples)


# EDA ANALYZA DAT
plt.figure(figsize=(8, 6))
sns.violinplot(data=flight_data, hue='class', y='price', palette="coolwarm")
plt.title('Price vs Class (Violin plot)')
plt.xlabel('Class')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=flight_data, x='duration', y='price', alpha=0.7, color='green')
plt.title('Effect of Duration on Ticket Price')
plt.xlabel('Duration (Hours)')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=flight_data, hue='airline', y='price', errorbar=None, palette="viridis")
plt.title('Average Price by Airline')
plt.xlabel('Airline')
plt.ylabel('Average Price')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=flight_data, hue='stops', y='duration', palette="muted")
plt.title('Effect of Stops on Flight Duration')
plt.xlabel('Number of Stops')
plt.ylabel('Duration (Hours)')
plt.show()

plt.figure(figsize=(8, 5))
sns.regplot(data=flight_data, x='days_left', y='price', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Relationship Between Days Left and Price')
plt.xlabel('Days Left Until Departure')
plt.ylabel('Price')
plt.show()


# DISKRETIZACIA CENY
flight_data['price_class'] = pd.qcut(flight_data['price'], q=3, labels=['low', 'medium', 'high'])
print(flight_data.head())


# KODOVANIE
target_encoder = ce.TargetEncoder(cols=['source_city', 'destination_city', 'airline'])
flight_data[['source_city', 'destination_city', 'airline']] = target_encoder.fit_transform(
    flight_data[['source_city', 'destination_city', 'airline']], flight_data['price'])

onehot_columns = ['class', 'departure_time', 'arrival_time']
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_encoded = onehot_encoder.fit_transform(flight_data[onehot_columns])
onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_columns)
onehot_dataframe = pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=flight_data.index)

label_encoder = LabelEncoder()
flight_data['stops'] = label_encoder.fit_transform(flight_data['stops'])

flight_data = pd.concat([flight_data.drop(columns=onehot_columns), onehot_dataframe], axis=1)


# PRIPRAVA DAT PRE KLASIFIKACIU
X = flight_data.drop(columns=['price', 'price_class'])
Y = flight_data['price_class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# MODELY
decision_tree_model = decision_tree(X_train, X_test, Y_train, Y_test, depth=10)
random_forest_model = random_forest(X_train, X_test, Y_train, Y_test, n_est=100)
export_predictions(random_forest_model, X_test, Y_test, "predictions.csv")

decision_tree_model.visualize_tree(depth=5)


# KORELACIA
data_correlation_matrix = correlation_matrix(flight_data, "Correlation Matrix")
features_correlation = data_correlation_matrix['price'].drop(labels='price')
selected_features_correlation = features_correlation[features_correlation.abs() > 0.1].index.tolist()
print("\nFiltered Features by Correlation:\n", selected_features_correlation)

flight_data_correlation = flight_data[selected_features_correlation]
X_train_correlation, X_test_correlation = train_test_split(flight_data_correlation, test_size=0.3, random_state=42)
random_forest(X_train_correlation, X_test_correlation, Y_train, Y_test, n_est=100)