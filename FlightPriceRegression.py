"""         F L I G H T   T I C K E T   P R I C E
                   ... vytvoril Lukas Patrnciak a Andrej Tomcik
                   Semestralny projekt na predmet "Metody klasifikacie a rozhodovania"
"""

# KNIZNICE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import category_encoders as ce


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


def evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"\n{model_name} performance:")
    print(f"Training -> MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, R2: {r2_train:.2f}")
    print(f"Testing  -> MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R2: {r2_test:.2f}")

    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Residuals Analysis", fontsize=16)

    ax[0].scatter(y_train_pred, residuals_train, color="blue", alpha=0.5)
    ax[0].axhline(y=0, color="black", linestyle="--")
    ax[0].set_title(f"{model_name} - Training Residuals")
    ax[0].set_xlabel("Predicated Values")
    ax[0].set_ylabel("Residuals")

    ax[1].scatter(y_test_pred, residuals_test, color="red", alpha=0.5)
    ax[1].axhline(y=0, color="black", linestyle="--")
    ax[1].set_title(f"{model_name} - Test Residuals")
    ax[1].set_xlabel("Predicated Values")
    ax[1].set_ylabel("Residuals")

    plt.show()


def decision_tree(x_train, x_test, y_train, y_test, depth, rand_state):
    model = DecisionTreeRegressor(max_depth=depth, random_state=rand_state)
    model.fit(x_train, y_train)
    evaluate_model(model, x_train, y_train, x_test, y_test, 'Decision Tree')

    return model


def random_forest(x_train, x_test, y_train, y_test, n_est, rand_state):
    model = RandomForestRegressor(n_estimators=n_est, random_state=rand_state)
    model.fit(x_train, y_train)
    evaluate_model(model, x_train, y_train, x_test, y_test, 'Random Forest')

    return model


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


# KODOVANIE
target_encoder = ce.TargetEncoder(cols=['source_city', 'destination_city', 'airline'])
flight_data[['source_city', 'destination_city', 'airline']] = target_encoder.fit_transform(flight_data[['source_city', 'destination_city', 'airline']], flight_data['price'])

onehot_columns = ['class', 'departure_time', 'arrival_time']
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_encoded = onehot_encoder.fit_transform(flight_data[onehot_columns])
onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_columns)
onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=flight_data.index)

label_encoder = LabelEncoder()
flight_data['stops'] = label_encoder.fit_transform(flight_data['stops'])

flight_data = pd.concat([flight_data.drop(columns=onehot_columns), onehot_df], axis=1)

print("\nSTRING CULUMNS ENCODED")


# ZOBRAZENIE SCATTERPLOT PRE DATA
fig_original_data = plt.figure(figsize=(12, 8))
ax_original_data = fig_original_data.add_subplot(111, projection='3d')
scatter = ax_original_data.scatter(
    flight_data['days_left'],
    flight_data['duration'],
    flight_data['stops'],
    c=flight_data['price'],
    cmap='viridis',
    alpha=0.7
)
ax_original_data.set_xlabel('days_left')
ax_original_data.set_ylabel('duration')
ax_original_data.set_zlabel('stops')
plt.colorbar(scatter, ax=ax_original_data, label='Price')
plt.title("3D Scatter Plot of Original Features")
plt.show()


# ROZDELENIE DAT
X = flight_data.drop('price', axis=1)
Y = flight_data['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# ROZHODOVACI STROM
decision_tree_model = decision_tree(X_train, X_test, Y_train, Y_test, 100, 42)

plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model, filled=True, feature_names=X.columns)
plt.title("Decision Tree")
plt.show()


# RANDOM FOREST
random_forest_model = random_forest(X_train, X_test, Y_train, Y_test, 100, 42)

# Filtrovanie priznakov vzhladom na korelaciu
data_correlation_matrix = correlation_matrix(flight_data, "Correlation Matrix")
features_correlation = data_correlation_matrix['price'].drop(labels='price')
selected_features_correlation = features_correlation[features_correlation.abs() > 0.1].index.tolist()

print("\nFiltered Features by Correlation:\n", selected_features_correlation)

flight_data_correlation = flight_data[selected_features_correlation]
X_train_correlation, X_test_correlation = train_test_split(flight_data_correlation, test_size=0.3, random_state=42)
random_forest(X_train_correlation, X_test_correlation, Y_train, Y_test, 100, 42)

# Filtrovanie priznakov vzhladom na dolezitost
feature_importances = random_forest_model.feature_importances_
selected_features_importance = feature_importances.argsort()[-7:]

plt.figure(figsize=(10, 8))
plt.barh(X.columns[selected_features_importance], feature_importances[selected_features_importance])
plt.xlabel("Feature Importance")
plt.title("Top Feature Importances in Random Forest")
plt.show()

print("\nFiltered Features by Importance:\n", X.columns[selected_features_importance])

flight_data_importance = flight_data[X.columns[selected_features_importance]]
X_train_importance, X_test_importance = train_test_split(flight_data_importance, test_size=0.3, random_state=42)
random_forest(X_train_importance, X_test_importance, Y_train, Y_test, 100, 42)