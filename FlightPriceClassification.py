"""         F L I G H T   T I C K E T   P R I C E
                   ... vytvoril Bc.Lukas Patrnciak a Bc.Andrej Tomcik
                   Semestralny projekt na predmet "Metody klasifikacie a rozhodovania"
                   xpatrnciak@stuba.sk, xtomcik@stuba.sk
"""

# KNIŽNICE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from dataclasses import dataclass


# STROM
@dataclass
class TreeNode:
    """
    Vytvorí základnú štruktúru strom
    """
    attribute: str = None
    children: dict = None
    value: any = None


# ROZHODOVACI STROM
class DecisionTree:
    """
    Vytvorí rozhodovací strom na základe vstupných údajov v konštruktori nižšie
    """
    def __init__(self, max_depth=None, random_state=None, max_features=None):
        """
        Vytvorí konštruktor so zvolenými parametrami
        """
        self.majority_class = None
        self.features = None
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)
        self.tree = None

    def fit(self, x, y):
        """
        Naučí rozhodovací strom zo vstupných údajov (x) a cieľových hodnôt (y).
        """

        # 1. Ak je x typu DataFrame (Pandas), získaj názvy atribútov a konvertuj ho na zoznam slovníkov
        if isinstance(x, pd.DataFrame):
            self.features = x.columns.tolist()  # Zoznam názvov stĺpcov ako atribúty
            x = x.to_dict(orient="records")  # Každý riadok ako slovník {atribút: hodnota} - zoznam slovníkov

        # 2. Ak je y typu Series (napr. pandas stĺpec), konvertuj ho na obyčajný zoznam
        if isinstance(y, pd.Series):
            y = y.tolist()

        # 3. Urči 1 najčastejšiu triedu v dátach – použije sa, keď strom nemá odpoveď
        class_counts = Counter(y)
        self.majority_class = class_counts.most_common(1)[0][0]

        # 4. Vytrénuj strom pomocou rekurzívneho budovania od koreňa (hĺbka = 0)
        self.tree = self._build_tree(
            x, y,
            attributes=self.features,
            depth=0
        )

    def _build_tree(self, x, y, attributes, depth):
        """
        Rekurzívne buduje rozhodovací strom na základe dát x a cieľových hodnôt y.
        """

        # 1. Ak všetky výstupy sú rovnaké, návrat listového uzla s touto hodnotou
        if len(set(y)) == 1:
            return TreeNode(value=y[0])

        # 2. Ak nemáme ďalšie atribúty, alebo sme dosiahli maximálnu hĺbku, vráť najčastejšiu triedu
        if not attributes or (self.max_depth is not None and depth >= self.max_depth):
            most_common_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=most_common_value)

        # 3. Vyber najlepší atribút na rozdelenie podľa informačného zisku
        best_attr = self._best_attribute(x, y, attributes)

        # 4. Vytvor nový vnútorný uzol so zvoleným atribútom
        node = TreeNode(attribute=best_attr, children={})

        # 5. Získaj všetky unikátne (neprázdne) hodnoty atribútu, podľa ktorého sa budeme vetviť
        unique_values = set()

        for row in x:
            val = row.get(best_attr)
            if val is not None:
                unique_values.add(val)

        # Pre každú unikátnu hodnotu atribútu vytvor jednu vetvu stromu
        for val in unique_values:
            x_subset = []  # Podmnožina vstupných dát, kde má atribút hodnotu `val`
            y_subset = []  # Zodpovedajúce výstupy (cieľové hodnoty)

            # Vyber vzorky, pre ktoré má atribút hodnotu `val`
            for row, label in zip(x, y):
                if row.get(best_attr) == val:
                    x_subset.append(row)
                    y_subset.append(label)

            # Ak nie sú žiadne záznamy s danou hodnotou (môže sa stať pri chýbajúcich údajoch),
            # vytvor listový uzol s najčastejšou triedou z pôvodnej množiny
            if not x_subset:
                most_common_value = Counter(y).most_common(1)[0][0] # Counter - zisti početnosť hodnôť (napr. low:3, ...)
                node.children[val] = TreeNode(value=most_common_value)

            else:
                # Vytvor zoznam atribútov pre ďalšiu úroveň (bez už použitého atribútu)
                new_attrs = []
                for attr in attributes:
                    if attr != best_attr:
                        new_attrs.append(attr)

                # Rekurzívne vytvor podstrom pre túto vetvu
                node.children[val] = self._build_tree(
                    x_subset,
                    y_subset,
                    new_attrs,
                    depth + 1
                )

        return node

    @staticmethod
    def _entropy(y):
        """
        Táto metóda vypočíta entrópiu
        """
        total_samples = len(y)  # Celkový počet vzoriek
        label_counts = Counter(y)  # Počet výskytov jednotlivých tried

        entropy = 0.0  # Počiatočná hodnota entropie

        for count in label_counts.values():
            probability = count / total_samples  # Pravdepodobnosť triedy

            if probability > 0:
                entropy -= probability * np.log2(probability)  # Príspevok do entropie

        return entropy

    def _information_gain(self, x, y, attribute):
        """
        Táto metóda vypočíta informačný zisk
        """

        total_entropy = self._entropy(y) # Spočítaj entropiu celej množiny (pred rozdelením podľa atribútu)
        total_samples = len(y) # Celkový počet vzoriek

        # Získaj všetky unikátne (neprázdne) hodnoty daného atribútu
        unique_values = set()

        for row in x:
            attr_value = row.get(attribute)

            if attr_value is not None:
                unique_values.add(attr_value)

        weighted_entropy = 0.0 # Inicializuj váženú entropiu po rozdelení

        # Pre každú hodnotu atribútu vypočítaj entropiu podmnožiny
        for value in unique_values:
            # Vyber triedy (y), ktorým zodpovedajú riadky s danou hodnotou atribútu
            subset_labels = []

            for row, label in zip(x, y):
                if row.get(attribute) == value:
                    subset_labels.append(label)

            subset_size = len(subset_labels) # Veľkosť podmnožiny
            subset_entropy = self._entropy(subset_labels) # Entropia tejto podmnožiny
            weight = subset_size / total_samples # Váha podmnožiny podľa jej veľkosti
            weighted_entropy += weight * subset_entropy # Pridaj príspevok tejto podmnožiny do celkovej váženej entropie

        information_gain = total_entropy - weighted_entropy # Informačný zisk = entropia pred rozdelením - entropia po rozdelení

        return information_gain

    def _best_attribute(self, x, y, attributes):
        """
        Nájde atribút s najvyšším informačným ziskom.
        Ak je nastavený limit na počet atribútov (max_features), použije sa náhodný výber.
        """

        # Ak je obmedzený počet atribútov, náhodne vyber podmnožinu (napr. v random foreste)
        if self.max_features is not None and len(attributes) > self.max_features:
            attributes = self.rng.choice(
                attributes,
                size=self.max_features,
                replace=False
            )

        # Vypočítaj informačný zisk pre každý atribút
        gains = []

        for attr in attributes:
            gain = self._information_gain(x, y, attr)
            gains.append((attr, gain))

        # Inicializuj najlepší atribút a jeho informačný zisk
        best_attribute = None
        best_gain = -float('inf')  # Začneme s najmenším možným číslom

        # Prejdi všetky dvojice (atribút, informačný zisk)
        for attr, gain in gains:
            if gain > best_gain:
                best_attribute = attr
                best_gain = gain

        return best_attribute

    def _predict_row(self, row, node):
        """
        Rekurzívne predikuje výstupovú triedu pre jeden riadok vstupných dát
        pomocou prechodu rozhodovacím stromom.
        """

        # 1. Ak sme v listovom uzle (t.j. uzol obsahuje výslednú triedu), vráť túto hodnotu
        if node.value is not None:
            return node.value

        # 2. Získaj hodnotu atribútu pre aktuálny uzol
        attr_val = row.get(node.attribute)

        # 3. Pokús sa nájsť vetvu (dieťa) zodpovedajúce tejto hodnote atribútu
        child = node.children.get(attr_val)

        # 4. Ak taká vetva neexistuje (hodnota sa nevyskytla počas trénovania),
        # vráť najčastejšiu triedu v trénovacích dátach ako predvolenú predikciu
        if child is None:
            return self.majority_class

        # 5. Inak pokračuj rekurzívne po strome smerom dole
        return self._predict_row(row, child)

    def predict(self, x):
        """
        Penerovanie predikcie tried (výstupov) pre nové vstupné dáta pomocou už natréno-
        vaného rozhodovacieho stromu.
        """
        predictions = []

        if isinstance(x, pd.DataFrame):
            x = x.to_dict(orient="records") # Premena na zoznam slovníkov

        # Pre každý riadok vstupných dát aplikuj rekurzívnu predikciu
        for row in x:
            prediction = self._predict_row(row, self.tree)
            predictions.append(prediction)

        return predictions

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        """
        Matematický ýpočet presností
        """
        correct_predictions = 0  # Počet správnych predpovedí

        for actual, predicted in zip(y_true, y_pred):
            if actual == predicted:
                correct_predictions += 1

        total_predictions = len(y_true)  # Celkový počet prípadov
        accuracy = correct_predictions / total_predictions  # Presnosť ako pomer správnych

        return accuracy

    def score(self, x, y, return_error=False):
        """
        Vyvolanie výpočtu presností
        """
        y_pred = self.predict(x)
        acc = self._accuracy_score(y, y_pred)

        if return_error:
            return 1 - acc

        return acc

    def visualize(self, max_depth=None):
        """
        Vizualizuje rozhodovací strom do konzoly ako stromovú štruktúru.
        Voliteľne môžeš obmedziť maximálnu hĺbku výpisu.
        """
        def _print_node(node, depth):
            """
             Realizácia vizualizácie rozhodovacieho stromu.
            """
            # Ak je zadaný limit hĺbky a sme za ním, skonči
            if max_depth is not None and depth > max_depth:
                return

            indent = "  " * depth  # Odsadenie podľa hĺbky uzla

            if node.value is not None:
                # Listový uzol – obsahuje predikovanú triedu
                print(f"{indent} Prediction: {node.value}")
            else:
                # Vnútorný uzol – obsahuje atribút, podľa ktorého sa vetví
                print(f"{indent} Attribute: {node.attribute}")

                # Pre každú hodnotu tohto atribútu vypíš podstrom
                for value, child in node.children.items():
                    print(f"{indent}  └── If {node.attribute} == {value}:")
                    _print_node(child, depth + 1)

        print("DECISION TREE VISUALIZATION")
        _print_node(self.tree, depth=0)


# RANDOM FOREST
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        """
        Výtvorí konštruktor so zvolenými parametrami
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.features_idx = []
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)  # lokálny RNG

    def fit(self, x, y):
        """
        Trénuje náhodný les (Random Forest) z viacerých rozhodovacích stromov.
        Každý strom je trénovaný na náhodnej vzorke (s opakovaním) z trénovacích dát.
        """

        # Overenie vstupných dát – musia byť vo forme pandas DataFrame a Series
        if not isinstance(x, pd.DataFrame):
            raise ValueError("X must be DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("Y must be Series")

        # Inicializácia zoznamov stromov a použitých atribútov (nepoužité zatiaľ)
        self.trees = []
        self.features_idx = []  # môže slúžiť na sledovanie výberu atribútov pre každý strom

        # Trénuj n rozhodovacích stromov (n_estimators)
        for _ in range(self.n_estimators):
            # Bootstrap vzorkovanie – náhodne vyber riadky s opakovaním
            indices = self.rng.choice(len(x), len(x), replace=True)

            # Urči počet atribútov, ktoré sa budú náhodne vyberať v rámci každého stromu
            p = int(np.sqrt(x.shape[1])) # Počet náhodne vybraných atribútov (realistickejší Random Forest)
            n_features = max(1, p)

            # Vytvor podmnožinu dát na trénovanie stromu
            x_sample = x.iloc[indices]
            y_sample = y.iloc[indices]

            # Vytvor nový rozhodovací strom s danou maximálnou hĺbkou a počtom atribútov
            tree = DecisionTree(max_depth=self.max_depth, max_features=n_features)

            # Natrénuj strom na tejto náhodnej vzorke
            tree.fit(x_sample, y_sample)

            # Ulož strom do zoznamu vytrénovaných stromov
            self.trees.append(tree)

    def predict(self, x):
        """
        Predikuje výstupy (triedy) pre vstupné dáta pomocou viacerých rozhodovacích stromov.
        Funguje na princípe väčšinového hlasovania (majority voting) medzi stromami.
        """

        # Skontroluj, že vstup je typu DataFrame — inak vyhoď chybu
        if not isinstance(x, pd.DataFrame):
            raise ValueError("X must be a DataFrame")

        predictions = []  # Zozbiera predikcie zo všetkých stromov

        # Pre každý strom v náhodnom lese získaj jeho predikcie
        for tree in self.trees:
            preds = tree.predict(x)  # predikcia pre všetky vzorky
            predictions.append(preds)

        # Transpozícia: každý riadok bude obsahovať všetky predikcie pre jednu konkrétnu vzorku
        # Napr. pre vzorku 0: ['low', 'low', 'medium', 'low']
        # Po získaní predikcií z viacerých stromov máš predikcie usporiadané podľa stromov,
        # ale potrebuješ ich usporiadať podľa vzoriek – aby si vedel pre každú vzorku určiť
        # väčšinovú predikciu (hlasovanie).
        predictions = np.array(predictions).T

        final_preds = []  # Sem sa uložia finálne predikcie pre každú vzorku

        # Pre každú vzorku (riadok) urči najčastejšiu predikciu zo všetkých stromov
        for row in predictions:
            label_counts = Counter(row)  # Spočítaj, koľko krát bola ktorá trieda predikovaná
            most_common_label = label_counts.most_common(1)[0][0]  # Vyber triedu s najvyšším počtom hlasov
            final_preds.append(most_common_label)

        # Vráť zoznam finálnych predikcií pre každú vzorku
        return final_preds

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        """
        Matematický ýpočet presností
        """
        correct_predictions = 0  # Počet správnych predpovedí

        for actual, predicted in zip(y_true, y_pred):
            if actual == predicted:
                correct_predictions += 1

        total_predictions = len(y_true)  # Celkový počet prípadov
        accuracy = correct_predictions / total_predictions  # Presnosť ako pomer správnych

        return accuracy

    def score(self, x, y, return_error=False):
        """
        Vyvolanie výpočtu presností
        """
        y_pred = self.predict(x)
        acc = self._accuracy_score(y, y_pred)

        if return_error:
            return 1 - acc

        return acc


# FUNKCIE
def remove_outliers(data):
    """
    Funkcia vymaže prípadné autliery (ak sa vyskytujú), čiže oebvyklé hodnoty v datasete
    """
    cleaned_data = data.copy()

    for col in cleaned_data.select_dtypes(include=[np.number]).columns:
        Q1 = cleaned_data[col].quantile(0.25)
        Q3 = cleaned_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]

    return cleaned_data

def evaluate_classifier(model, x_test, y_test, model_name):
    """
    Funkcia pre výpis presností a chýb modelov, vrátane konfúznej matice
    """
    y_pred = model.predict(x_test)
    accuracy = model.score(x_test, y_test) * 100
    error = model.score(x_test, y_test, return_error=True) * 100

    print(f"\n{model_name} PERFORMANCE")
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print(f"Classification Error: {error:.2f}%\n")

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def decision_tree(x_train, x_test, y_train, y_test, depth=None):
    """
    Funkcia pre vloženie parametrov do triedy DecisionTree
    """
    model = DecisionTree(max_depth=depth, random_state=42)
    model.fit(x_train, y_train)
    evaluate_classifier(model, x_test, y_test, 'Decision Tree Classifier')
    return model

def random_forest(x_train, x_test, y_train, y_test, n_est):
    """
    Funkcia pre vloženie parametrov do triedy RandomForest
    """
    model = RandomForest(n_estimators=n_est, random_state=42)
    model.fit(x_train, y_train)
    evaluate_classifier(model, x_test, y_test, 'Random Forest Classifier')
    return model

def export_predictions(model, x_test, y_test, output_path, model_name):
    """
    Funkcia pre export skutočných hodnôt a predikcií modelov do zvlášť CSV súborov
    """
    predictions = model.predict(x_test)
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })
    results_df.to_csv(output_path, index=False)
    print(f"{model_name} predictions exported to {output_path}")


# SPRACOVANIE DÁT
file_path = "dataset_flights.csv"
flight_data = pd.read_csv(file_path)

# Získanie informácii o duplicitách a nulových hodnotách
null_values = flight_data.isnull().sum().sum()
duplicates = flight_data.duplicated().sum()
samples = flight_data.shape[0]

print("ORIGINAL DATA STATS:\nmissing values:", null_values, "\nduplicates:", duplicates, "\nsamples:", samples, "\n")

flight_data = flight_data.drop(columns=['ID', 'flight'], errors='ignore')

flight_data = flight_data.drop_duplicates()
flight_data = flight_data.dropna()
flight_data = remove_outliers(flight_data)

null_values = flight_data.isnull().sum().sum()
duplicates = flight_data.duplicated().sum()
samples = flight_data.shape[0]

print("CLEANED DATA STATS:\nmissing values:", null_values, "\nduplicates:", duplicates, "\nsamples:", samples)


# EDA ANALÝZA DAT
plt.figure(figsize=(8, 5))
sns.barplot(data=flight_data, hue='airline', y='price', errorbar=None, palette="viridis")
plt.title('Average Price by Airline')
plt.xlabel('Airline')
plt.ylabel('Average Price')
plt.show()


# KATEGORIZÁCIA CENY
flight_data['price'] = pd.qcut(flight_data['price'], q=3, labels=['low', 'medium', 'high'])
flight_data['duration'] = pd.qcut(flight_data['duration'], q=3, labels=['short', 'medium', 'long'])
flight_data['days_left'] = pd.qcut(flight_data['days_left'], q=3, labels=['last_minute', 'soon', 'flexible'])

print(flight_data.head())


# PRÍPRAVA PRE KLASIFIKÁCIU
X = flight_data.drop(columns=['price'])
Y = flight_data['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# MODELY
decision_tree_model = decision_tree(X_train, X_test, Y_train, Y_test)
decision_tree_model.visualize()
export_predictions(decision_tree_model, X_test, Y_test, "decision_tree_predictions.csv", "DECISION TREE")

random_forest_model = random_forest(X_train, X_test, Y_train, Y_test, n_est=150)
export_predictions(random_forest_model, X_test, Y_test, "random_forests_predictions.csv", "RANDOM FOREST")