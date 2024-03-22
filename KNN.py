import pandas as pd
import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            distances = np.sqrt(np.sum((self.X_train - X_test.iloc[i])**2, axis=1))
            nearest_neighbors = distances.argsort()[:self.k]
            nearest_labels = self.y_train.iloc[nearest_neighbors]
            most_common_label = nearest_labels.mode()[0]
            predictions.append(most_common_label)
        return predictions

def Knn_tets(df):    # Separate features and target
    # Prepare data
    X = df.drop(columns=['A_id', 'Quality'])  # Features
    y = df['Quality']  # Target

    #Dividindo o data set em 2, com 70% para o treinamento
    # e 30% de treinamento
    train_size = int((70/100) * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Instantiate and train the KNN model
    k = 5  # You can adjust k as needed
    model = KNN(k)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = (predictions == y_test).mean()
    print("Accuracy:", accuracy)


def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)
        return None

def treat_DataSet(df):
    # Transforma os dados da coluna em inteiros (good = 1) (bad = 0)
    for i in range(len(df)):
        if df.loc[i, 'Quality'] == 'good':
            df.loc[i, 'Quality'] = 1
        elif df.loc[i, 'Quality'] == 'bad':
            df.loc[i, 'Quality'] = 0

    #Temos que mudar a coluna Acidity para float, já ela é do tipo
    #object que no Data Set e Quality para int, ela também é object
    #(df.info() para verificar o typo de dado da coluna)
    df.info()
    #Tem alguns dados inválidos,(,,,,,,,Created_by_Nidula_Elgiriyewithana,),
    # a linha os remove 
    #(df.dropna(inplace=True)->drop rows with NaN values)
    df.dropna(inplace = True)
    df['Acidity'] = df['Acidity'].astype(float)
    df['Quality'] = df['Quality'].astype(int)
    df.info()

    return df

def main():
    file_path = 'apple_quality.csv' 
    apples = read_csv_file(file_path)
    
    if apples is not None:
        apples = treat_DataSet(apples)
        Knn_tets(apples)

if __name__ == "__main__":
    main()