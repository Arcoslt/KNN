import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def Knn_tets(df, k):    # Separate features and target
    
    #Atritutos, é a parte do data sem as colunas A_id e Quality 
    X = df.drop(columns=['A_id', 'Quality'])
    
    #Objetivo, se a maçã é boa ou ruim
    y = df['Quality']

    #Normalidanzo o dataset
    X = (X - X.min()) / (X.max() - X.min())

    #Dividindo o data set em 2, com 70% para o treinamento
    # e 30% de treinamento
    train_size = int((70/100) * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    predictions = []
    for i in range(len(X_test)):
        distances = np.sqrt(np.sum((X_train - X_test.iloc[i])**2, axis=1))
        nearest_neighbors = distances.argsort()[:k]
        nearest_labels = y_train.iloc[nearest_neighbors]
        most_common_label = nearest_labels.mode()[0]
        predictions.append(most_common_label)
    
    print((predictions == y_test).mean()*100)

    return predictions



def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Arquivo não encontrado.")
    except Exception as e:
        print("Error: ", e)
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
        Knn_tets(apples, k=5)

if __name__ == "__main__":
    main()