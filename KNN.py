#pip install pandas
import pandas as pd
import numpy as np
#pip install scikit-learn
from sklearn.metrics import confusion_matrix

def Knn_tets(df, k):    # Separate features and target
    
    #Atritutos, é a parte do data sem as colunas A_id e Quality 
    X = df.drop(columns=['A_id', 'Quality'])
    
    #Objetivo, se a maçã é boa ou ruim
    Y = df['Quality']

    #Normalizando o dataset
    X = (X - X.min()) / (X.max() - X.min())
    Normal = X.join(Y)
    print("MATRIZ NORMALIZADA:")
    print(Normal)

    #Dividindo o data set em 2, com 70% para o treinamento
    # e 30% de treinamento
    #Obs: o conjunto de dados é de 4000 linhas
    train_size = int((70/100) * len(df))
                    #0 a train size / train size ate 4000
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    predictions = []
    for i in range(len(X_test)):
        distances = np.sqrt(np.sum((X_train - X_test.iloc[i])**2, axis=1))
        #Organiza do o Array Distance:argsort()
        #Retorna apenas os primeiros k elementos do rara:[:k]
        nearest_neighbors = distances.argsort()[:k]
        #Seleciona as linhas do conjunto de rótulos de treinamento 
        #correspondentes aos índices dos vizinhos mais próximos 
        #calculados anteriormente
        nearest_labels = Y_train.iloc[nearest_neighbors]
        #Retorna os valores mais comuns no conjunto de dados
        #Adicionando [0], estamos acessando o primeiro elemento,
        #que é o valor mais comum.
        most_common_label = nearest_labels.mode()[0]
        #Append o mais comun, 0 ou 1, que é
        #a predição se a maçã é boa ou ruim
        predictions.append(most_common_label)
    
    print("Matriz de confusão:")
    #Usando o sklearn para caulcular a matriz de confusão
    print(confusion_matrix(Y_test, predictions))
    #Se a predição for igual ao Y_test
    #.mean() calcula a proporção dos iguais
    print("Acurácia:",((predictions == Y_test).mean()*100),"%")

    return predictions


#Leitura do arquivo csv, o que contém o data set
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
    k = 5
    
    if apples is not None:
        apples = treat_DataSet(apples)
        Knn_tets(apples, k)

if __name__ == "__main__":
    main()