import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Recebe um DataFrame df contendo os dados e um parâmetro k para o algoritmo k-NN.
def knn_tests(df, k):
    # Divide os dados em atributos (variáveis independentes) x e rótulos (variável dependente) y.
    x = df.drop(columns=['A_id', 'Quality'])
    y = df['Quality']

    # Normaliza os dados no intervalo [0, 1].
    x = (x - x.min()) / (x.max() - x.min())
    normal = x.join(y)
    print("normalized matrix:")
    print(normal)

    # Divide o conjunto de dados em conjuntos de treinamento e teste.
    train_size = int((70 / 100) * len(df))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Implementa o algoritmo k-NN para fazer previsões no conjunto de teste.
    predictions = []
    for i in range(len(x_test)):
        distances = np.sqrt(np.sum((x_train - x_test.iloc[i]) ** 2, axis=1))
        nearest_neighbors = distances.argsort()[:k]
        nearest_labels = y_train.iloc[nearest_neighbors]
        most_common_label = nearest_labels.mode()[0]
        predictions.append(most_common_label)

    # Calcula a matriz de confusão e a precisão da classificação.
    print("Confusion matrix:")
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    print("Accuracy:", ((predictions == y_test).mean() * 100), "%")

    # Plota a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Samples")
    plt.show()

    # Retorna as previsões feitas pelo modelo.
    return predictions

# Lê um arquivo CSV especificado por file_path usando a biblioteca pandas.
def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("Error: ", e)
        return None


def treat_dataset(df):
    # Realiza o pré-processamento dos dados.
    # Mapeamento dos valores 'good' e 'bad' para 1 e 0, respectivamente.
    df['Quality'] = df['Quality'].replace({'good': 1, 'bad': 0})

    # Remoção de linhas com valores ausentes
    df.dropna(inplace=True)

    # Conversão dos tipos de dados das colunas 'Acidity' e 'Quality'.
    df['Acidity'] = df['Acidity'].astype(float)
    df['Quality'] = df['Quality'].astype(int)

    df.info()

    return df


# Define o caminho do arquivo CSV contendo os dados das maçãs. Chama a função read_csv_file() para ler os dados
# do arquivo. Se os dados forem lidos com sucesso, chama a função treat_dataset() para pré-processá-los e, em
# seguida, chama a função knn_tests() para realizar o teste com o algoritmo KNN.
def main():
    file_path = 'apple_quality.csv'
    apples = read_csv_file(file_path)
    k = 5

    if apples is not None:
        apples = treat_dataset(apples)
        knn_tests(apples, k)


# Garante que o código dentro deste bloco só será executado se o script for executado diretamente e não importado
# como um módulo em outro script. Chama a função main() para iniciar o processo de análise de dados.
if __name__ == "__main__":
    main()
