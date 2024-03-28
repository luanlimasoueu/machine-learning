import tensorflow as tf
import keras


def main():
    print(tf.__version__)

    # carrega os dados do BD de Keras (já dividindo em casos de teste e treino)
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # mostra a representação da matriz com os tons de cores (0 a 255) para cada pixel.
    # Teste com outro valor do array (ex: 42, ao invés de 0)!
    print(train_labels[0])
    print(train_images[0])
    # normalizando os valores para 0 e 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # define a arquitetura da rede neural com 3 camadas
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),  # hidden layer
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # constrói o modelo da rede neural definido antes
    # define-se um otimizador, uma função de perda e métricas
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # treina a rede neural
    # define-se o número de épocas desejada
    # no final do treinamento, deve-se ver uma acurácia.
    # Ex: 0.9098 representa 91% de certeza que a imagem vai ser classsificada corretamente.
    model.fit(train_images, train_labels, epochs=5)
    # testa a rede neural
    # usa dados que a rede ainda não viu para testar sua acurácia
    # no meu caso, retornou 0.8838, ou seja, 88% de acurácia.
    model.evaluate(test_images, test_labels)

    #Exercício 1a
    classifications = model.predict(test_images)
    print(classifications[0])
    #Exercício 1b
    print(test_labels[0])


if __name__ == "__main__":
    main()