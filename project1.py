from string import punctuation, digits
import numpy as np
import random

#==============================================================================
#===  PART I  =================================================================
#==============================================================================

# A função get_order tem por objetivo retornar uma ordem específica de índices para amostras.
def get_order(n_samples):
    """
    Gera uma lista de índices aleatórios para um conjunto de dados de tamanho `n_samples`.

    A função tenta ler uma lista de índices pré-definida de um arquivo dos arquivos TXT:
    '200.txt' ou '400.txt'. 
    Se o arquivo não for encontrado, ela gera uma lista aleatória de índices.

    Args:
        `n_samples`: O número de amostras no conjunto de dados.

    Returns:
        Uma lista de índices aleatórios.
    """
    try:
        # Tenta abrir o arquivo de texto com os índices pré-definidos
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            # Converte a linha lida em uma lista de inteiros
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        # Se o arquivo não for encontrado, gera uma lista aleatória de índices
        random.seed(1)  # Define a semente aleatória para reprodutibilidade
        indices = list(range(n_samples))
        random.shuffle(indices)  # Embaralha os índices aleatoriamente
        return indices

#==============================================================================    
# def get_order(n_samples):
#    return np.random.permutation(n_samples)  # Embaralha os índices
#==============================================================================

#==============================================================================
# Hinge Loss on One Data Sample
#==============================================================================
def hinge_loss_single(feature_vector: np.ndarray, label: float, theta: np.ndarray, theta_0: float) -> float:
    """
    Calcula a margem de perda (hinge loss) para um único ponto de dados, isto é,
    mede o erro da predição com base em uma margem de decisão.

    Args:
        feature_vector (np.ndarray): Vetor de características do exemplo.
        label (float): Rótulo verdadeiro (+1 ou -1).
        theta (np.ndarray): Vetor de pesos do classificador.
        theta_0 (float): Termo de viés (bias).

    Returns:
        float: Valor do hinge loss.
    """
    # Calcula a margem: y * (theta^T * x + theta_0)
    margin = label * (np.dot(theta, feature_vector) + theta_0)
    
    # Calcula a perda de dobradiça
    loss = max(0, 1 - margin)
    
    return float(loss)
    raise NotImplementedError

#==============================================================================
# The Complete Hinge Loss
#==============================================================================
def hinge_loss_full(feature_matrix: np.ndarray, labels: np.ndarray, theta: np.ndarray, theta_0: float) -> float:
    """
    Calcula o hinge loss médio para um conjunto de dados.
    A função calcula a perda para cada ponto de dados e retorna a média dessas perdas.

    Args:
        feature_matrix (np.ndarray): Matriz onde cada linha representa um ponto de dados.
        labels (np.ndarray): Vetor de rótulos (+1 ou -1), onde cada elemento corresponde à classificação correta da linha correspondente em feature_matrix.
        theta (np.ndarray): Vetor de pesos do classificador.
        theta_0 (float): Termo de deslocamento (bias) do classificador.

    Returns:
        float: Valor da perda de dobradiça média associada ao conjunto de dados e parâmetros fornecidos.
    """
    # Calcula a margem para todos os exemplos de uma vez (vetorização)
    # `np.dot(theta, feature_vector)` calculaa o produto escalar de forma eficiente
    margins = labels * (np.dot(feature_matrix, theta) + theta_0)
    
    # Calcula a margem de perda média
    losses = np.maximum(0, 1 - margins)
    
    # Retorna a média das perdas
    return np.mean(losses)
    raise NotImplementedError

#==============================================================================
# Perceptron Single Step Update
#==============================================================================
def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Realiza atualização de um único passo do Perceptron com base em um exemplo de treino.
    O código verifica se um exemplo de treinamento está corretamente classificado.
    Caso contrário, atualiza os pesos para melhorar a classificação.

    Parâmetros:
    feature_vector : np.array, vetor de características do exemplo de entrada.
    label : int, rótulo associado ao exemplo, assumindo valores {+1, -1}.
    current_theta : np.array, vetor de pesos atual do modelo.
    current_theta_0 : float, termo de viés (bias) atual do modelo.

    Retorno:
    new_theta : np.array, vetor de pesos atualizado, caso a classificação seja incorreta.
    new_theta_0 : float, termo de viés atualizado, caso a classificação seja incorreta.
    """
    epsilon = 1e-8  # Pequeno valor para tolerância numérica, evita problemas de arredondamento
    
    # Calcula a predição: θ·x + θ₀
    decision = np.dot(current_theta, feature_vector) + current_theta_0
    
    # Calcula a margem de decisão: y * (θ·x + θ₀)
    margin = label * decision
    
    # Verifica se a margem é menor ou igual a zero (considerando erro numérico)
    if margin <= epsilon:
        # eta = 0.1  # Taxa de aprendizado
        # Atualiza θ e θ₀ se houver erro ou estiver na fronteira
        new_theta = current_theta + (label * feature_vector)
        new_theta_0 = current_theta_0 + label
        return (new_theta.astype(float), float(new_theta_0))
    else:
        # Retorna os parâmetros atuais se não houver erro
        return (current_theta.astype(float), float(current_theta_0))

#==============================================================================
# Full Perceptron Algorithm
#==============================================================================
def perceptron(feature_matrix, labels, T):
    """
    Implementa o algoritmo do Perceptron para aprendizado supervisionado.

    Parâmetros:
    feature_matrix : np.array, shape (n_samples, n_features)
        Matriz de características onde cada linha representa um exemplo de
        treinamento e cada coluna representa uma característica.

    labels : np.array, shape (n_samples,)
        Vetor contendo os rótulos dos exemplos de treinamento, assumindo
        valores {+1, -1}.

    T : int, Número de épocas (iterações completas sobre o conjunto de dados).

    Retorno:
    theta : np.array, shape (n_features,), vetor de pesos aprendido pelo Perceptron.
    theta_0 : float, termo de viés (bias) aprendido pelo Perceptron.

    Observações:
    -----------
    - O algoritmo itera sobre os exemplos de treinamento `T` vezes.
    - A atualização dos pesos ocorre apenas quando um exemplo estiver classificado incorretamente
      (ou seja, quando `y_i * (theta ⋅ x_i + theta_0) <= 0`).
    - A função `get_order(n_samples)` deve ser definida previamente para gerar uma ordem
      aleatória dos exemplos em cada época, o que pode ajudar na convergência.
    """
    n_samples, n_features = feature_matrix.shape
    theta = np.zeros(n_features, dtype=float)  # Inicializa vetor de pesos
    theta_0 = 0.0  # Inicializa viés (bias)

    for _ in range(T):
        order = get_order(n_samples)  # Obtém a ordem dos índices para a iteração atual
        for i in order:
            x_i = feature_matrix[i]  # Vetor de características do i-ésimo exemplo
            y_i = labels[i]          # Rótulo correto do i-ésimo exemplo

            # Calcula a margem: y_i * (θ·x_i + θ_0)
            margin = y_i * (np.dot(theta, x_i) + theta_0)

            # Se a margem for <= 0, atualiza θ e θ_0
            if margin <= 0:
                theta += y_i * x_i
                theta_0 += y_i

    return theta, theta_0

    # raise NotImplementedError
    # for t in range(T):
    #     for i in get_order(nsamples):
    #         # Your code here
    #         raise NotImplementedError
    # # Your code here
    # raise NotImplementedError

#==============================================================================
# Average Perceptron
#==============================================================================

def average_perceptron(feature_matrix, labels, T):
    """
    Implementa o algoritmo do Perceptron Médio (Average Perceptron).

    Parâmetros:
    feature_matrix : np.array, shape (n_samples, n_features), matriz de características,
        cada linha representa um exemplo de treinamento
        e cada coluna representa uma característica.
    labels : np.array, shape (n_samples,), vetor contendo os rótulos dos exemplos de treinamento,
        assumindo valores {+1, -1}.
    T : int, número de épocas (iterações completas sobre o conjunto de dados).

    Retorno:
    avg_theta : np.array, shape (n_features,), vetor de pesos médios aprendidos pelo Perceptron Médio.
    avg_theta_0 : float, termo de viés médio aprendido pelo Perceptron Médio.
    """
    n_samples, n_features = feature_matrix.shape

    # Inicialização dos vetores de pesos
    theta = np.zeros(n_features, dtype=float)  # Vetor de pesos
    theta_0 = 0.0  # Viés

    # Variáveis para acumular os valores das atualizações
    sum_theta = np.zeros(n_features, dtype=float)
    sum_theta_0 = 0.0
    count = 0  # Contador de atualizações

    for _ in range(T):
        order = get_order(n_samples)  # Obtém a ordem dos índices para a iteração atual
        for i in order:
            x_i = feature_matrix[i]  # Vetor de características do i-ésimo exemplo
            y_i = labels[i]          # Rótulo correto do i-ésimo exemplo

            # Calcula a margem de decisão: y_i * (θ·x_i + θ_0)
            margin = y_i * (np.dot(theta, x_i) + theta_0)

            # Se a margem for <= 0, atualiza θ e θ_0
            if margin <= 0:
                theta += y_i * x_i
                theta_0 += y_i

            # Acumula os valores para a média
            sum_theta += theta
            sum_theta_0 += theta_0
            count += 1

    # Calcula a média dos pesos se houver atualizações
    if count == 0:
        return theta, theta_0
    else:
        avg_theta = sum_theta / count
        avg_theta_0 = sum_theta_0 / count
        return avg_theta, avg_theta_0

# print("Vetor de pesos final:", theta)
# print("Termo de viés (bias) final:", theta_0)

#==============================================================================
# Pegasos Single Step Update
#==============================================================================
def pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0):
    """
    Atualiza os parâmetros theta e theta_0 em uma única etapa do Pegasos.
    Retorna novos parâmetros em vez de modificar no local.

    Args:
        feature_vector (np.ndarray): Vetor de características do exemplo.
        label (int): Classificação correta (+1 ou -1).
        L (float): Parâmetro de regularização (lambda).
        eta (float): Taxa de aprendizado.
        theta (np.ndarray): Vetor de pesos atual.
        theta_0 (float): Viés (bias) atual.

    Returns:
        tuple: (theta_atualizado, theta_0_atualizado)
    """

    # Calcula a margem para o exemplo atual
    margin = label * (np.dot(theta, feature_vector) + theta_0)

    # Verifica se o exemplo contribui para a perda (hinge loss > 0)
    # Usar <= 1 para incluir casos de precisão limítrofe
    if margin <= 1:
        # Atualiza theta com regularização e gradiente da perda
        theta_new = (1 - eta * L) * theta + eta * label * feature_vector
        # Atualiza theta_0 apenas com gradiente da perda (sem regularização)
        theta_0_new = theta_0 + eta * label
    else:
        # Apenas aplica a regularização em theta (sem gradiente de perda)
        theta_new = (1 - eta * L) * theta
        theta_0_new = theta_0

    return (theta_new, theta_0_new)
#==============================================================================
# Full Pegasos Algoritm
#==============================================================================

# Versão 1.0
def pegasos(feature_matrix, labels, T, L):
    """
    Executa o algoritmo Pegasos para T iterações através do conjunto de dados.

    Args:
        feature_matrix (np.ndarray): Matriz de características, onde cada linha é um exemplo.
        labels (np.ndarray): Vetor de classificações (+1 ou -1).
        T (int): Número de iterações (épocas) através do conjunto de dados.
        L (float): Parâmetro de regularização (lambda).

    Returns:
        tuple: (theta, theta_0) - Parâmetros atualizados do modelo.
    """
    m, n = feature_matrix.shape
    theta = np.zeros(n) 
    theta_0 = 0.0
    t = 1  # Contador de atualizações

    for _ in range(T):
        # Embaralha os índices dos exemplos a cada época
        shuffled_indices = np.random.permutation(m)
        
        for i in shuffled_indices:
            feature_vector = feature_matrix[i]
            label = labels[i]
            
            # Calcula a margem para o exemplo atual
            margin = label * (np.dot(theta, feature_vector) + theta_0)
            
            # Taxa de aprendizado: η = 1 / sqrt(t)
            eta = 1.0 / np.sqrt(t)
            
            # Atualiza theta e theta_0 conforme a condição da margem
            if margin <= 1:
                theta = (1 - eta * L) * theta + eta * label * feature_vector
                theta_0 += eta * label
            else:
                theta = (1 - eta * L) * theta  # Apenas regularização
            
            t += 1  # Incrementa o contador de atualizações
    
    return (theta, theta_0)
#    raise NotImplementedError



#==============================================================================
#===  PART II  ================================================================
#==============================================================================


#==============================================================================
#pragma: coderesponse template
#==============================================================================

# def decision_function(feature_vector, theta, theta_0):
#     return np.dot(theta, feature_vector) + theta_0
# def classify_vector(feature_vector, theta, theta_0):
#     return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1

#==============================================================================
#pragma: coderesponse end
#==============================================================================

def classify(feature_matrix, theta, theta_0):
    """
    Uma função de classificação que usa parâmetros fornecidos para classificar um conjunto de pontos de dados.

    Args:
        feature_matrix: Uma matriz NumPy descrevendo os dados fornecidos. Cada linha representa 
        um único ponto de dados.
        theta: Um array NumPy descrevendo o classificador linear.
        theta_0: Um número real representando o parâmetro de deslocamento.

    Returns:
        Um array NumPy de 1s e -1s onde o k-ésimo elemento do array é a classificação prevista 
        da k-ésima linha da matriz de recursos usando o theta e theta_0 fornecidos. 
        Se uma previsão for MAIOR QUE zero, ela deve ser considerada uma classificação positiva.
    """
    # Calcula a pontuação para cada ponto de dados.
    scores = np.dot(feature_matrix, theta) + theta_0  

    # Corrige pequenas imprecisões numéricas: trata valores muito próximos de 0 como zero
    scores[np.isclose(scores, 0, atol=1e-10)] = 0

    # Retorna 1 se a pontuação for maior que 0, caso contrário, retorna -1.
    return np.where(scores > 0, 1, -1)  
#   raise NotImplementedError

def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, **kwargs):
    """
    Treina um classificador linear e calcula a acurácia. O classificador é treinado nos dados de treinamento. 
    A acurácia do classificador nos dados de treinamento e validação é então retornada.

    Args:
        classifier: Uma função de aprendizado que recebe argumentos (matriz de recursos, rótulos, **kwargs) 
                    e retorna (theta, theta_0).
        train_feature_matrix: Uma matriz NumPy descrevendo os dados de treinamento. 
                              Cada linha representa um único ponto de dados.
        val_feature_matrix: Uma matriz NumPy descrevendo os dados de validação. 
                            Cada linha representa um único ponto de dados.
        train_labels: Um array NumPy onde o k-ésimo elemento do array é a classificação correta da 
                      k-ésima linha da matriz de recursos de treinamento.
        val_labels: Um array NumPy onde o k-ésimo elemento do array é a classificação correta da k-ésima 
                    linha da matriz de recursos de validação.
        kwargs: Argumentos nomeados adicionais para passar para o classificador (por exemplo, T ou L).

    Returns:
        Uma tupla em que o primeiro elemento é a acurácia (escalar) do classificador treinado nos dados de 
        treinamento e o segundo elemento é a acurácia do classificador treinado nos dados de validação.
    """
    # Treina o classificador com os dados de treinamento e obtém os parâmetros theta e theta_0.
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)

    # Classifica os dados de treinamento e validação usando os parâmetros aprendidos.
    train_preds = classify(train_feature_matrix, theta, theta_0)
    val_preds = classify(val_feature_matrix, theta, theta_0)

    # Calcula a acurácia das previsões nos dados de treinamento e validação.
    train_accuracy = accuracy(train_preds, train_labels)
    val_accuracy = accuracy(val_preds, val_labels)

    # Retorna a acurácia de treinamento e validação como uma tupla.
    return (train_accuracy, val_accuracy)
#    raise NotImplementedError


def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    for c in punctuation + digits:
        text = text.replace(c, f' {c} ')
    return text.lower().split()
    raise NotImplementedError

    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()


def bag_of_words(texts, remove_stopword=True):
    """
    Gera um dicionário mapeando cada palavra presente em 'texts' para um índice único.
    
    Se o parâmetro 'remove_stopword' for True, a função tenta carregar as stopwords 
    a partir do arquivo "stopwords.txt" e não inclui essas palavras no dicionário.
    
    Args:
        texts (list[str]): Lista de textos em linguagem natural.
        remove_stopword (bool): Indica se as stopwords devem ser removidas do dicionário (padrão False).
        
    Returns:
        dict: Dicionário onde cada chave é uma palavra única (exceto as stopwords, se removidas) 
              e cada valor é um índice inteiro único atribuído àquela palavra.
    """
    indices_by_word = {}
    
    # Inicializa o conjunto de stopwords se necessário
    stopwords = set()
    if remove_stopword:
        try:
            with open("stopwords.txt", "r", encoding="utf-8") as f:
                # Cada linha do arquivo deve conter uma stopword
                for line in f:
                    palavra = line.strip()  # Remove espaços em branco e quebras de linha
                    if palavra:
                        stopwords.add(palavra)
        except FileNotFoundError:
            print("Aviso: arquivo 'stopwords.txt' não encontrado. Nenhuma stopword será removida.")
    
    # Itera sobre cada texto na lista de textos
    for text in texts:
        # Extrai as palavras utilizando a função auxiliar 'extract_words'
        words = extract_words(text)
        # Itera sobre cada palavra extraída
        for word in words:
            # Se a remoção de stopwords estiver ativada e a palavra for uma stopword, ignora-a
            if remove_stopword and word in stopwords:
                continue
            # Se a palavra ainda não foi adicionada ao dicionário, adiciona com um índice único
            if word not in indices_by_word:
                indices_by_word[word] = len(indices_by_word)
    
    return indices_by_word


def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Extrai as features de bag-of-words (BoW) para um conjunto de reviews.

    Essa função cria uma matriz de features onde cada linha representa um review e cada coluna
    representa uma palavra única presente no dicionário 'indices_by_word'. O valor na posição [i, j]
    indica a presença (ou frequência) da palavra correspondente no review i.

    Args:
        reviews (list[str]): Lista de strings, onde cada string é um review em linguagem natural.
        indices_by_word (dict): Dicionário que mapeia cada palavra para um índice único.
        binarize (bool): Se True, utiliza indicadores binários (0 ou 1) para indicar a presença da palavra;
                         se False, utiliza a contagem de ocorrências da palavra no review.

    Returns:
        numpy.ndarray: Matriz de features de tamanho (n_reviews, n_palavras) do tipo float64,
                       onde cada elemento representa a presença ou frequência de uma palavra em um review.
    """
    # Cria uma matriz de zeros com número de linhas igual ao número de reviews e 
    # número de colunas igual ao número de palavras no dicionário
    feature_matrix = np.zeros((len(reviews), len(indices_by_word)), dtype=np.float64)
    
    # Itera sobre cada review, obtendo o índice e o texto do review
    for i, text in enumerate(reviews):
        # Extrai a lista de palavras do texto utilizando a função auxiliar extract_words
        words = extract_words(text)
        
        # Itera sobre cada palavra extraída
        for word in words:
            # Verifica se a palavra está presente no dicionário indices_by_word
            if word in indices_by_word:
                # Se binarize for True, marca a presença da palavra com 1
                if binarize:
                    feature_matrix[i, indices_by_word[word]] = 1
                # Se binarize for False, incrementa o contador (frequência) da palavra
                else:
                    feature_matrix[i, indices_by_word[word]] += 1
                    
    # Retorna a matriz de features construída
    return feature_matrix

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
