# Sentiment-Analysis
Este projeto utiliza algoritmos de classificação (Perceptron, Average Perceptron e Pegasos) em machine learning para análise de sentimentos ou opiniões em avaliações de produtos e serviços.

O arquivo `main.py` é o script principal que orquestra o fluxo de trabalho de `project1.py`, realizando a carga de dados, extração de _features_, treinamento, ajuste de hiperparâmetros, avaliação dos classificadores e visualização dos resultados. Em outras palavras, o script integra diversas etapas do _pipeline_ de um sistema de análise de sentimentos, ou seja, as funções definidas em `project1.py` e `utils.py` para executar experimentos de análise de sentimentos usando representações do tipo _bag-of-words_, conforme as seguintes etapas:

1. **Carga e Pré-processamento:** Leitura dos dados, extração do vocabulário e construção das features a partir de textos.
2. **Treinamento e Visualização:** Aplicação de diferentes algoritmos de classificação em um conjunto "toy" para validação visual.
3. **Avaliação e Tuning:** Treinamento nos dados reais, ajuste de hiperparâmetros com base em acurácias de validação e visualização dos resultados.
4. **Teste Final e Interpretação:** Aplicação do melhor modelo ao conjunto de teste, extração de palavras-chave explicativas e experimentos com diferentes representações de features.

A seguir um detalhamento sobre a estrutura e funcionalidades dos arquivos do sistema de análise de sentimentos.

---

## **Estrutura e funcionalidades de `main.py`**

### 1. Carregamento dos Dados

- **Leitura dos arquivos de dados reviews:**  
  O script utiliza uma função em `utils.py` para carregar os dados dos arquivos TSV (`reviews_train.tsv`, `reviews_val.tsv` e `reviews_test.tsv`). Para cada partição (treino, validação e teste), ele extrai os textos dos reviews e os rótulos (sentimento, +1 ou -1).

- **Extração do vocabulário e construção das features:**  
  A partir dos textos de treinamento, é construído um dicionário de palavras (_bag-of-words_), usando a função `bag_of_words` definida em *project1.py*. Em seguida, o script converte os textos dos três conjuntos em matrizes de _features_ utilizando `extract_bow_feature_vectors`.

### 2. Experimentos com dados "Toy"

- **Carregamento de um conjunto de dados simples (`toy_data.tsv`):**  
  O script carrega um conjunto de dados 2D simples para facilitar a visualização e compreensão dos algoritmos.

- **Treinamento com diferentes algoritmos:**  
  São treinados três classificadores, a partir das funções disponíveis em `project1.py`:
  - **Perceptron:** Usando a função `perceptron`.
  - **Perceptron Médio:** Utilizando `average_perceptron`.
  - **Pegasos:** Aplicando o algoritmo Pegasos através da função `pegasos` com parâmetros T (número de épocas) e L (parâmetro de regularização).

- **Visualização dos Resultados:**  
  Cada conjunto de parâmetros obtidos (vetor de pesos e viés) é visualizado com a função `plot_toy_data` (do módulo `utils.py`), que plota os pontos do conjunto "toy" e a respectiva fronteira de decisão.

### 3. Avaliação dos Classificadores em Dados Reais

- **Cálculo da acurácia de treinamento e validação:**  
  Para os três algoritmos (Perceptron, Perceptron Médio e Pegasos), o script utiliza a função `classifier_accuracy` para treinar o modelo nos dados de treinamento e, em seguida, calcular a acurácia tanto no conjunto de treinamento quanto no de validação.

- **Impressão dos resultados:**  
  São exibidos os valores de acurácia para cada método, permitindo comparar o desempenho entre eles.

### 4. Ajuste de Hiperparâmetros (_Tuning_)

- **Tuning para o Perceptron e o Perceptron Médio:**  
  Utilizando funções como `tune_perceptron` e `tune_avg_perceptron` (definidas em `utils.py`), o script varre diferentes valores para T (número de épocas) e avalia a acurácia nos dados de validação.

- **Tuning para o Pegasos:**  
  Para o algoritmo Pegasos, há duas etapas:
  - Primeiramente, fixando um valor de `L` (lambda), são testados diferentes valores para `T` com a função `tune_pegasos_T`.
  - Em seguida, fixando o melhor `T` obtido, são avaliados diferentes valores para `L` através de `tune_pegasos_L`.

- **Visualização dos resultados de tuning:**  
  São gerados gráficos (usando `plot_tune_results`) que mostram como a acurácia varia em função dos hiperparâmetros, ajudando a selecionar os melhores valores.

### 5. Avaliação Final no Conjunto de Teste

- **Treinamento com os melhores hiperparâmetros:**  
  Com base nos resultados de validação, o algoritmo treina novamente o modelo (neste exemplo, o Pegasos) utilizando os melhores valores de T e L.

- **Predição e Cálculo da Acurácia no Teste:**  
  O classificador treinado é aplicado ao conjunto de teste para obter predições, e a acurácia é calculada e impressa.

### 6. Análise das palavras mais relevantes e experimentos com Stopwords

- **Extração das Palavras Mais Explanatórias:**  
  O script utiliza a função `most_explanatory_word` (definida em `utils.py`) para identificar as palavras com maiores pesos no modelo (`best_theta`). Isso indica quais termos têm maior influência na predição do sentimento.

- **Experimentos com remoção de stopwords e com contagens:**  
  O código também demonstra como alterar a construção do dicionário:
  - Primeiro, gera-se o dicionário com a opção de remover _stopwords_.
  - Depois, são extraídas _features_ binarizadas (presença/ausência) e não binarizadas (contagens) para comparar a acurácia dos modelos.

A estrutura modular de **main.py** permite tanto a experimentação, quanto a análise comparativa dos métodos de classificação linear, facilitando a identificação do melhor modelo para a tarefa de análise de sentimentos.  

---

## **Estrutura e funcionalidades de `project1.py`**

O **project1.py** implementa os principais métodos de classificação linear (Perceptron, Perceptron Médio e Pegasos) com funções auxiliares para:
 
- Calcular o _hinge loss_ de modo a se avaliar o desempenho do classificador.
- Atualizar os parâmetros de forma condicional com base na margem de decisão.
- Extrair _features_ de textos, usando a técnica de _"bag-of-words"_, com opção de remoção de _"stopwords"_.

### 1. Funções de Utilidade e Pré-processamento

- **`get_order(n_samples):`**  
  Essa função gera uma ordem (lista) de índices a serem usados para iterar sobre as amostras durante o treinamento. Ela tenta primeiro ler uma ordem pré-definida dos arquivos de dados (por exemplo, `200.txt` ou `400.txt`) e, se não o encontrar, gera uma ordem aleatória (com semente fixa para reprodutibilidade). Essa abordagem garante que, em ambientes em que um “_shuffle_” pré-definido seja desejável, a ordem dos exemplos seja controlada.

- **`extract_words(text):`**  
  Dado um texto, essa função trata a pontuação e os dígitos como tokens separados (inserindo espaços ao redor deles), converte o texto para minúsculas e o divide em palavras. Essa função é a base para a construção do vocabulário.

- **`bag_of_words(texts, remove_stopword=True):`**  
  A partir de uma lista de textos, constrói um dicionário onde cada palavra (excetuando as _stopwords_, se assim configurado) é mapeada para um índice único. Isso forma a base para a representação de `bag-of-words` dos textos.

- **`extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):`**  
  Utilizando o dicionário criado, essa função gera uma matriz de _features_ em que cada linha corresponde a um _review_ e cada coluna a uma palavra do vocabulário. Se o parâmetro `binarize` for `True`, cada entrada indica apenas a presença (1) ou ausência (0) da palavra; caso contrário, utiliza a contagem de ocorrências.

### 2. Cálculo de Perda (_Hinge Loss_)

- **`hinge_loss_single(feature_vector, label, theta, theta_0)`:**  
  Essa função calcula o *hinge loss* para um único exemplo, computando a margem definida por  
  $\text{margin} = y \times (\theta^\top x + \theta_0)$ e retorna $\max(0, 1 - \text{margin})$. Assim, se o exemplo estiver “dentro” da margem (ou classificado incorretamente), a perda é positiva; caso contrário, é zero.

- **`hinge_loss_full(feature_matrix, labels, theta, theta_0)`:**  
  Essa função generaliza o cálculo para todo um conjunto de dados. Ou seja, calcula, de forma vetorizada, a margem para todos os exemplos, determinando a perda para cada um (usando o mesmo $\max(0, 1-\text{margin})$) e retorna a média dessas perdas.

### 3. Implementações dos Algoritmos de Classificação Linear

O código implementa três variantes de algoritmos de aprendizado linear:

#### 3.1. Perceptron

- **`perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):`**  
  Realiza uma atualização única do Perceptron para um dado exemplo. Calcula a decisão e a margem; se a margem for menor ou igual a um pequeno epsilon (considerando possíveis imprecisões numéricas), atualiza os parâmetros conforme:  
    
  $$\theta \leftarrow \theta + y \times x \quad \text{e} \quad \theta_0 \leftarrow \theta_0 + y$$

  Caso o exemplo esteja classificado corretamente ($margem > 0$), os parâmetros permanecem inalterados.

- **`perceptron(feature_matrix, labels, T)`:**  
  Implementa o algoritmo completo do Perceptron. Inicialmente, os parâmetros são zerados. Para cada época (T iterações sobre todo o conjunto), os exemplos são processados na ordem definida por `get_order(n_samples)`. Se um exemplo estiver classificado incorretamente (ou seja, se $y_i \times (\theta^\top x_i + \theta_0) \leq 0$), os parâmetros são atualizados. Ao final, retorna o vetor de pesos e o viés.

#### 3.2. Average Perceptron

- **`average_perceptron(feature_matrix, labels, T):`**  
  Essa variante acumula os valores dos parâmetros ($θ e θ₀$) em cada iteração e, ao final, retorna a média desses valores. A ideia por trás do Perceptron médio é que a média dos iterados tende a ter melhor generalização, pois suaviza flutuações decorrentes das atualizações individuais.

#### 3.3. Pegasos

Pegasos é um algoritmo de **Gradiente Descendente Estocástico** (SGD) para resolver o problema de SVM com margem de perda (_hinge loss_) e termo de regularização.

- **`pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0):`**  
  Realiza uma única atualização do algoritmo Pegasos. Primeiro, calcula-se a margem $y \times (\theta^\top x + \theta_0)$. Se essa margem for menor ou igual a 1, o modelo sofre perda e os parâmetros são atualizados tanto pelo gradiente da perda quanto pelo efeito da regularização:

  $$\theta \leftarrow (1 - \eta \times L) \times \theta + \eta \times y \times x,$$

  e o viés é atualizado como  

  $$\theta_0 \leftarrow \theta_0 + \eta \times y.$$
  
  Se a margem for maior que 1, apenas é aplicada a regularização (multiplicando \(\theta\) por \(1 - \eta \times L\)) e o viés permanece inalterado.

- **`pegasos(feature_matrix, labels, T, L):`**  
Implementa o algoritmo completo do Pegasos, com os parâmetros inicializados com zeros. Em cada época, os exemplos são processados em ordem aleatória (usando `np.random.permutation`). A taxa de aprendizado é definida de forma adaptativa como $\eta = 1/\sqrt{t}$, onde $t$ é o contador global de atualizações. Em cada iteração, a atualização é feita conforme a regra descrita anteriormente (com ou sem gradiente da perda, dependendo se o exemplo está "dentro" da margem ou não). Ao final, o algoritmo retorna os parâmetros finais ($θ e θ₀$).

### 4. Funções de Predição e Avaliação

- **`classify(feature_matrix, theta, theta_0):`**  
  Essa função recebe uma matriz de características e os parâmetros do classificador e retorna um vetor de predições. Sua finalidade é calcular $\theta^\top x + \theta_0$ para cada exemplo e retornar 1 se o resultado for maior que zero e -1 caso contrário. Há um tratamento especial para valores muito próximos de zero (usando `np.isclose`) para evitar problemas numéricos.

- **`classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, **kwargs):`**  
  Essa função recebe uma função de treinamento (por exemplo, `perceptron`, `average_perceptron` ou `pegasos`), treina o classificador com os dados de treinamento, realiza predições tanto no conjunto de treinamento quanto no de validação e calcula a acurácia (usando a função auxiliar `accuracy`). O retorno é uma tupla com as acurácias de treinamento e validação.

- **`accuracy(preds, targets):`**  
  Calcula a fração de predições corretas, ou seja, a média de `(preds == targets)`.

### Considerações Gerais

A estrutura modular facilita a compreensão dos passos e a manutenção futura, permitindo que diferentes algoritmos sejam testados, comparados e aplicados a conjuntos de dados reais (como os de _reviews_ de produtos). É possível averiguar tanto a clareza na separação das responsabilidades quanto o uso de técnicas de vetorização e atualização adaptativa (no caso do Pegasos) para garantir eficiência e escalabilidade.

- **Modularidade e Reusabilidade:**  
  Cada função é projetada para ser modular e reutilizável. Por exemplo, as funções de cálculo da perda (tanto para um exemplo quanto para o conjunto completo) são separadas dos algoritmos de treinamento, permitindo que possam ser testadas e modificadas independentemente.

- **Vetorização:**  
  Funções como `hinge_loss_full` aproveitam operações vetorizadas do NumPy para computar margens e perdas de forma eficiente, o que é crucial para escalabilidade em grandes conjuntos de dados.

- **Atualizações Condicionais:**  
  Tanto o Perceptron quanto o Pegasos realizam atualizações condicionais: somente quando o exemplo é classificado incorretamente (ou quando a margem é insuficiente, no caso do Pegasos) os parâmetros são modificados. Essa lógica é central para garantir que os algoritmos se ajustem aos erros e, ao mesmo tempo, mantenham regularização (no caso do Pegasos).

- **Aplicação em NLP:**  
  A segunda parte do código (bag-of-words e extração de features) está orientada para o processamento de linguagem natural, convertendo textos em vetores de características que podem ser usados pelos algoritmos de classificação. Essa abordagem é típica em tarefas de análise de sentimentos, onde os textos (reviews) são transformados em representações numéricas.

---

## **Estrutura e funcionalidades de `utils.py`**

O arquivo **utils.py** contém um conjunto de funções auxiliares que suportam as operações de carregamento, processamento, visualização e ajuste (_tuning_) dos experimentos desenvolvidos no projeto. O código visa separar as rotinas genéricas e de apoio da lógica central dos algoritmos, promovendo modularidade e reuso do código.

### 1. Importações e Configuração Inicial

- **Importações:**  
  São importadas bibliotecas essenciais como `csv` (para manipulação de arquivos CSV/TSV), `numpy` (para operações numéricas e vetorização), `matplotlib.pyplot` (para plotagem de gráficos) e o módulo `project1` (que contém os algoritmos de aprendizado). Também consta a verificação da versão do Python para compatibilidade com Python 3.

- **Variável PYTHON3:**  
  A variável `PYTHON3` é definida para determinar se o código está sendo executado em Python 3, o que influencia na forma de abrir arquivos (codificação, etc.).

### 2. Funções de Carregamento de Dados

- **`load_toy_data(path_toy_data):`**  
  Carrega um conjunto de dados “_toy_” (simples) a partir de um arquivo TSV.  
  - Usa a função `np.loadtxt` para ler o arquivo, desempacotando os dados em três vetores: um para os rótulos e dois para as coordenadas dos pontos.  
  - As coordenadas são empilhadas verticalmente e transpostas para formar uma matriz onde cada linha é um exemplo em 2D.  
  - Retorna uma tupla com a matriz de features e o vetor de rótulos.

- **`load_data(path_data, extras=False):`**  
  Lê dados de reviews a partir de um arquivo TSV e retorna uma lista de dicionários.  
  - Cada dicionário representa um exemplo com, no mínimo, as chaves básicas `'sentiment'` e `'text'`.  
  - Se o parâmetro `extras` for `True`, informações adicionais (como `productId`, `userId`, `summary`, `helpfulY` e `helpfulN`) também são mantidas.  
  - Converte valores numéricos (definidos em `numeric_fields`) para inteiros quando apropriado.

- **`write_predictions(path_submit_data, preds):`**  
  Essa função é responsável por atualizar um arquivo de submissão com as predições.  
  - Lê o arquivo original, verifica se o número de predições é compatível e substitui o valor do campo `'sentiment'` para cada exemplo.  
  - Em seguida, regrava os dados no arquivo utilizando um `DictWriter`.

### 3. Funções de Plotagem e Visualização

- **`plot_toy_data(algo_name, features, labels, thetas):`**  
  Plota o conjunto de dados 2D "toy" juntamente com a fronteira de decisão do classificador.  
  - Os pontos são coloridos (por exemplo, azul para rótulo +1 e vermelho para -1).  
  - A fronteira de decisão é calculada com base na equação $\theta[0] \cdot x + \theta[1] \cdot y + \theta_0 = 0$, resolvendo para $y$ e plotando uma linha.

- **`plot_tune_results(algo_name, param_name, param_vals, acc_train, acc_val):`**  
  Cria gráficos que ilustram a variação da acurácia de treinamento e validação em função de um hiperparâmetro (por exemplo, número de épocas ou valor de lambda).  
  - Plota duas curvas (uma para os dados de treinamento e outra para validação) e adiciona legendas e títulos para facilitar a interpretação dos resultados.

### 4. Funções para ajuste (_tuning_) de hiperparâmetros

- **`tune(train_fn, param_vals, train_feats, train_labels, val_feats, val_labels):`**  
  Essa função genérica executa uma varredura (_grid search_) sobre um conjunto de valores para um hiperparâmetro.  
  - Para cada valor, treina o modelo (usando a função `train_fn`) e avalia a acurácia tanto no conjunto de treinamento quanto no de validação.  
  - Retorna dois arrays com as acurácias correspondentes aos diferentes valores testados.

- **`tune_perceptron, tune_avg_perceptron, tune_pegasos_T e tune_pegasos_L:`**  
  São funções _wrapper_ que especializam a função genérica `tune` para os diferentes algoritmos (Perceptron, Perceptron Médio e Pegasos).  
  - No caso do Pegasos, há uma diferenciação entre ajustar o número de iterações `T` (mantendo fixo lambda) e ajustar o valor de lambda (mantendo fixo `T`).

### 5. Função para extração de palavras explanatórias

- **most_explanatory_word(theta, wordlist):**  
  Dada uma lista de pesos (best_theta) e uma lista ordenada de palavras (wordlist), essa função retorna as palavras ordenadas de acordo com a magnitude dos pesos (em ordem decrescente).  
  - Essa função é útil para interpretar o modelo, identificando quais palavras têm maior influência nas decisões do classificador.

Em linhas gerais, o módulo **utils.py** serve como um conjunto de ferramentas auxiliares que facilitam a:
  
- **Carga e processamento dos dados:** permite ler tanto conjuntos de dados reais quanto conjuntos de dados de teste (toy).  
- **Visualização dos resultados:** Gera gráficos para visualizar a fronteira de decisão e a performance dos modelos em função dos hiperparâmetros.  
- **Ajuste de hiperparâmetros:** Fornece funções para explorar diferentes configurações e identificar os melhores parâmetros para os algoritmos de aprendizado.  
- **Interpretação dos modelos:** auxilia na identificação das _features_ mais influentes (neste caso, palavras) no modelo de análise de sentimentos.

Em suma, esse arquivo é fundamental para manter o código principal (implementado em *project1.py* e *main.py*) organizado, modular e focado, isolando as tarefas de apoio e análise que permitem a fácil experimentação e visualização dos resultados.  

