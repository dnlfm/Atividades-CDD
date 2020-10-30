#!/usr/bin/env python
# coding: utf-8

# <h1>Prova 1 - Introdução à Ciência dos Dados</h1>
# <h3>Daniel Freitas Martins - 2304</h3>

# In[642]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import percentileofscore
#from scipy.stats import shapiro # tem limitacao pra 5000 dados
from scipy.stats import normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[643]:


def lerCSV(caminho_arquivo, header):
    return pd.read_csv(caminho_arquivo, index_col=False, header=header, squeeze=True)


# <h2>Lendo dados de "Filmes-Prova01.csv", correspondentes a lista de filmes de quatro plataformas de <i>Streaming</i> (Netflix, Amazon Prime, Disney+ e Hulu)</h2>

# In[644]:


df = lerCSV("Filmes-Prova01.csv", header=0)
df.head(3)


# <h2>1) Quantos filmes cada uma das 4 plataformas possui? Faça um gráfico de barras para ilustrar esses valores</h2>

# In[645]:


qtd_filmes_netflix = df['Netflix'].sum()
qtd_filmes_hulu = df['Hulu'].sum()
qtd_filmes_amazon = df['Amazon_prime'].sum()
qtd_filmes_disney = df['Disney+'].sum()
print("Quantidade de filmes em cada plataforma:")
print("\t- Netflix:", qtd_filmes_netflix)
print("\t- Hulu:", qtd_filmes_hulu)
print("\t- Amazon Prime:", qtd_filmes_amazon)
print("\t- Disney+:", qtd_filmes_disney)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plataformas = ['Netflix', 'Hulu', 'Amazon Prime', 'Disney+']
qtd_filmes_plataformas = [qtd_filmes_netflix, qtd_filmes_hulu,
                         qtd_filmes_amazon, qtd_filmes_disney]
ax.barh(plataformas, qtd_filmes_plataformas)
for p in ax.patches:
    ax.annotate(p.get_width(), (p.get_width()+5,p.get_y()))
ax.set_ylabel('Plataforma')
ax.set_xlabel('Número de filmes')
plt.title('Plataforma X Número de filmes')
plt.show()


# <h2>2) Quantos filmes estão em mais de uma plataforma?</h2>

# In[646]:


qtd = 0
for index, row in df.iterrows(): # iterando sobre todas as linhas
    if(row['Netflix'] == 1): # se tiver em Netflix, avaliar se tem para algum outro
        if(row['Hulu'] == 1 or row['Amazon_prime'] == 1 or row['Disney+'] == 1):
            qtd+=1 # se tiver para Netflix e algum outro, soma 1
            continue # vá para a próxima linha
    if(row['Hulu'] == 1): # se tiver em Hulu, já sabemos que não tem em Netflix
        if(row['Amazon_prime'] == 1 or row['Disney+'] == 1): # avaliando para os restantes
            qtd+=1 # Analogamente, soma-se 1 se tiver para Hulu e outra plataforma
            continue
    if(row['Amazon_prime'] == 1):
        if(row['Disney+'] == 1):
            qtd+=1
            continue
print("Nº de filmes que estão em mais de uma plataforma:", qtd)


# <h2>3) Qual a plataforma que possui a maior média de nota IMDb?</h2>

# In[647]:


df_filmes_netflix = df[df['Netflix'] == 1]
df_filmes_hulu = df[df['Hulu'] == 1]
df_filmes_amazon = df[df['Amazon_prime'] == 1]
df_filmes_disney = df[df['Disney+'] == 1]


# In[648]:


def calcularMediaIMDb(df, nome_plataforma):
    media = df['IMDb'].sum() / len(df)
    print('Média de nota IMDb da plataforma', nome_plataforma, "=", media)
    return media


# In[649]:


media_netflix = calcularMediaIMDb(df_filmes_netflix, 'Netflix')
media_hulu = calcularMediaIMDb(df_filmes_hulu, 'Hulu')
media_amazon = calcularMediaIMDb(df_filmes_amazon, 'Amazon Prime')
media_disney = calcularMediaIMDb(df_filmes_disney, 'Disney+')

v = {'Netflix': media_netflix, 'Hulu': media_hulu, 'Amazon Prime': media_amazon, 'Disney+': media_disney}
nome_plat = max(v.items(), key=operator.itemgetter(1))[0]

print("\nA plataforma que possui maior média de nota IMDb é a %s, com uma média de aproximadamente %.2f."%(nome_plat, v[nome_plat]))


# <h2>4) Qual o percentual de filmes de cada plataforma com nota IMDb maior que 8?</h2>

# In[650]:


def calcularPercentualNotaIMDb(df, nome_plataforma, limiar=8):
    qtd = 0
    for index, row in df.iterrows():
        if(row['IMDb'] > 8):
            qtd+=1
    percentual = qtd/len(df)*100
    print('\t-', nome_plataforma, "= {:.2f}%".format(percentual))
    return percentual


# In[651]:


print('Percentual de filmes de cada plataforma com nota IMDb maior que 8:')
pnet = calcularPercentualNotaIMDb(df_filmes_netflix, "Netflix")
phul = calcularPercentualNotaIMDb(df_filmes_hulu, "Hulu")
pama = calcularPercentualNotaIMDb(df_filmes_amazon, "Amazon Prime")
pdis = calcularPercentualNotaIMDb(df_filmes_disney, "Disney+")

v = {'Netflix': pnet, 'Hulu': phul, 'Amazon Prime': pama, 'Disney+': pdis}
nome_plat = max(v.items(), key=operator.itemgetter(1))[0]

print('\n')
print(nome_plat, 'possui um maior percentual de filmes com nota IMDb maior que 8.')


# <h2>5) Se uma pessoa é uma apreciadora de filmes clássicos antigos, qual plataforma você mais recomenda? Justifique.</h2>

# In[652]:


df_antigos = df[df['Ano'] <= 1990]
qtd_antigos_netflix = df_antigos['Netflix'].sum()
qtd_antigos_hulu = df_antigos['Hulu'].sum()
qtd_antigos_amazon = df_antigos['Amazon_prime'].sum()
qtd_antigos_disney = df_antigos['Disney+'].sum()
print("Nº filmes antigos Netflix:", qtd_antigos_netflix)
print("Nº filmes antigos Hulu:", qtd_antigos_hulu)
print("Nº filmes antigos Amazon Prime:", qtd_antigos_amazon)
print("Nº filmes antigos Disney+:", qtd_antigos_disney)

v = {'Netflix': qtd_antigos_netflix, 'Hulu': qtd_antigos_hulu, 'Amazon Prime': qtd_antigos_amazon, 'Disney+': qtd_antigos_disney}
nome_plat = max(v.items(), key=operator.itemgetter(1))[0]

print('\n')
print(nome_plat, 'possui um maior número de filmes antigos.')


# <p>Considerando-se que os filmes antigos clássicos estão abaixo do ano de 1990 (inclusive), eu recomendaria a plataforma <b>Amazon Prime</b> por possuir um maior número de filmes antigos. Em comparação com Disney+, a Amazon Prime possui 20 vezes mais filmes antigos disponíveis para escolha.</p>

# <h2>6) Quantas categorias de classificação etária existem? Faça um gráfico de barras com a quantidade de filmes por classificação etária.</h2>

# In[653]:


df_classificacao_unique = df['Classificacao_etaria'].unique
df_classificacao_unique


# <p>Note que existem valores <b>NaN</b>. No entanto, vou considerar para ver também a quantidade de filmes que não possuem a classificação etária informada...</p>

# In[654]:


#df_classificacao_unique = df['Classificacao_etaria'].dropna().unique()
df_classificacao_unique = df['Classificacao_etaria'].unique()
df_classificacao_unique


# In[655]:


qtd_categorias_classificacao_etaria = len(df_classificacao_unique)
print('- Existem', (qtd_categorias_classificacao_etaria-(1 if pd.isnull(df_classificacao_unique).any() else 0)), 
      'categorias de classificação etária diferentes nesta base de dados.\n')
print('- Existem filmes em que a categoria de classificação etária está ausente, como visto acima.')


# In[656]:


dict_somas = dict((el if el is not np.nan else 'Não informado',0) for el in df_classificacao_unique)
for k in dict_somas:
    if(k == 'Não informado'):
        dict_somas[k] = len(df[pd.isnull(df['Classificacao_etaria'])])
    else:
        dict_somas[k] = len(df[df['Classificacao_etaria'] == k])
#for index, row in df.iterrows():
#    dict_somas[row['Classificacao_etaria'] if row['Classificacao_etaria'] is not np.nan else 'Não informado'] += 1
dict_somas


# In[657]:


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
classificacoes_etarias = list(dict_somas)
qtd_filmes = list(dict_somas.values())
ax.barh(classificacoes_etarias, qtd_filmes)
for p in ax.patches:
    ax.annotate(p.get_width(), (p.get_width()+5,p.get_y()))
ax.set_ylabel('Classificação etária')
ax.set_xlabel('Número de filmes')
plt.title('Classificação etária X Número de filmes')
plt.show()


# <p>Note que existe um número muito grande de filmes em que a classificação indicativa não está disponível. Isso pode ser um empecilho na hora de escolher determinados filmes para assistir com toda a família, por exemplo. Das cinco categorias, existem mais filmes adultos (18+) do que os demais.</p>

# <h2>7) Qual plataforma você mais indicaria para uma criança? Justifique.</h2>

# <p>L8069 - Planalto: Art. 2º Considera-se criança, para os efeitos desta Lei, a pessoa até doze anos de idade incompletos, e adolescente aquela entre doze e dezoito anos de idade. (http://www.planalto.gov.br/ccivil_03/leis/l8069.htm)</p>

# In[658]:


df_classificacao = df.groupby('Classificacao_etaria')
df_classificacao_sum = df_classificacao.sum()
df_classificacao_sum


# In[659]:


df_menores = df_classificacao_sum[(df_classificacao_sum.index == '7+') | (df_classificacao_sum.index == 'all')].sum()
qtd_menores_netflix = df_menores['Netflix']
qtd_menores_hulu = df_menores['Hulu']
qtd_menores_amazon = df_menores['Amazon_prime']
qtd_menores_disney = df_menores['Disney+']
print("Nº filmes para crianças Netflix:", qtd_menores_netflix)
print("Nº filmes para crianças Hulu:", qtd_menores_hulu)
print("Nº filmes para crianças Amazon Prime:", qtd_menores_amazon)
print("Nº filmes para crianças Disney+:", qtd_menores_disney)
v = {'Netflix': qtd_menores_netflix, 'Hulu': qtd_menores_hulu, 'Amazon Prime': qtd_menores_amazon, 'Disney+': qtd_menores_disney}
nome_plat = max(v.items(), key=operator.itemgetter(1))[0]

print('\n')
print(nome_plat, 'possui um maior número de filmes próprios para crianças (menores de 12 anos).')


# <p>A plataforma Amazon Prime tem um maior número de filmes com classificação indicativa destinada à crianças (menores de 12 anos). No entanto, ela possui um grande número de filmes adultos.</p>
# <p>- Caso haja um acompanhamento com os pais ou um controle facilitado pela plataforma, eu recomendaria a Amazon Prime.</p>
# <p>- Caso não haja esse acompanhamento com os pais, permitindo que a criança possa assistir e escolher os filmes livremente, eu recomendaria a Disney+, por conter um número razoável de filmes para crianças e poucos filmes destinados aos públicos adolescente e adulto.</p>

# <h2>8) Faça um gráfico de linhas com a quantidade de filmes disponíveis por ano de lançamento. Qual o ano que possui mais filmes, e quantos filmes foram lançados nesse ano? Qual a média de lançamento de filmes por ano? Trace uma linha reta no gráfico com a média, e responda em quais anos foram lançados mais filmes do que a média.</h2>

# In[660]:


df_anos = df.copy()
df_anos['qtd_filmes'] = 1
df_anos = df_anos.groupby('Ano')
df_anos_sum = df_anos['qtd_filmes'].sum()
ax = df_anos_sum.plot(kind='line')

media_lancamentos = df_anos_sum.agg('mean')
print('Média de lançamentos de filmes por ano: %.2f'%media_lancamentos)

plt.axhline(y=media_lancamentos, color='r', linestyle='-')
plt.title('Número de filmes por ano')
ax.set_ylabel('Número de filmes')
plt.show()


# In[661]:


print('Ano que possui mais filmes e o nº de filmes lançados neste ano:')
df_anos_sum.sort_values(ascending=False).head(1)


# In[662]:


df_anos_acima_media = df_anos_sum[df_anos_sum > media_lancamentos]
print("Quantidade de anos cujos nº de lançamento foram maiores que a média: ", len(df_anos_acima_media))
print('Anos com número de lançamentos maior do que a média:')
print('Ano ','Nº lançamentos')
for k in df_anos_acima_media.keys():
    print(k, df_anos_acima_media[k])


# <p>Apenas para sintetizar a resposta:</p>
# <p>- O ano de 2017 teve o maior nº de lançamentos (1401) dentre todos os anos desta base de dados.</p>
# <p>- A média de lançamento de filmes por ano foi de aproximadamente 153 filmes por ano.</p>
# <p>- Dos anos de 2000 a 2019, 20 anos ao todo, tiveram o nº de lançamentos maiores do que a média.</p>

# <h2>9) Faça o Boxplot da duração dos filmes. Em seguida, remova os outliers que achar necessário, e faça o histograma da duração. Ajuste o parâmetro "bins" se necessário, para visualizar melhor. Em seguida,  responda se os valores de duração seguem a distribuição Normal. Justifique.</h2>

# In[663]:


ax = df['Duracao_min'].plot(kind='box')
plt.title('Boxplot - Duração dos filmes')
ax.set_ylabel('Duração (em minutos)')
plt.show()


# In[664]:


lim = 300
df_duracao_limpo = df[df['Duracao_min'] < lim]['Duracao_min']
ax = df_duracao_limpo.plot(kind='box')
plt.title('Boxplot - Duração dos filmes (menor que %d)'%lim)
ax.set_ylabel('Duração (em minutos)')
print('Nº elementos:', len(df_duracao_limpo))
plt.show()


# In[665]:


ax = df_duracao_limpo.hist(bins=75)
plt.title('Histograma - Duração dos filmes')
ax.set_ylabel('Frequência')
ax.set_xlabel('Duração (em minutos)')
plt.show()


# <p>Pelo histograma acima, é possível observar que os valores de duração <b>parecem seguir uma distribuição normal</b>. Para confirmar matematicamente, podemos usar a função <b>scipy.stats.normaltest</b>, baseada nos testes de D’Agostino e Pearson’s</p>
# 
# <p><b>Hipótese nula</b>: Os valores de duração seguem uma distribuição normal.</p>
# <p><b>Hipótese alternativa</b>: Os valores de duração não seguem uma distribuição normal.</p>

# In[666]:


nt = normaltest(df_duracao_limpo)
print("Normal Test: (statistic, p-value) = {}".format(nt))
k2, p = nt
alpha = 1e-3
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("A hipótese nula pode ser rejeitada.")
else:
    print("A hipótese nula não pode ser rejeitada.")


# <p>Note que o p-value obtido pela função normaltest é bem pequeno. De acordo com [1] e [2], se este valor é bem próximo de zero, então a hipótese nula pode ser rejeitada a 0,1% de grau de confiança, sendo <b>improvável que estes dados sigam uma distribuição normal</b>.</p>
# <p>Um resultado curioso. A olho nú parece seguir uma distribuição normal, mas matematicamente, não segue.</p>
# <p>[1] Scipy. Normaltest. Disponível em: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html</p>
# <p>[2] Stackoverflow. Scipy Normaltest how is it used? Disponível em: https://stackoverflow.com/questions/12838993/scipy-normaltest-how-is-it-used</p>

# <h2>10) Quantos gêneros distintos existem? Quantos filmes de cada gênero existem?</h2>

# In[667]:


generos = {}
for index, row in df.iterrows():
    if(pd.isnull(row['Generos'])):
        if('NaN' in generos):
            generos['NaN'] += 1
        else:
            generos['NaN'] = 1
        continue
    v = row['Generos'].split(',')
    for k in v:
        if(k in generos):
            generos[k] += 1
        else:
            generos[k] = 1


# <p>Considerei os valores ausentes também para contabilizar os filmes que não foram classificados em nenhum gênero. Essa informação pode ser útil para saber como normalizar isto, uma vez que a falta desse tipo de informação é prejudicial nas buscas por determinados filmes que não possuem seus gêneros informados.</p>

# In[668]:


print('Existem %d gêneros distintos (excluindo-se os com valores NaN). A quantidade de filmes que cada'%(len(generos) - 1),
      'um possui pode ser visualizada logo abaixo.\n')
print('Existem %d filmes sem nenhum gênero informado (valores NaN / ausentes).\n'%(generos['NaN']))
print("{:<12} {:<20}\n".format('Gênero','Nº filmes'))   
for k in generos:
    print("{:<12} {:<20}".format(k, generos[k]))


# <p>Em resumo, existem 27 gêneros distintos nesta base. O número de filmes que cada gênero possui pode ser visualizado acima.</p>
# <p>Além do que foi pedido, foram contabilizados 275 filmes que não possuem nenhum gênero associado. Em sistemas de busca de informação, a falta desta informação pode tornar determinados filmes difíceis de serem encontrados. Saber este valor pode ser interessante para corrigir isso na base de dados.</p>

# <h2>11) Qual o diretor possui a melhor media de nota IMDb dos seus filmes?</h2>

# In[669]:


#Diretores
diretores = {}
diretores_cont = {}
for index, row in df.iterrows():
    if(pd.isnull(row['Diretores'])):
        if('NaN' in diretores):
            diretores['NaN'] += row['IMDb']
            diretores_cont['NaN'] += 1
        else:
            diretores['NaN'] = row['IMDb']
            diretores_cont['NaN'] = 1
        continue
    v = row['Diretores'].split(',')
    for k in v:
        if(k in diretores):
            diretores[k] += row['IMDb']
            diretores_cont[k] += 1
        else:
            diretores[k] = row['IMDb']
            diretores_cont[k] = 1
for k in diretores:
    diretores[k] = diretores[k] / diretores_cont[k]


# In[670]:


diretores_ordenados = {k: v for k, v in sorted(diretores.items(), reverse=True, key=lambda x: x[1])}
print('Imprimindo os 20 diretores com maiores médias de nota IMDb:')
cont = 0
for k in diretores_ordenados:
    cont+=1
    print("{:<25} {:<20}".format(k, diretores_ordenados[k]))
    if(cont >= 20):
        break
print('...')


# <p>O diretor que possui a melhor média de notas de seus filmes é <b>Sergio Leone</b>, com 8.65 de média.</p>

# <h2>12) Utilize regras de associação para responder às seguintes questões:</h2>
# <h3>a) Qual o conjunto de itens de gêneros com mais de um gênero que aparece em mais filmes?</h3>

# In[671]:


df4 = df.copy().dropna()
df4['qtd_filmes'] = 1
df4 = df4[df4['Generos'].str.contains(',')]
df4 = df4.groupby('Generos')
df4['qtd_filmes'].sum().sort_values(ascending=False).head(3)


# <p>O conjunto de itens de gêneros com mais de um gênero que aparece em mais filmes é: <b>{Comedy, Drama}</b></p>

# <h3>b) Qual o percentual de filmes em que esse conjunto de itens aparece?</h3>

# In[672]:


p = df4['qtd_filmes'].sum().sort_values(ascending=False).head(1)/len(df)*100 
print('Percentual de filmes em que o conjunto de itens {Comedy, Drama} aparece:')
p


# <p>O percentual de filmes que o conjunto {Comedy, Drama} aparece é de <b>1,24%</b>.</p>

# <h3>c) Considerando uma pessoa que gosta de filmes de "Crime", quais outros gêneros você recomendaria  a essa pessoa? Justifique sua resposta.</h3>

# In[673]:


df2 = df.dropna()
df2 = df2[df2['Generos'].str.contains(',')]
df2 = df2[['Generos']]
for k in generos:
    df2[k] = 0
for i, row in df.iterrows():
    if(pd.isnull(row['Generos'])):
        continue
    v = row['Generos'].split(',')
    for k in v:
        df2.at[i,k] = 1
df2 = df2.drop('Generos', 1)


# In[674]:


df3 = df2.copy()
df3 = df3.dropna()
df3.head(15)


# In[675]:


frequent_itemsets = apriori(df3, min_support=0.06, use_colnames=True)


# In[676]:


rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)


# In[677]:


rules


# In[678]:


rules[rules['support'] == rules.support.max()]


# In[679]:


rules[rules['confidence'] == rules.confidence.max()]


# In[680]:


rules[rules['lift'] == rules.lift.max()]


# <p>Observando as regras acima, é possível perceber que pessoas que assistem filmes com o gênero "Crime", também assistem aos filmes de gênero <b>Drama e Thriller</b>. A confiança para com estes valores é de mais de 50% e o número de suporte pode se considerar aceitável para esta análise. Desta forma, eu recomendaria a essa pessoa a assistir filmes de Drama e Thriller.</p>
