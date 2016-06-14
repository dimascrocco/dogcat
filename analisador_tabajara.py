import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import IPython.lib.display as display 

import mir3.modules.tool.wav2spectrogram as spec
import mir3.modules.features.centroid as cent
import mir3.modules.features.energy as energ
import mir3.modules.features.rolloff as roll
import mir3.modules.features.flux as specfl
import mir3.modules.features.flatness as flatn
import mir3.modules.features.low_energy as lowen

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

import glob

def feature(sound, feature, plot=False, show=False, xlabel="Tempo (s)", ylabel=""):
    if isinstance(feature, lowen.LowEnergy):
        ft = feature.calc_track(sound, 10) # default is 40 (deveria...)
    else:
        ft = feature.calc_track(sound)

    if plot:
        T = ft.metadata.sampling_configuration.ofs
        t = np.linspace(0, len(ft.data)/T, len(ft.data))
        plt.plot(t, ft.data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.savefig(target + ".png")
        if show:
            plt.show()

    return ft

def vetor(data):
    # min/max/mean/var
    #return  [np.min(data), np.max(data), np.mean(data), np.var(data)] # len(data)]
    return  [np.mean(data), np.var(data)] # len(data)]

def processar(target, features, plot=False):
    #print "- ", target
    wav2spec = spec.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    som = wav2spec.convert(open(target, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    ro=fe=ce=en=ft=le=None
    f = features
    if f[0]: ro = feature(som, roll.Rolloff(), plot=plot, ylabel="Rolloff"); 
    if f[1]: fe = feature(som, specfl.Flux(), plot=plot, ylabel="Spectral Flux"); 
    if f[2]: ce = feature(som, cent.Centroid(), plot=plot, ylabel="Centroid"); 
    if f[3]: en = feature(som, energ.Energy(), plot=plot, ylabel="Energy"); 
    if f[4]: ft = feature(som, flatn.Flatness(), plot=plot, ylabel="Flatness"); 
    if f[5]: le = feature(som, lowen.LowEnergy(), plot=plot, ylabel="Low Energy"); 

    result = []
    if ro: vetor_ro = vetor(ro.data); result += vetor_ro
    if fe: vetor_fe = vetor(fe.data); result += vetor_fe
    if ce: vetor_ce = vetor(ce.data); result += vetor_ce
    if en: vetor_en = vetor(en.data); result += vetor_en
    if ft: vetor_ft = vetor(ft.data); result += vetor_ft
    if le: vetor_le = vetor(le.data); result += vetor_le

    return result

def plot_spread(x_gatos, x_caes, title):
    mean_gatos = [v[0] for v in x_gatos]
    var_gatos = [v[1] for v in x_gatos]
    mean_caes = [v[0] for v in x_caes]
    var_caes = [v[1] for v in x_caes]

    plt.figure()
    plt.title(spread_title)
    plt.scatter(mean_gatos, var_gatos, color='red');
    plt.scatter(mean_caes, var_caes, color='blue');
    plt.ylabel('Media');
    plt.xlabel('Variancia');
    plt.legend(['Gatos', 'Caes'], loc=4);
    plt.show()




# CABINE DE CONTROLE... nao pressione o botao vermelho!
gatos = glob.glob("db/cats/*.wav")
caes = glob.glob("db/dogs/*.wav")
PLOT_MULTIPLO = False # plota todas as amostras de uma vez

# Features a serem utilizadas (utilize uma por vez, multiplas podem exibir graficos sobrepostos)
f_rolloff = 0
f_spectral_flux = 0
f_centroid = 0
f_energy = 0
f_flatness = 0
f_low_energy = 1

features = [f_rolloff, f_spectral_flux, f_centroid, f_energy, f_flatness, f_low_energy]
features_labels = ['Rolloff','Spectral Flux','Centroid','Energy','Flatness','Low Energy']

# TODO da pra fazer isso em uma linha? converter lista em mapa indexado?
spread_title = []
count = 0
for v in features:
    if v:
        spread_title.append(features_labels[count])
    count += 1
spread_title = ' + '.join(spread_title)


# resultados
n_gatos = len(gatos)
n_caes = len(caes)
X_gatos = []
X_caes = []
y_gatos = [] 
y_caes= [] 


print "* processando gatos..."
for som in gatos[0:n_gatos]:
    result = processar(som, features, plot=PLOT_MULTIPLO)
    X_gatos.append(result)
    y_gatos.append('G')

if PLOT_MULTIPLO:
    plt.show()
    plt.clf()

print "* processando caes..."
for som in caes[0:n_caes]:
    result = processar(som, features, plot=PLOT_MULTIPLO)
    X_caes.append(result)
    y_caes.append('C')

# one big list to bind them all...
X = X_gatos + X_caes
y = y_gatos + y_caes

if PLOT_MULTIPLO:
    plt.show()

if 1 > 0:
    plot_spread(X_gatos, X_caes, spread_title)


#i = 0
#print "label feature-vector"
#for r in X:
#    print y[i], r # label, feature vector
#    i += 1

if False:
    neigh = KNeighborsClassifier(n_neighbors=3); # n_neighbors = K
    dados_treino, dados_teste, label_treino, label_teste = train_test_split(X, y, train_size=.5)
    #neigh.fit(X_gatos + X_caes, y_gatos + y_caes)
    neigh.fit(dados_treino, label_treino)
    score = neigh.score(dados_teste, label_teste)
    print "% acertos: ", 100*score


# Parametros para executar busca exaustiva
train_size_min = 0.2
train_size_max = 0.95
train_size_step = 0.05

# Numero de iteracoes para cada tamanho de conjunto de treino
n_iter = 1000

# Listas que armazenarao os resultados
steps = []
medias = []
variancias = []

train_size_atual = train_size_min
while train_size_atual <= train_size_max: # para cada tamanho do conjunto de treino
    acertos = []
    for k in xrange(n_iter): # para cada iteracao do processo Monte Carlo
        neigh = KNeighborsClassifier(n_neighbors=5); # n_neighbors = K
        dados_treino, dados_teste, label_treino, label_teste = train_test_split(X, y, train_size=train_size_atual)
        neigh.fit(dados_treino, label_treino);
        score = neigh.score(dados_teste, label_teste);
        acertos.append(score)
    
    steps.append(train_size_atual)
    medias.append(np.mean(np.array(acertos)))
    variancias.append(np.std(np.array(acertos)))
    
    train_size_atual += train_size_step


plt.figure();
plt.errorbar(steps, medias, yerr=variancias);
plt.title(spread_title)
plt.ylabel('Indice de acertos');
plt.xlabel('Tamanho do conjunto de treino');
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.axvspan(0.475, .525, facecolor='g', alpha=0.5)

plt.show()

# MODO MANUAL
#print "# testes"
#print "# gatos"
#for som in gatos[n_gatos/2:n_gatos]:
#    #print som
#    result = processar(som, features, plot=False)
#    print "--> ", neigh.predict([result])
#
#print "# caes"
#for som in caes[n_caes/2:n_caes]:
#    #print som
#    result = processar(som, features, plot=False)
#    print "--> ", neigh.predict([result])


    

