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

def processar(target, features):
    wav2spec = spec.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    som = wav2spec.convert(open(target, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    ro=fe=ce=en=ft=le=None
    f = features
    if f[0]: ro = feature(som, roll.Rolloff(), plot=True, ylabel="Rolloff"); 
    if f[1]: fe = feature(som, specfl.Flux(), plot=True, ylabel="Spectral Flux"); 
    if f[2]: ce = feature(som, cent.Centroid(), plot=True, ylabel="Centroid"); 
    if f[3]: en = feature(som, energ.Energy(), plot=True, ylabel="Energy"); 
    if f[4]: ft = feature(som, flatn.Flatness(), plot=True, ylabel="Flatness"); 
    if f[5]: le = feature(som, lowen.LowEnergy(), plot=True, ylabel="Low Energy"); 

    result = []
    if ro: vetor_ro = vetor(ro.data); result += vetor_ro
    if fe: vetor_fe = vetor(fe.data); result += vetor_fe
    if ce: vetor_ce = vetor(ce.data); result += vetor_ce
    if en: vetor_en = vetor(en.data); result += vetor_en
    if ft: vetor_ft = vetor(ft.data); result += vetor_ft
    if le: vetor_le = vetor(le.data); result += vetor_le

    return result


# SCRIPT #
gatos = glob.glob("db/cats/*.wav")
caes = glob.glob("db/dogs/*.wav")

# Features a serem utilizadas
f_rolloff = 1
f_spectral_flux = 0
f_centroid = 0
f_energy = 0
f_flatness = 0
f_low_energy = 0

features = [f_rolloff, f_spectral_flux, f_centroid, f_energy, f_flatness, f_low_energy]
features_labels = ['Rolloff','Spectral Flux','Centroid','Energy','Flatness','Low Energy']

spread_title = []
count = 0
for v in features:
    if v:
        spread_title.append(features_labels[count])
    count += 1
spread_title = ' + '.join(spread_title)

print spread_title 

n_gatos = len(gatos)
n_caes = len(caes)

# valores
X_gatos = []
X_caes = []

# labels
y = [] 


print "# GATOS"
for som in gatos[0:n_gatos]: #/2]:
    print som
    result = processar(som, features)
    X_gatos.append(result)
    y.append('G')

plt.show()
plt.clf()

print "# CAES"
for som in caes[0:n_caes]: #/2]:
    print som
    result = processar(som, features)
    X_caes.append(result)
    y.append('C')

plt.show()
PLOT_SPREAD = True
if PLOT_SPREAD:
    mean_gatos = [v[0] for v in X_gatos]
    var_gatos = [v[1] for v in X_gatos]
    mean_caes = [v[0] for v in X_caes]
    var_caes = [v[1] for v in X_caes]

    plt.figure()
    plt.title(spread_title)
    plt.scatter(mean_gatos, var_gatos, color='red');
    plt.scatter(mean_caes, var_caes, color='blue');
    plt.ylabel('Media');
    plt.xlabel('Variancia');
    #plt.xlim([60, 120]);
    #plt.ylim([1.6, 2.2]);
    plt.legend(['Gatos', 'Caes'], loc=4);
    plt.show()



X = X_gatos + X_caes
i = 0
print "label feature-vector"
for r in X:
    print y[i], r # label, feature vector
    i += 1

#print "Global min/max:", np.min(m), np.max(M)
#print np.unique(map(len, X))

neigh = KNeighborsClassifier(n_neighbors=3); # n_neighbors = K
neigh.fit(X_gatos + X_caes, y)

acertos_gatos = 0
acertos_caes = 0

print "# testes"
print "# gatos"
for som in gatos[n_gatos/2:n_gatos]:
    #print som
    result = processar(som, features)
    print "--> ", neigh.predict([result])

print "# caes"
for som in caes[n_caes/2:n_caes]:
    #print som
    result = processar(som, features)
    print "--> ", neigh.predict([result])


    

