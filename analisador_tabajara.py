import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import IPython.lib.display as display 

import mir3.modules.tool.wav2spectrogram as spec
import mir3.modules.features.centroid as cent
import mir3.modules.features.energy as energ
import mir3.modules.features.rolloff as roll
import mir3.modules.features.flux as specfl

from sklearn.neighbors import KNeighborsClassifier
import glob

def energia(sound, plot=False):
    en = energ.Energy()
    energy = en.calc_track(sound)

    if plot:
        T = energy.metadata.sampling_configuration.ofs
        t = np.linspace(0, len(energy.data)/T, len(energy.data))
        plt.plot(t, energy.data)
        plt.xlabel('Tempo (s) ')
        plt.ylabel('Energia')
        #plt.savefig(target + ".png")
        plt.show()

    return energy

def centroide(sound, plot=False):
    centr = cent.Centroid()
    centroid = centr.calc_track(sound)

    if plot:
        T = centroid.metadata.sampling_configuration.ofs
        t = np.linspace(0, len(centroid.data)/T, len(centroid.data))

        plt.plot(t, centroid.data)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Centroide espectral')
        #plt.savefig(target + ".png")
        plt.show()
    
    return centroid

def fluxo_espectral(sound, plot=False):
    fl = specfl.Flux()
    flux = fl.calc_track(sound)

    if plot:
        T = flux.metadata.sampling_configuration.ofs
        t = np.linspace(0, len(flux.data)/T, len(flux.data))

        #h = np.log10(flux.data/np.max(flux.data))
        #tg = []
        #g = []
        #for i in xrange(len(h)-2):
        #    if (h[i+1] > h[i]) and (h[i+1] > h[i+2]): # Condicao 1: eh um pico
        #        if (h[i+1] > -.5): # Condicao 2: magnitude acima de um limiar
        #            g.append(h[i+1])
        #            tg.append(t[i+1])
        #plt.plot(t, h)
        #plt.plot(tg, g, 'ro')

        plt.plot(t, flux.data)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Fluxo Espectral')
        plt.show()

    return flux

def rolloff(sound, plot=False):
    roff = roll.Rolloff()
    roll_off = roff.calc_track(sound)

    if plot:
        T = roll_off.metadata.sampling_configuration.ofs
        t = np.linspace(0, len(roll_off.data)/T, len(roll_off.data))
        plt.plot(t, roll_off.data)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Roll Off espectral')
        #plt.show()

    return roll_off

def calcular_vetor(data):
    # min/max/mean/var
    #return  [np.min(data), np.max(data), np.mean(data), np.var(data)] # len(data)]
    return  [np.mean(data), np.var(data)] # len(data)]

def processar(target):
    wav2spec = spec.Wav2Spectrogram() # Objeto que converte arquivos wav para espectrogramas
    som = wav2spec.convert(open(target, 'rb'), window_length=1024, window_step=512, spectrum_type='magnitude')

    ro = rolloff(som)
    fe = fluxo_espectral(som)
    ce = centroide(som)
    en = energia(som)

    vetor_ro = calcular_vetor(ro.data)
    vetor_fe = calcular_vetor(fe.data)
    vetor_ce = calcular_vetor(ce.data)
    vetor_en = calcular_vetor(en.data)

    #return vetor_ro 
    #return vetor_fe # bom
    #return vetor_ce # bom
    #return vetor_en
    #return vetor_ro + vetor_fe + vetor_ce + vetor_en
    return vetor_fe + vetor_ce


# SCRIPT #
gatos = glob.glob("db/cats/*.wav")
caes = glob.glob("db/dogs/*.wav")

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
    result = processar(som)
    X_gatos.append(result)
    y.append('G')

#plt.show()
#plt.clf()

print "# CAES"
for som in caes[0:n_caes]: #/2]:
    print som
    result = processar(som)
    X_caes.append(result)
    y.append('C')

#plt.show()
PLOT_SPREAD = True
if PLOT_SPREAD:
    mean_gatos = [v[0]for v in X_gatos]
    var_gatos = [v[1] for v in X_gatos]
    mean_caes = [v[0] for v in X_caes]
    var_caes = [v[1] for v in X_caes]

    plt.figure()
    plt.title('')
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
    result = processar(som)
    print "--> ", neigh.predict([result])

print "# caes"
for som in caes[n_caes/2:n_caes]:
    #print som
    result = processar(som)
    print "--> ", neigh.predict([result])


    

