from SwarmOptimisation_NRN_OneRune_Mooving_SL_marging_cleancode_func import *
import datetime as dt
from datetime import datetime
import numpy as np
import pandas as pd
import time
from datetime import date

#DateStart, DateEnd = '2020-05-01 00:00:00', '2020-10-15 00:00:00' #Sideway
#DateStart, DateEnd = '2020-09-10 00:00:00', '2021-03-01 00:00:00' #Uptrend2
#DateStart, DateEnd = '2018-12-01 00:00:00', '2019-07-01 00:00:00' #Uptrend1


DateStart, DateEnd = '2017-12-08 00:00:00', '2018-11-12 00:00:00' #DownTrend1
#DateStart, DateEnd = '2017-12-08 00:00:00', '2019-02-01 00:00:00' #DownTrend15
#DateStart, DateEnd = '2019-07-01 00:00:00', '2020-03-01 00:00:00' #DownTrend2

#DateStart, DateEnd = '2019-07-01 00:00:00', '2020-01-01 00:00:00' #backtest downtrend
#DateStart, DateEnd = '2019-01-15 00:00:00', '2020-01-01 00:00:00' #back test up and down


#DateStart, DateEnd = '2017-12-15 00:00:00', '2019-10-01 00:00:00' #backtest2
#DateStart, DateEnd = '2017-12-15 00:00:00', '2020-10-01 00:00:00' #backtest2

#DateStart, DateEnd = '2017-12-08 00:00:00', '2021-03-01 00:00:00' #Full 3 years
#DateStart, DateEnd = '2020-05-12 00:00:00', '2021-02-01 00:00:00' #Stop loss test


Signals_tot = pd.read_csv('./Signal_reconstruit.txt', sep="\t")

##Je détermine ici les indices d'itération pour parcourir le fichier Date. Par exemple, la date 2017-12-08 00:00:00 correspond à l'indice 2466
# et la date '2018-03-12 00:00:00 à l'indice 4722. Lors du calcul, la boucle for parcourera la liste Signals_tot[2466:4722]
Date = Signals_tot['Date'].tolist()
Ite_start, Ite_Stop = IterationForDate(Date,DateStart,DateEnd)

#Paramètres initiaux du portefeuille
USD_init = 200
BTC = 0
USD = USD_init
fee = 0.001
NbTxPerWeekRequested = 3
ErrAccepted = 2

##Parameètres devant avoir une valeur pour le tout début du calcul 
bth = 0.75          #bth sera déterminer ensuite comme étant le max du stimulu sur la période delayS; mulitplié par thcoeff
sth = -0.75         #bth sera déterminer ensuite comme étant le min du stimulu sur la période delayS; mulitplié par thcoeff

##Parametres à optimiser
#Rewarding rules
StimuliUp = 0.756             #La valeur que gagne chaque régle à chaque vfois qu'elle prédit correctement la bonne tendance (note maximal = 100).
StimuliDown = -0.756          #La valeur que perd chaque régle à chaque fois qu'elle prédit mal tendance (note minimal = 0).
TreshStart = -39.4            #La note à laquelle la regle débute après avoir frenchit le treshold de trigger (trigger fixé à 100)
shift = 62                    #Plateau du signal sigmoidal
tau = 134                     #Constante de temps lors de la décroissance du signal sigmoidal

#Traitement des stimulus pour positionnement
Integral = 46                 #Période sur laquelle le stimulus est intégré pour la comparaison avec les treshold de positionnement (sth et bth)
delayS = 72                   #Retard imposer à l'intégral du stimulus lors de la comparaison avec les treshold de positionemment (sth et bth)
thcoeff = 0.762               #Modificateur qui vient multiplier le max/min du stimulus, determinant les nouvelles valeurs de bth et sth

#Stop loss
delaySL = 3                   #Période sur laquelle on va venir calculé la variation relative du cours du bitcoin pour faire agir le StopLoss
Sl_val = 0.094                #Treshold que doit atteidre la variation relative du cours du bitcoin pour pass-by le trading bot.

IntegralBool = 5              #Période sur laquelle on va venir calculé l'écart type des Highs et Lows du cours du bitcoin (estimation de la volatilité)
CoeffBool = 2                 #Nombre d'écart type pris en compte pour déterminer le treshold du stop loss.
DelayBool = 2                 #Retard imposé à la volatilité lors de la comparaision avec le cours du bitcoin.

#Marging effect
SizeMargin = 0.2              #Pourcetage du capital du trading bot pris pour rélaiser le marging
leverage = 5                  #levier utiliser lors du marging

#######
#Initialisation des paramètres initiaux.
#Constante
CurrVal = list(Signals_tot['Close'])[Ite_start]
CurrDate = list(Signals_tot['Date'])[Ite_start]
NbRules = len(Signals_tot.columns.values)-4
Amp = (1/(1+2.718**((-shift)/tau)))
BorrowVal = CurrVal
NbIteration = abs(Ite_start-Ite_Stop)
BTC_init = USD_init/CurrVal

#Variables incrémentées
USDfee = 0
Transaction = 0
GainUSDSurBTC = 0
Nb_liquidation = 0
margingfailed = 0
margingsucceed = 0
USDMargin = 0
Marging = 0
SL = 0
USDBorrow = 0
S_raw = 0

#Création de liste
bth_list = list()
sth_list = list()
Val_list = list()
USDMargin_list = list()
Marging_list = list()
High_list = list()
ValTreshTop_list = list()
ValTreshBot_list = list()
Vall_list = list()
Price_list = list()
Stimulus_list = list()
USDtotlist = list()
USDHOLD = list()
CurrValprevious = list()
USDratio = list()
S_int_list = list()

Stimulus_raw_list = list()

#Création des arrays
IaRules = np.zeros(shape=NbRules)
CallRule = np.zeros(shape=NbRules)
Crules = np.zeros(shape=(NbIteration, NbRules))
Wrules = np.zeros(shape=(NbIteration, NbRules))
WrulesNonNorma = np.zeros(shape=(NbIteration, NbRules))
profit = np.zeros(shape=NbRules)
USDRules = np.zeros(shape=NbRules)
BTCRules = np.zeros(shape=NbRules)
BTCRules[:] = np.array([BTC_init for i in range(NbRules)])


start = time.time()
for ite in range(Ite_start, Ite_Stop):
    #On vient prendre la ligne des signaux correspondant au temps regardé
    Signals = Signals_tot.iloc[[ite]]   
    CurrDate, CurrVal, HighVal, LowVal = Signals.values[0][0:4]    
    Signal_array = Signals.values[0][4:NbRules+4]
    Price_list.append(CurrVal)
    if ite == Ite_start:
        CurrVal_init = CurrVal
        BTC_init = USD_init/CurrVal_init     

    ##Ici on maintient une longueur de "delaySL" une liste des valeurs du bitcoin. En effet, les variations utilisés pour 
    #déclencher le stop loss qui short circuit le trading bot sera déterminer par la viaration entre le premier 
    #élement de cette liste et le dernire.
    Vall_list.append(CurrVal)    
    if ite-Ite_start > delaySL:        
        del Vall_list[0]    
    ##Dans cette partie sont détemrinés la méthode de selection des signaux.
    #   1) On définit le profit de chaque regle. Si la regle Hold du BTC, alors, le profit se détermine comme la différence
    #du cour du BTC entre i et i+1. BTC augment, le profit et positif. BTC diminue, le profit est négatif.
    #Si je vend du BTC, le profit se détermine comme la difference du cour du BTC entre i+1 et i. Si le BTC augmente, le profit
    #est négatif, si le BTC diminue, le profit est positif. L'amplitude du profit n'est pas pris en compte dans cette méthode.
    #   2)On actualise ensuite les prodictions de chaque regle pour la prochain itération et leurs demande de prendre position.
    #Le cours sera t il haussier ? Si oui et que je suis en USD alors j'achete du BTC. Sinon je garde ma position. 
    #Le cours sera t il baissier ? Si oui et que je suis en BTC alors je vend du BTC. Sinon je garde ma position
    #   3) Suivant les profits, on détermine si les "note d'état" des reglès (Crules) doivent être incrémenté ou décrémenté (la regle à
    # eu raison +1, la regle à eu faux -1).On regarde ensuite les notes d'état de chaque regle. Si elles atteignent une
    #valeur treshold, alors le coefficient de pondération de la regle (Wrule) est trigger et suit une évolution sigmoidale
    #au cours des prochaines itérations.
    #   4) Une fois les coefficients de pondération de toutes les regles calculés, on peut ensuite calculés le stimulus
    #de sortie du neurone. Chaque signal de regle est compris entre -1 et 1, il faut donc normalisés les signaux d'
    #entré afin d'avoir un signal de sorti du neuron compris également entre -1 et 1.    
    #   5) On applique des filtres sur le stimulus.


    # 1) On calcul le profit de chaque regle
    DelayProfit = 2 #Itération de l'heure précédente
    CurrValprevious = Vall_list[len(Vall_list)-DelayProfit]  #CurrValPrevious est ici la valeur de l'itération i-DelayProfit
    profit = np.where(BTCRules[:] != 0, CurrVal-CurrValprevious, profit)    #Je hold du BTC
    profit = np.where(USDRules[:] != 0, -CurrVal+CurrValprevious, profit)   #Je hold du USD

    # 2) On actualise les predictions de chaque regles et calcul leurs profits
    for r in range(NbRules):
        if Signal_array[r] > 0 and BTCRules[r] == 0:
            BTCRules[r] = (USDRules[r]/CurrVal)*(1-fee)
            USDRules[r] = 0
        if Signal_array[r] < 0 and USDRules[r] == 0:                 
            USDRules[r] = (BTCRules[r]*CurrVal)*(1-fee)
            BTCRules[r] = 0    
    I = ite-Ite_start

    # 3) On regarde les performances de chaque regles et on incrémente le trigger
    tick = np.array([np.sign(i) for i in profit])   
    tick = np.where(tick <= -1, StimuliDown, tick)
    tick = np.where(tick >= 1, StimuliUp, tick) 
    Crules[ite-Ite_start][:] = Crules[ite-Ite_start-1][:] + tick 
    Sig = ([(1/(1+2.718**((I-i-shift)/tau))) for i in IaRules])
    Wrules[ite-Ite_start][:] = Sig
    Wrules[ite-Ite_start][:] = np.where(Crules[ite-Ite_start][:] >= 100, Amp, Wrules[ite-Ite_start][:])
    IaRules = np.where(Crules[ite-Ite_start][:] >= 100, ite-Ite_start, IaRules)
    Crules[ite-Ite_start][:] = np.where(Crules[ite-Ite_start][:] >= 100, TreshStart, Crules[ite-Ite_start][:])
    Crules[ite-Ite_start][:] = np.where(Crules[ite-Ite_start][:] <= 0, 0, Crules[ite-Ite_start][:])
    
    # 4) On calcul le stimulus               
    Wrules[ite-Ite_start,:] = Wrules[ite-Ite_start,:]/(Wrules[ite-Ite_start,:].sum()+0.0001) ## le +0.0001 sert uniquement à ne peut avoir une division par 0 quand la sum est nul
    Stimulus = np.array(Signals.values[0][4:NbRules+4]) * Wrules[ite-Ite_start,:]
    

    TradingBotMode, S_raw, SL = StopLossMecanism(Vall_list, Sl_val, tau, S_raw)
    TradingBotMode, S_int, SL = StimulusFiltre(TradingBotMode, Stimulus, Stimulus_list, Integral, SL,S_raw)
    if len(S_int_list) > delayS:
        bth, sth = Threshold_BuySell(S_int_list, delayS, True, thcoeff)
    bth_list.append(bth)    
    sth_list.append(sth)    
    S_int_list.append(S_int)
    S = S_int     
        
    #On achete
    #Le stimulus est supérieur au treshold d'achat et nous étions en train de shorter
    if S > bth and BTC == 0:
        BTC = ((USD+USDMargin)/CurrVal)*(1-fee)     #On converti en BTC la totalité des USD (USD classique + USD Margin)  
        USD = 0                                     #On met le portfeuoille USD à 0 car tout est vendu        
        Marging += (USDMargin-USDBorrow)            #Pour les logs, On ajoute les gains/pertes générés par le marging
        USDBorrow = 0                               #On réinitialise le portfeuille USD Borrow car le marging est fini
        USDMargin = 0                               #Pour les logs, On remet le log USD margin à 0 car tout est vendu        
        USDfee += USD*fee                           #Pour les logs, on stock en mémoire l'argent perdu en fees   
        Transaction += 1                            #Pour les logs, on stock en mémoire le nombre de tx 
    #On vend
    #Le stimulus est inférieur au treshold de vente et nous étions en train de Hold
    if S < sth and (USD+USDMargin) == 0:
        USD = (BTC*CurrVal)*(1-fee)                 #On converti en USD la totalité des BTC
        BTC = 0                                     #On met le portfeuoille BTC à 0 car tout est vendu
        USDBorrow = SizeMargin*USD                  #On utilise une fraction du capital (SizeMargin) pour effectuer du margin trade, on alloue cette somme comme gage de perte
        USD = (1-SizeMargin)*USD                    #L'autre partie est conservé sous forme de USD
        BorrowVal = CurrVal                         #On recupere la valeur à laquelle le margin a été initié pori la suite des caclculs

        USDfee += BTC*CurrVal*fee                   #Pour les logs, on stock en mémoire l'argent perdu en fees                 
        Transaction += 1                            #Pour les logs, on stock en mémoire le nombre de tx 

    
    #On calcul ici l'évolution du portgefeuoille USD marging et vérie si notre position est liquidée.

    #USD margine se définie comme étant l'argenrt mis en gage (USD Borrow) + les gains/pertes générés par le contrat marging, fois le levier.
    #Si CurrVal > BorrowVal alors, leverage*(((USDBorrow/CurrVal) - (USDBorrow/BorrowVal))*CurrVal) est négatif. USD Marging est donc inférieur à USDBorrow : Nous sommes perdant
    #Si CurrVal < BorrowVal alors, leverage*(((USDBorrow/CurrVal) - (USDBorrow/BorrowVal))*CurrVal) est positif. USD Marging est donc supérieur à USDBorrow : Nous sommes gagnant 
    USDMargin = USDBorrow + leverage*(((USDBorrow/CurrVal) - (USDBorrow/BorrowVal))*CurrVal) #Expression pour du marging en short
    #Si USD Marging est inférieur ou égal à 0, cela signifie que les pertes sont égales au montant USD engagé (USD borrow). Dans ce cas, la position doit 
    #être liquidée. 
    if USDMargin < 0:
        Marging += (USDMargin-USDBorrow)    #Pour les logs, on tock en mémoire les gains fait par marging
        USDBorrow = 0                       #On réinitialise USD Borrow
        USDMargin = 0                       #On réinitialise le portfeuille USD Marging (Nous venons donc de perdre USD borrow sur le capital)
        Nb_liquidation += 1                 #Pour les logs, on stock en mémoire le nombre de liquidation

    USDtot = USDMargin + USD + BTC*CurrVal   
    USDBTC = USD_init*(1+(CurrVal-CurrVal_init)/CurrVal_init)
    GainUSDSurBTC = 100*(USDtot-USDBTC)/USDBTC

    Modificateur = Transaction**(1/3)               #Le modificateur permet de prendre en compte le Nb de tx dans l'indice du swarm. Ici on choisit la racine cubique
    SwarmIndicator = GainUSDSurBTC*Modificateur

    USDHOLD.append(USDBTC)
    USDtotlist.append(USDtot)  
    Marging_list.append(Marging)
    USDMargin_list.append(USDMargin)

end = time.time()
Time = end-start
TimePerIte = 1000*Time/ite
print('Time elapsed : {} s,\t Time per ite : {} ms, \t Gain sur BTC : {} %'.format("%.2f" % Time, "%.1f" % TimePerIte, "%.2f" % GainUSDSurBTC))