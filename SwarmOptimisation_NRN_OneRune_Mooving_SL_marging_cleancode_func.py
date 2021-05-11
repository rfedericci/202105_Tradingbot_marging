import numpy as np
import pandas as pd


def IterationForDate(Date,DateStart,DateEnd):
    for i in range(len(Date)):
        if Date[i] == DateStart:
            Ite_start = i   
            break       
    for i in range(Ite_start, len(Date)):
        if Date[i] == DateEnd:
            Ite_Stop = i
            break
    return(Ite_start, Ite_Stop)


def StopLossMecanism(Vall_list, Sl_val, tau, S_raw):
    #On regarde si le stop loss est déclenché. Si le stopp loss est déclenché alors le stimulus Sbis est forcé de prendre
    #la valeur -1 ou 1 suivant la variation soudaine du cours du BTC (achat ou vente). Et un delai (SL) avant réactivition 
    #du trading bot est mis en place. Ce délai est fixé comme étant 0.75 fois tau.
    ValEnd = Vall_list[len(Vall_list)-1]
    ValStart = Vall_list[0]
    if (ValEnd-ValStart)/ValStart > Sl_val:
        S_raw = 1
        SL = 0.75*tau
        return False, S_raw, SL
    elif (ValEnd-ValStart)/ValStart < -Sl_val:
        S_raw = -1
        SL = 0.75*tau
        return False, S_raw, SL
    else:
        return True, S_raw, 0

def Threshold_BuySell(Sbis_list, delayS, LowCap, thcoeff):
    if LowCap == True:
        bth = max(thcoeff*max(Sbis_list[len(Sbis_list)-delayS:len(Sbis_list)-1]),0.1)
        sth = min(thcoeff*min(Sbis_list[len(Sbis_list)-delayS:len(Sbis_list)-1]),-0.1)
    else:
        bth = max(thcoeff*max(Sbis_list[len(Sbis_list)-delayS:len(Sbis_list)-1]))
        sth = min(thcoeff*min(Sbis_list[len(Sbis_list)-delayS:len(Sbis_list)-1]))
    return bth, sth

def StimulusFiltre(TradingBotMode, Stimulus, Stimulus_list, Integral, SL, S_raw):
    if TradingBotMode == True: #Le trading bot peut s'exprimer
        S_raw = np.array(Stimulus).sum()          
    else: #On attend et décrémente le temps d'attente de 1, Sbis reste fixé à 1 ou -1
        SL -= 1        
        if SL <= 0: #Une fois que le temps d'attente SL est nul alors le trading bot peut reprendre son acitivité
            TradingBotMode = True  

    Stimulus_list.append(S_raw)
    LL = len(Stimulus_list)   
    if LL > Integral:
        S_int = np.array(Stimulus_list[LL-Integral:LL-1]).mean()
    else:
        S_int = np.array(Stimulus_list).mean()        
    return(TradingBotMode, S_int, SL)



