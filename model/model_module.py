
import numpy as np


def naive(history, seasonality, n_seq):

    seas_1 = 168*2*2+168*2
    seas_2 = 168*2*2 
    seas_3 = 168*2 

    list_seas_1 = []
    list_seas_2 = []
    list_seas_3 = []
        
    for i in reversed(range(seas_2+1,seas_1+1)):
        season_1 = history[-i][-1]
        list_seas_1.append(season_1)
        
    for i in reversed(range(seas_3+1,seas_2+1)):
        season_2 = history[-i][-1]
        list_seas_2.append(season_2)
        
    for i in reversed(range(1,seas_3+1)):
        season_3 = history[-i][-1]
        list_seas_3.append(season_3)
        
    array_pred = np.array([np.vstack(list_seas_1),
                           np.vstack(list_seas_2),
                           np.vstack(list_seas_3)]).mean(axis=0)
            
    return array_pred[:n_seq]










