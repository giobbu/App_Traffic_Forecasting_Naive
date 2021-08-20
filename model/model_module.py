
import numpy as np





def naive(history, seasonality, n_seq):

    list_seas = []
    
    # seas_0 = 168*2*2*2+23
    seas_1 = 168*2*2+168*2
    seas_2 = 168*2*2 
    seas_3 = 168*2 
    

    
    list_seas_0 = []
    list_seas_1 = []
    list_seas_2 = []
    list_seas_3 = []
    
    # if seasonality > history.shape[0]:
        
    #     for i in reversed(range(1,24*2+23 +1)):
    #         season = history[-i][-1]
    #         list_seas.append(season)
    #         array_pred = np.vstack(list_seas)
    # else:
        
    # for i in reversed(range(seas_1+1,seas_0+1)):
    #     season_0 = history[-i][-1]
    #     list_seas_0.append(season_0)
        
    for i in reversed(range(seas_2+1,seas_1+1)):
        season_1 = history[-i][-1]
        list_seas_1.append(season_1)
        
    for i in reversed(range(seas_3+1,seas_2+1)):
        season_2 = history[-i][-1]
        list_seas_2.append(season_2)
        
    for i in reversed(range(1,seas_3+1)):
        season_3 = history[-i][-1]
        list_seas_3.append(season_3)
        
#  np.vstack(list_seas_0),
#  np.vstack(list_seas_1),
    array_pred = np.array([np.vstack(list_seas_1),
                            np.vstack(list_seas_2),
                           np.vstack(list_seas_3)]).mean(axis=0)
            
    return array_pred[:n_seq]










