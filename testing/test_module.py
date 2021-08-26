from altair.vegalite.v4.schema.channels import Opacity, StrokeDash
import numpy as np
import time
import pandas as pd
from model.model_module import naive
from util import logging
import streamlit as st
import altair as alt
from testing.util import evaluation_fct
from testing.app_util import deck, layer_deck, plot_multistep_error, plot_line_all
import pydeck as pdk
import gc


def testing(out_sqc, lst, streets, timestamp, X_vl, X_ts, Y_ts):

    logging.info('Testing started')
    forecasts = []
    targets = []
    target_plot = []
    rmse_list = []
    mae_list = []



    st.markdown("""---""")
    st.subheader('Next 30 minutes')

    map = st.empty() 



    st.markdown("""---""")

    st.subheader('Total Belgian Traffic Flow')
    
    timestamp_t_h = st.empty()

    
    chart_all = st.empty()
    chart_multi = st.empty()

    st.markdown("""---""")
    st.subheader('RMSE and MAE Performance Metrics ')

    st.subheader(' Error For Each Time Horizon Separately  ')
    st.write(' historic values (red) vs current values (black)')

    
    col3, col4 = st.beta_columns(2)

    with col3:
            chart_errorrmse_multi = st.empty()           
                      
    with col4:
            chart_errormae_multi = st.empty()
            

    st.subheader(' Total Error Over Time  ')

    col5, col6 = st.beta_columns(2)

    with col5:
        df_rmse = pd.DataFrame({'timestamp':[],'RMSE': []})
        c = alt.Chart(df_rmse).mark_line().encode(x ='timestamp:T',y='RMSE',  tooltip=['RMSE']).properties(width=500, height=200)
        chart_rmse = st.altair_chart(c, use_container_width=True)

    with col6:
        df_mae = pd.DataFrame({'timestamp':[],'MAE': []})
        c = alt.Chart(df_mae).mark_line().encode(x ='timestamp:T', y='MAE',  tooltip=['MAE']).properties(width=500, height=200)
        chart_mae = st.altair_chart(c, use_container_width=True)
            
    X_old = X_vl.copy()
    X_new = X_ts.copy()

    
    
    for step in range(len(Y_ts)):


            # naive model
            new_instance = X_new[step]
            X_old = np.insert(X_old, X_old.shape[0], new_instance, axis=0)[1:]

            past = X_new[step] 
            pred = naive(X_old, 168*2*2*2, out_sqc).astype(np.int32)
            truth = Y_ts[step]

            layer = layer_deck( lst, streets, pred)
            r = deck(layer)
            map.pydeck_chart(r)
                      
            forecasts.append(pred)
            targets.append(truth)

            rmse, mae = evaluation_fct(targets, forecasts, out_sqc)

            # logging.info(' -- step '+ str(step)+' mae: ' +str(np.mean(mae))+' rmse: '+str(np.mean(rmse)))
            
            rmse_list.append(rmse)
            mae_list.append(mae)

            if step ==0 :
                target_plot.append(past)
                mean_rmse = np.mean(rmse)
                maen_mae = np.mean(mae)
                rmse_recent, mae_recent= evaluation_fct(targets, forecasts, out_sqc)

            else:
                target_plot.append(past[-1])
                mean_rmse = np.mean(rmse_list[-1])
                maen_mae = np.mean(mae_list[-1])
                rmse_recent, mae_recent = evaluation_fct(list([targets[-1]]), list([forecasts[-1]]), out_sqc)

            
            mean_pred_multi = np.sum(pred, axis=1)
            mean_truth_multi = np.sum(truth, axis=1)
            all_truth = np.sum(np.vstack(target_plot), axis=1)          

            mean_rmse_multi = np.mean(rmse, axis=1)
            mean_stdrmse_multi = np.std(rmse, axis=1)

            recent_mean_rmse_multi = np.mean(rmse_recent, axis=1)
            recent_mean_stdrmse_multi = np.std(rmse_recent, axis=1)

            mean_mae_multi = np.mean(mae, axis=1)
            mean_stdmae_multi = np.std(mae, axis=1)

            recent_mean_mae_multi = np.mean(mae_recent, axis=1)
            recent_mean_stdmae_multi = np.std(mae_recent, axis=1)

            time_window = timestamp.iloc[step+12:step+24]

            
            with col3:
                
                rmse_ci, rmse_dot = plot_multistep_error( time_window, mean_rmse_multi, mean_stdrmse_multi, 'red', 0.1 ,  350, 200)
                recent_rmse_ci, recent_rmse_dot = plot_multistep_error( time_window, recent_mean_rmse_multi, recent_mean_stdrmse_multi, 'black', 0.8 , 350, 200, 'ErrorBar')
                chart_errorrmse_multi.altair_chart( rmse_ci + rmse_dot + recent_rmse_ci + recent_rmse_dot, use_container_width=True)

                df_rmse = pd.DataFrame({'timestamp':[timestamp.iloc[step+12]],'RMSE': [mean_rmse]})
                chart_rmse.add_rows(df_rmse)

            with col4:

                mae_ci, mae_dot = plot_multistep_error( time_window, mean_mae_multi, mean_stdmae_multi, 'red', 0.1 , 350, 200)
                recent_mae_ci, recent_mae_dot = plot_multistep_error( time_window, recent_mean_mae_multi, recent_mean_stdmae_multi, 'black', 0.8 ,  350, 200, 'ErrorBar')
                chart_errormae_multi.altair_chart( mae_ci + mae_dot + recent_mae_ci + recent_mae_dot, use_container_width=True)

                df_mae = pd.DataFrame({'timestamp':[timestamp.iloc[step+12]], 'MAE': [maen_mae]})
                chart_mae.add_rows(df_mae)


            time_past = timestamp.iloc[:step+12]
            time_window = timestamp.iloc[step+12:step+24]

            timestamp_t_h.subheader('From '+str(timestamp.iloc[step+12]) +' To ' +str(timestamp.iloc[step+23]))
            line_past, line_targ, line_pred, line_zoom = plot_line_all( time_past, time_window, all_truth, mean_pred_multi, mean_truth_multi,  800, 500)

            chart_all.altair_chart(line_past + line_targ + line_pred)
            chart_multi.altair_chart(line_zoom)
            
            time.sleep(2)

            
            
            del recent_rmse_ci, recent_rmse_dot
            del recent_mae_ci, recent_mae_dot
            del line_past, line_targ, line_pred, line_zoom
            gc.collect()
