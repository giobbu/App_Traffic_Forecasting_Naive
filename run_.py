
import pickle

from util import logging, load_streets, set_seed
from app_util import  plot_loss, plot_deck, plot_network
from app_util import plot_line_alt, plot_hist_alt, create_df_stats, plot_mat_alt, plot_trends, plot_violin_alt, plot_ridge_alt

from data.data_module import data_reader, feat_engin,  data_loader_naive
from data.data_module import  data_loader_naive

from data.app_util import plot_distribution, plot_gantt, plot_tensor
from testing.test_module import testing


import time
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import numpy as np
import folium
import geopandas as gpd
import pandas as pd


# App packages
import streamlit as st
import altair as alt
import SessionState
import pydeck as pdk
import matplotlib.pyplot as plt


session_state = SessionState.get(check1=False)

file_streets = 'data/Belgium_streets.json'
checkpoint_dir = 'trained_model_dir'


# with open('config.yaml') as file:
#         config = yaml.safe_load(file)

seed = 42
path = 'data/Flow_BEL_street_30min.csv' #config['script_path']
mean_value = 10 #config['data']['threshold']
n_feat_time = 4 #config['data']['time_feature']
validation_period = 336 #config['data']['validation']
testing_period = 336 #config['data']['testing']

inp_sqc = 12 #config['loader']['input_sqc']
out_sqc = 12 #config['loader']['output_sqc']

def main():

        st.set_page_config(page_title="Traffic Forecasting APP")

        set_seed(seed)

        st.markdown("""---""")
        # Space out the maps so the first one is 2x the size of the other three
        with st.beta_container():

                col1, col2 = st.beta_columns([20,20])
                with col1:
                        st.image('mlg.png', width=250)
                with col2:
                        st.image('ULB.jpeg', width=200)
                        
                st.markdown("""---""")
                st.title('Freight Traffic Forecasting')
                st.write('Reliable Traffic Forecasting schemes are fundamental to design Proactive Intelligent Transportation Systems (ITS).')
                st.markdown("""---""")
                
                st.header('Multi-step and Multivariate Forecasting:')
                st.write('In the literature the task of predicting the traffic condition for multiple time horizons and for multiple locations corresponds to problem of multi-step and multivariate forecasting. More references [here](https://link.springer.com/chapter/10.1007/978-3-642-36318-4_3).')
                st.header('Naive Model:')
                st.write('Within a sliding window, observations at the same time and same day in previous week seasons are collected and the mean of those observations is returned as persisted forecast.')
                st.header('OBU Data:')
                st.write('Data related to lorries travelling on Belgian public roads. In the study case here presented the data have been processed via the big data architecture at [Machine Learning Group, ULB, Bruxelles](https://mlg.ulb.ac.be/wordpress/).')
                st.markdown("""---""")

                st.subheader('Click and See')


                if st.button('Data Set Summary'):
                        st.title('Overview')
                        df = data_reader(path)
                        streets = load_streets(file_streets)
                        
                        st.header('Highways')
                        st.text('Folium Visualization')
                        plot_network(file_streets)
                        
                        st.header('Traffic Data')
                        st.subheader('Raw OBU Data')              
                        st.dataframe(df.head())

                        col0, col1, col2 = st.beta_columns(3)

                        with col0:
                                st.subheader('Temporal')
                                st.text('-- Period-- ')
                                st.text('from ' + df.iloc[0,0])
                                st.text('to '+ df.iloc[-1,0])
                                st.text('Granularity: 30 minutes')
                                st.text('Number Observations ' + str(df.shape[0]))

                        with col1:
                                st.subheader('Spatial')
                                st.text('Number of Streets: ' +str(df.shape[1]))
                                st.text('-- Bounds Box --')
                                box = streets.bounds.values[0]
                                st.text('Upper Left: ' + str(round(box[3],3)) +', '+ str(round(box[0],3)))
                                st.text('Lower Right: ' + str(round(box[1],3)) +', '+ str(round(box[2],3)))

                        with col2:
                                st.subheader('Traffic')
                                mean = round(np.mean(df.iloc[:,1:].sum(axis=1).values),0)
                                st.text('Total Mean: ' + str(mean))
                                max = np.max(df.iloc[:,1:].sum(axis=1))
                                st.text('Total Max: ' + str(max))
                                min = np.min(df.iloc[:,1:].sum(axis=1))
                                st.text('Total Min: ' + str(min))

                        st.markdown("""---""")

                        if st.button("BACK"):
                                st.text("Restarting...")

                elif st.button('Data Set Exploration'):

                        st.markdown("""---""")
                        st.title('Info & Viz')

                        df = data_reader(path)
                        # compute max OBU traffic
                        df_stats = create_df_stats(df)
                        streets = load_streets(file_streets)
                        
                        st.subheader('Check the whole period')
                        # plot timeseries total obu traffic
                        plot_line_alt(df)

                        # plot histogram mean obu traffic
                        plot_hist_alt(df)

                        # plot matrix OBU traffic
                        df_max = df_stats.groupby('date').agg({'flow_sum':'max'}).reset_index()
                        plot_mat_alt(df_max)
                        
                        st.subheader('Check working days/weekend')
                        # plot avg trend working, sat, sund
                        plot_trends(df)

                        # plot max OBU data working, sat, sund
                        df_ww = df_stats.copy()
                        df_ww.loc[df_ww.week < 5, "week"] = "max_working_days"
                        df_ww.loc[df_ww.week == 5, "week"] = "max_saturdays"
                        df_ww.loc[df_ww.week ==6, "week"] = "max_sundays"
                        df_max = df_ww.groupby('week').agg({'flow_sum':'max'}).reset_index()
                        plot_violin_alt(df_max)


                        st.subheader('Check days of the week')

                        # plot max OBU data distribution days of the week

                        df_stats.loc[df_stats.week == 0, "week"] = "1_mon"
                        df_stats.loc[df_stats.week == 1, "week"] = "2_tue"
                        df_stats.loc[df_stats.week == 2, "week"] = "3_wed"
                        df_stats.loc[df_stats.week == 3, "week"] = "4_thu"
                        df_stats.loc[df_stats.week == 4, "week"] = "5_fri"
                        df_stats.loc[df_stats.week == 5, "week"] = "6_sat"
                        df_stats.loc[df_stats.week ==6, "week"] = "7_sun"
                        df_max = df_stats.groupby('week').agg({'flow_sum':'max'}).reset_index()
                        plot_ridge_alt(df_max, title = 'Max OBU traffic per day of the week ')
                        
                        st.subheader('Check hours of working day')
                        df_max = df_stats.groupby('hour').agg({'flow_sum':'max'}).reset_index()
                        plot_ridge_alt(df_max, title = 'Max OBU traffic per hour of the working days ')



                        if st.button("BACK"):
                                st.text("Restarting...")




                elif st.button('Naive Inference'):

                        st.header('Streets with mean traffic flow > 10 km/30min')

                        df = data_reader(path)
                        
                        streets = load_streets(file_streets)                

                        df_new, lst_streets, timestamp = feat_engin(mean_value, df)

                        X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts, timestamp_test = data_loader_naive(df_new.values, inp_sqc, out_sqc, validation_period, testing_period, timestamp)

                        pred, targ, rmse, mae = testing(out_sqc, lst_streets, streets, timestamp_test, X_tr, Y_tr, X_vl, Y_vl, X_ts, Y_ts)#[inp_sqc:]) 

        # logging.info("Finally, I can eat my pizza(s)")


if __name__ == "__main__":
        main()
