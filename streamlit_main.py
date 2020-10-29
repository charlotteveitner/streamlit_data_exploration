# importing general models
import os
import sys

# importing desired packages for python
import pandas as pd
import numpy as np
import seaborn as sn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# global settings
CWD = os.path.abspath('.')
sys.path.append(CWD + '/writings')

import streamlit_helper as st_help


# set up path to access raw data and folder to store any results
@st.cache(allow_output_mutation=True)
def load_data(CWD):
    """ loads the data from a parquet file specified below

    input: CWD = current working directory path

    output: df_raw = raw data from parquet file as pandas dataframe

    """

    folderpath_processed_data = CWD + '/writings/data_sample.parquet'
    df_raw = pd.read_parquet(folderpath_processed_data)

    return df_raw


def build_app():
    # Streamlit Layout
    st.beta_set_page_config(layout="wide")
    #st.markdown('<font style="font-family: Helvetica; font-size:23pt"> :tv: **Explorative Spot Analysis** </font>', unsafe_allow_html=True)
    st.markdown('<font style="font-family: Helvetica; font-size:11pt"> Test *Conversion is calculated as ratio between audience and reactions* </font>', unsafe_allow_html=True)

    data_load_state = st.markdown('<font style="font-family: Helvetica; font-size:10pt" > :hourglass: Loading data... </font>', unsafe_allow_html=True)
    # Load the data from parquet file on hard drive
    df_raw = load_data(CWD)
    data_load_state.markdown('<font style="font-family: Helvetica; font-size:10pt" > :heavy_check_mark: Loading data...done! </font>', unsafe_allow_html=True)

    # Checkbox whether to show the raw dataframe
    st.markdown('---')

    # Streamlit Sidebar Layout
    st.sidebar.markdown('<font style="font-family: Helvetica; font-size:18pt" > Analysis specifications :pencil2: </font>', unsafe_allow_html=True)
    st.sidebar.markdown('<font style="font-family: Helvetica; font-size:12pt" > Change settings for explorative analysis </font>', unsafe_allow_html=True)
    #raw_data_check = st.sidebar.checkbox('Show raw Data')

    # Radio button whether to show & compare to different dates or one analysis
    comparison = st.sidebar.radio('Compare two dates?', ['No', 'Yes'])

    # When chosing one view only create graphs and layout
    if comparison =='No':
        left, right = st.beta_columns((1,2))
        # Pick a date to show the analysis for
        # filter dataframe based on the desired constrains and use the desired frame for the rest of the analysis
        date_picker = st.sidebar.selectbox('Select Date Range', ['All', 'Pick Quarter', 'Pick Date Range', 'Last [X] months'])
        if date_picker == 'All':
            df= df_raw
        elif date_picker == 'Pick Quarter':
            quater_select = st.sidebar.multiselect('Quarters to be included', [1,2,3,4], [1,2,3,4])
            df_raw['quarter'] = df_raw.ad_timestamp.apply(lambda x: x.quarter)
            df = df_raw[df_raw.quarter.isin(quater_select)].drop(columns='quarter')
        elif date_picker == 'Pick Date Range':
            start_date = st.sidebar.date_input('Start Date', pd.Timestamp(df_raw.ad_timestamp.min()))
            end_date = st.sidebar.date_input('End Date', pd.Timestamp(df_raw.ad_timestamp.max()))
            df = df_raw[(df_raw.ad_timestamp > (start_date - pd.DateOffset(days=0))) & (df_raw.ad_timestamp < (end_date + pd.DateOffset(days=1)))]
        elif date_picker == 'Last [X] months':
            last_months = st.sidebar.slider('Chose the number of months to go back', min_value=1, max_value=20, value=6, step=1)
            cut_date = pd.Timestamp.today() - pd.DateOffset(day=1, months=last_months)
            st.sidebar.write('Including all data starting 1st of {} {}'.format(cut_date.month_name(), cut_date.year))
            df = df_raw[(df_raw.ad_timestamp >= cut_date) ]


        # Streamlit Sidebar selection of what to see
        show_what = st.sidebar.selectbox('Select a graph to see', ['Pivot table', 'Histogram', 'Correlation Matrix', 'Scatter Plot'])

        # If Pivot table is selected for analysis
        if show_what == 'Pivot table':
            # define a list of columns that can be selected as index columns
            # to use for aggregation of the pivot table
            pivot_index_options = [
                                    'is_prime_time',
                                    'weekday',
                                    'hour',
                                    'channel_name',
                                    'relative_spot_position',
                                    'tv_spotlaenge',
                                    'brand_name'
                                  ]
            pivot_index_second_option = ['None'] + pivot_index_options.copy()
            first_index = left.selectbox('Select a column to use for pivot table', pivot_index_options)
            pivot_index_second_option_updates = pivot_index_second_option#.remove(first_index)
            second_index = left.selectbox('Select a secondary column to use for pivot table', pivot_index_second_option_updates)
            show_total_numbers = left.checkbox('Show numbers for audience and reactions')
            if second_index != 'None':
                index_columns = [first_index, second_index]
            elif second_index == 'None':
                index_columns = [first_index]
            pivot_show = st_help.make_pivot_table(
                                                df=df,
                                                show_total_numbers=show_total_numbers,
                                                index_columns=index_columns,
                                                value_columns=['audience','total_react_linear']
                                            )
            right.write(pivot_show)

        # If Histogram is selected for analysis
        elif show_what == 'Histogram':
            # List of features that can be used for the histogram
            features_to_include =  [
                                            'hour',
                                            'weekday',
                                            'is_prime_time',
                                            'channel_name',
                                            'tv_position_im_block',
                                            'tv_spotanzahl_im_block',
                                            'relative_spot_position',
                                            'tv_spotlaenge',
                                            #'audience',
                                            'brand_name',
                                            'motiv_name'
                                            #'total_react_linear'
                                        ]
            select_options = features_to_include
            # select an option for the specific metric defined above to show a histrogram for
            # that filtered option
            additional_filter = left.selectbox('Filter histogram by additional metric', options=select_options)
            title = 'Histogram for all observations'
            filter_metrics = list(df_raw[additional_filter].drop_duplicates().sort_values())
            filter_keep =  left.selectbox('Select desired filtering option', options=filter_metrics)
            grouped = left.checkbox('Display grouped raw data')
            fig_hist, fig_filtered_hist, df_grouped_sorted = st_help.create_histogram(df, additional_filter, filter_keep)
            right.plotly_chart(fig_hist)
            right.plotly_chart(fig_filtered_hist)
        elif show_what == 'Correlation Matrix':

            features_to_filter = ['None', 'brand_name', 'channel_name', 'weekday']
            additional_filter_corr = left.selectbox('Filter histogram by additional metric', options=features_to_filter)
            if additional_filter_corr == 'None':
                filter_keep_corr = 'None'
            else:
                filter_metrics_corr = list(df_raw[additional_filter_corr].drop_duplicates().sort_values())
                filter_keep_corr =  left.selectbox('Select desired filtering option', options=filter_metrics_corr)
            df_corr = df.copy()
            feature_column_list =  [
                                            'hour',
                                            'is_prime_time',
                                            'channel_name',
                                            'tv_position_im_block',
                                            'tv_spotanzahl_im_block',
                                            'relative_spot_position',
                                            'tv_spotlaenge',
                                            'audience',
                                            'day_time',
                                            'spot_position_at_edge',
                                            'react_per_100_audience'
                                        ]
            correlation_heatmap_general, df_full_correlation = st_help.create_correlation_heatmap(df_corr, feature_column_list, additional_filter_corr, filter_keep_corr)

            if additional_filter_corr != 'None':
                right.write('Correlation Heatmap with Pearson’s correlation coefficient where {} is {}'.format(additional_filter_corr, filter_keep_corr))
            else:
                right.write('General Correlation Heatmap with Pearson’s correlation coefficient')
            right.write(correlation_heatmap_general)
            if left.checkbox('Show Full Correlation Details Dataframe'):
                right.write(df_full_correlation)
        elif show_what == 'Scatter Plot':
            # List of features that can be used for the histogram
            features_to_include =  [
                                            'hour',
                                            'weekday',
                                            'is_prime_time',
                                            'channel_name',
                                            'tv_position_im_block',
                                            'tv_spotanzahl_im_block',
                                            'relative_spot_position',
                                            'tv_spotlaenge',
                                            #'audience',
                                            'brand_name',
                                            'motiv_name'
                                            #'total_react_linear'
                                        ]
            select_options = features_to_include
            # select an option for the specific metric defined above to show a histrogram for
            # that filtered option
            additional_filter = left.selectbox('Chose additional metric for Scatterplot', options=select_options)
            title = 'Scatter Plot for all observations'
            df_scatter = df.copy()
            fig_scatter = st_help.create_scatter_plot(df_scatter, additional_filter)
            right.plotly_chart(fig_scatter)

    if comparison =='Yes':
        left, middle, right = st.beta_columns((1,2,2))
        date_picker = st.sidebar.selectbox('Select Date Range', ['Pick Quarter', 'Pick Date Range'])

        if date_picker == 'Pick Quarter':
            df_raw['quarter'] = df_raw.ad_timestamp.apply(lambda x: x.quarter)
            quater_select1 = st.sidebar.multiselect('Quarters to be included [Timespan 1]', [1,2,3,4], [1,2])
            df1 = df_raw[df_raw.quarter.isin(quater_select1)].drop(columns='quarter')
            quater_select2 = st.sidebar.multiselect('Quarters to be included [Timespan 2]', [1,2,3,4], [3,4])
            df2 = df_raw[df_raw.quarter.isin(quater_select2)].drop(columns='quarter')

        elif date_picker == 'Pick Date Range':
            start_date1 = st.sidebar.date_input('Start Date [Timespan 1]', pd.Timestamp(df_raw.ad_timestamp.min()))
            end_date1 = st.sidebar.date_input('End Date [Timespan 1]', pd.Timestamp('2019-12-31'))
            df1 = df_raw[(df_raw.ad_timestamp > (start_date1 - pd.DateOffset(days=0))) & (df_raw.ad_timestamp < (end_date1 + pd.DateOffset(days=1)))]
            start_date2 = st.sidebar.date_input('Start Date [Timespan 2]', pd.Timestamp('2020-01-01'))
            end_date2 = st.sidebar.date_input('End Date [Timespan 2]', pd.Timestamp(df_raw.ad_timestamp.max()))
            df2 = df_raw[(df_raw.ad_timestamp > (start_date2 - pd.DateOffset(days=0))) & (df_raw.ad_timestamp < (end_date2 + pd.DateOffset(days=1)))]
        show_what = st.sidebar.selectbox('Select a graph to see', ['Pivot table', 'Histogram', 'Correlation Matrix', 'Scatter Plot'])

        if show_what == 'Pivot table':

            pivot_index_options = [
                                    'weekday',
                                    'is_prime_time',
                                    'hour',
                                    'channel_name',
                                    'relative_spot_position',
                                    'tv_spotlaenge',
                                    'brand_name'
                                  ]
            pivot_index_second_option = ['None'] + pivot_index_options.copy()
            first_index = left.selectbox('Select a column to use for pivot table', pivot_index_options)
            pivot_index_second_option_updates = pivot_index_second_option#.remove(first_index)
            second_index = left.selectbox('Select a secondary column to use for pivot table', pivot_index_second_option_updates)
            show_total_numbers = left.checkbox('Show numbers for audience and reactions')
            if second_index != 'None':
                index_columns = [first_index, second_index]
            elif second_index == 'None':
                index_columns = [first_index]
            middle.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 1** </font>', unsafe_allow_html=True)
            pivot_show1 = st_help.make_pivot_table(
                                                df=df1,
                                                show_total_numbers=show_total_numbers,
                                                index_columns=index_columns,
                                                value_columns=['audience','total_react_linear']
                                            )
            middle.write(pivot_show1)

            right.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 2** </font>', unsafe_allow_html=True)
            pivot_show2 = st_help.make_pivot_table(
                                        df=df2,
                                        show_total_numbers=show_total_numbers,
                                        index_columns=index_columns,
                                        value_columns=['audience','total_react_linear']
                                    )
            right.write(pivot_show2)

        elif show_what == 'Histogram':

            features_to_include =  [
                                            'hour',
                                            'weekday',
                                            'is_prime_time',
                                            'channel_name',
                                            'tv_position_im_block',
                                            'tv_spotanzahl_im_block',
                                            'relative_spot_position',
                                            'tv_spotlaenge',
                                            #'audience',
                                            'brand_name',
                                            'motiv_name'
                                            #'total_react_linear'
                                        ]
            select_options = features_to_include
            additional_filter = left.selectbox('Filter histogram by additional metric', options=select_options)
            title = 'Histogram for all observations'
            filter_metrics = list(df_raw[additional_filter].drop_duplicates().sort_values())
            filter_keep =  left.selectbox('Select desired filtering option', options=filter_metrics)
            grouped = st.checkbox('Display grouped raw data')
            fig_hist1, fig_filtered_hist1, df_grouped_sorted1 = st_help.create_histogram(df1, additional_filter, filter_keep)
            fig_hist2, fig_filtered_hist2, df_grouped_sorted2 = st_help.create_histogram(df2, additional_filter, filter_keep)

            middle.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 1** </font>', unsafe_allow_html=True)
            middle.plotly_chart(fig_hist1)
            middle.plotly_chart(fig_filtered_hist1)
            right.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 2** </font>', unsafe_allow_html=True)
            right.plotly_chart(fig_hist2)
            right.plotly_chart(fig_filtered_hist2)
            if grouped:
                middle.write(df_grouped_sorted1)
                right.write(df_grouped_sorted2)

        elif show_what == 'Correlation Matrix':
            features_to_filter = ['None', 'brand_name', 'channel_name', 'weekday']
            additional_filter_corr = left.selectbox('Filter correlation matrix by additional metric', options=features_to_filter)
            if additional_filter_corr == 'None':
                filter_keep_corr = 'None'
            else:
                filter_metrics_corr = list(df_raw[additional_filter_corr].drop_duplicates().sort_values())
                filter_keep_corr =  left.selectbox('Select desired filtering option', options=filter_metrics_corr)

            feature_column_list =  [
                                            'hour',
                                            'is_prime_time',
                                            'channel_name',
                                            'tv_position_im_block',
                                            'tv_spotanzahl_im_block',
                                            'relative_spot_position',
                                            'tv_spotlaenge',
                                            'audience',
                                            'day_time',
                                            'spot_position_at_edge',
                                            'react_per_100_audience'
                                        ]
            correlation_heatmap_general1, df_full_correlation1 = st_help.create_correlation_heatmap(df1, feature_column_list, additional_filter_corr, filter_keep_corr)
            correlation_heatmap_general2, df_full_correlation2 = st_help.create_correlation_heatmap(df2, feature_column_list, additional_filter_corr, filter_keep_corr)

            if additional_filter_corr != 'None':
                left.write('Correlation Heatmap with Pearson’s correlation coefficient where {} is {}'.format(additional_filter_corr, filter_keep_corr))
            else:
                left.write('General Correlation Heatmap with Pearson’s correlation coefficient')

            full_corr_df_show = left.checkbox('Show Full Correlation Details Dataframe')
            middle.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 1** </font>', unsafe_allow_html=True)
            middle.write(correlation_heatmap_general1)
            if full_corr_df_show:
                middle.write(df_full_correlation1)
            right.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 2** </font>', unsafe_allow_html=True)
            right.write(correlation_heatmap_general2)
            if full_corr_df_show:
                right.write(df_full_correlation2)

        elif show_what == 'Scatter Plot':
            # List of features that can be used for the histogram
            features_to_include =  [
                                            'hour',
                                            'weekday',
                                            'is_prime_time',
                                            'channel_name',
                                            'tv_position_im_block',
                                            'tv_spotanzahl_im_block',
                                            'relative_spot_position',
                                            'tv_spotlaenge',
                                            #'audience',
                                            'brand_name',
                                            'motiv_name'
                                            #'total_react_linear'
                                        ]
            select_options = features_to_include
            # select an option for the specific metric defined above to show a histrogram for
            # that filtered option
            additional_filter = left.selectbox('Chose additional metric for Scatterplot', options=select_options)

            fig_scatter1 = st_help.create_scatter_plot(df1, additional_filter)
            fig_scatter2 = st_help.create_scatter_plot(df2, additional_filter)

            middle.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 1** </font>', unsafe_allow_html=True)
            middle.write(fig_scatter1)
            right.markdown('<font style="font-family: Helvetica; font-size:23pt"> **Timespan 2** </font>', unsafe_allow_html=True)
            right.write(fig_scatter2)


build_app()
