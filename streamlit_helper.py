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


def make_pivot_table(df, show_total_numbers=False, index_columns=['is_prime_time'], value_columns=['audience','total_react_linear']):
    """ creates a pivot table for the given daraframe
    using the audience and total react linear as the aggregation values
    from those values the conversion rate (defined as reactions/audience)
    is calculated as the final value to show in the pivot table

    input: df = dataframe to make the pivot table out of
           show_total_numbers = boolean that defines whether to show the
                                total numbers for the audience or reactions.
                                if == False then only conversion will be shown
           index_columns = single column or set of two columns to be used as
                           columns to group by (if nothing is chosen default is
                           set to prime time)
           value_columns = columns to be aggregated. default is audience and linear react

    output: pivot_color = pivot table with color scheme based on values

    """
    # create pivot table with internal pandas function
    pivot = pd.pivot_table(
                            data=df,
                            index=index_columns,
                            values=value_columns,
                            aggfunc= np.sum
                          )

    pivot = pivot.reset_index()
    # convert the value columns to integer
    for i in value_columns:
        pivot[i] = pivot[i].astype('int')
    # if both audience and reactions are present caluclate the conversion
    if 'audience' in value_columns and 'total_react_linear' in value_columns:
         pivot['conversion_rate[%]'] = (pivot['total_react_linear']/pivot['audience'])*100
         pivot['conversion_rate[%]'] = pivot['conversion_rate[%]'].apply(lambda x: round(x,3))
    # if weekday is the only column to aggregate by, then sort it by the weekdays
    if ('weekday' in index_columns) and len(index_columns)==1:
        cats = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        cats = list(set(cats).intersection(pivot['weekday']))
        pivot = pivot.groupby(index_columns).agg({'audience':np.sum, 'total_react_linear':np.sum, 'conversion_rate[%]':np.mean}).reindex(cats).reset_index(drop=False)

    # else sort by the values in the index columns
    else:
        pivot = pivot.sort_values(by=index_columns)

    # if the total number checkmark is checked, then show everything, else drop the columns
    # that contain the audience and the total react values
    if show_total_numbers:
        pivot_show = pivot.copy()
    elif ~ show_total_numbers:
        pivot_show = pivot.copy().drop(columns=['audience','total_react_linear'])

    # color the pivot table based on the values of the conversion
    cm = sn.light_palette("green", as_cmap=True)
    pivot_color = pivot_show.style.background_gradient(cmap=cm, subset=['conversion_rate[%]'])
    return pivot_color


def create_histogram(df_hist, additional_filter, filter_keep):
    """ creates a histogram for given dataframe using a specific metric

    input: df_hist = dataframe to make the pivot table out of
           additional_filter =  metric to use as X-axis for the histogram
           filter_keep = filter histogram by one value of the metric
                         and show that data
           grouped = boolean that defines whether to show the
                     total numbers for the audience or reactions.
                     if == False then only conversion will be shown

    output: pivot_color = pivot table with color scheme based on values

    """

    #df_hist['react_per_100_audience'] = 100*(df_hist['total_react_linear']/df_hist['audience'])
    df_hist['is_prime_time'] = np.where(df_hist['is_prime_time']==1, 'prime', 'no_prime')
    df_filtered_hist = df_hist.copy()
    # Explore Target variable total react total_react_linear

    df_grouped = df_filtered_hist.groupby(additional_filter, as_index=False).agg({'react_per_100_audience':[np.mean, np.std, np.max]})
    if additional_filter == 'weekday':
        cats = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        df_grouped = df_filtered_hist.groupby(additional_filter).agg({'react_per_100_audience':[np.mean, np.std, np.max]}).reindex(cats).reset_index(drop=False)
    else:
        df_grouped = df_grouped.sort_values(by=additional_filter,ascending=True)
    df_grouped.columns = ["".join(x) for x in df_grouped.columns.ravel()]
    #df_grouped.columns = df_grouped.columns.droplevel(0)
    df_grouped_sorted = df_grouped.rename(
                                            columns={
                                                        'react_per_100_audiencemean': 'average',
                                                        'react_per_100_audiencestd':'standard_deviation',
                                                        'react_per_100_audienceamax':'maximum'
                                                    }
                                            )
    fig_filter = go.Figure()
    fig_filter.add_trace(go.Histogram(
        x= df_grouped_sorted[additional_filter],
        y= df_grouped_sorted['average'],
        histfunc='avg',
        name='Avg Reactions / 100 aud',
        nbinsx=len(filter_metrics)+1
    ))
    fig_filter.add_trace(go.Scatter(
        name='Std dev',
        mode='markers',
        x=df_grouped_sorted[additional_filter], y=df_grouped_sorted['average'],
        error_y=dict(type='data', array=df_grouped_sorted['standard_deviation'], color='purple', thickness=1.5, width=3),
        marker=dict(color='black', size=5, opacity=0.75)
    ))

    fig_filter.update_layout(
                                title='Average reactions per 100 audience by {}'.format(additional_filter),
                                xaxis_title=additional_filter,
                                yaxis=dict(rangemode='nonnegative'),
                                yaxis_title="Avg Reactions / 100 aud",
                                #legend_title="Legend Title",
                                autosize=False,
                                width=600,
                                height=400,
                                font=dict(
                                    #family="Courier New, monospace",
                                    size=12,
                                    color="black"
                                )
                             )

    title = 'Histogram for observations where {} is {}'.format(additional_filter, filter_keep)
    df_filtered_hist = df_filtered_hist[df_filtered_hist[additional_filter]==filter_keep]
    cutoff = 0.9
    fig = px.histogram(
                        np.clip(df_filtered_hist['react_per_100_audience'], -0.1, cutoff),
                        x = 'react_per_100_audience',
                        nbins=400,
                        histnorm='percent'
                       )
    fig.update_layout(
                        title=title,
                        xaxis_title="Reactions per 100 audience",
                        yaxis_title="Distribution in %",
                        autosize=False,
                        width=600,
                        height=400,
                        font=dict(
                                    #family="Courier New, monospace",
                                    size=12,
                                    color="black"
                                 )
                     )
    return fig_filter, fig, df_grouped_sorted


def create_correlation_to_target_value(df_corr,feature_column_list, target_column='react_per_100_audience'):
    """
    Creates a correlation matrix of all numerical features
    input:
            df_corr: the original dataframe to make build the correlations
            feature_column_list: features to be included in the correlation analysis
            target_column: the colum that stores the target value
    output:
            a table of different correlation metrics for the speicifc features & target value
    """

    print(feature_column_list)
    if target_column in feature_column_list:
        feature_column_list.remove(target_column)
    list_of_dfs = []
    for i in feature_column_list:
        try:
            df_pg = pg.corr(x=df_corr[i], y=df_corr[target_column])
            df_pg.index = [i]
            list_of_dfs.append(df_pg)
        except:
            print('correlation did not work for {}'.format(i))
    df_complete_corr = pd.concat(list_of_dfs, ignore_index=False)
    df_complete_corr = df_complete_corr[[
                                            'n',
                                            'r',
                                            'CI95%',
                                            'p-val',
                                            'BF10'
                                        ]].rename(
                                                    columns={
                                                                'n':'# sample',
                                                                'r': 'correlation coefficient',
                                                                'CI95%': '95% confidence interval',
                                                                'p-val': 'p value',
                                                                'BF10': 'Bayes Factor'
                                                            }
                                                  )
    return df_complete_corr


def create_correlation_heatmap(df_corr, feature_column_list, additional_filter_corr, filter_keep_corr):
    """
    Creates a correlation matrix of all numerical features
    input:
            df_corr: the original dataframe to make build the correlations
            feature_column_list: features to be included in the correlation analysis
            additional_filter_corr: additional feature to filter the correlation heatmap
            filter_keep_corr: filter metric for the addiitonal filter
    output:
            a figure of a heatmap with the correlations
    """
    if additional_filter_corr != 'None':
        df_corr = df_corr[df_corr[additional_filter_corr]==filter_keep_corr]
    if len(df_corr)>15000:
        # st.write('Only using last 15.000 entries to create correlation Heatmap')
        df_corr = df_corr.sort_values(by='ad_timestamp', ascending=False).head(15000)
    # find categorical columns



    df_sample_all_corr = df_corr[feature_column_list]
    # calculate all correlation details df
    df_full_correlation = create_correlation_to_target_value(df_corr,feature_column_list, target_column='react_per_100_audience')
    # create correlation heatmap
    corrs = df_sample_all_corr.corr()
    mask = np.zeros_like(corrs)
    mask[np.triu_indices_from(mask)] = True

    #plt.title('Correlation matrix')
    #
    # correlation_matrix_all = df_sample_all_corr.corr()
    # mask = np.zeros_like(correlation_matrix_all)
    # mask[np.triu_indices_from(mask)] = True
    with sn.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 8))
        ax = sn.heatmap(corrs, cmap='Spectral_r', mask=mask, square=True, vmin=-.4, vmax=.4, annot=True, fmt='.1g')

    return ax.get_figure(), df_full_correlation


def create_scatter_plot(df_scatter, feature):
    """
    Creates a correlation matrix of all numerical features
    input:
            df_scatter: the original dataframe to make make the scatter plot for
            feature: feature used for X-Axis of the scatter plot
    output:
            a figure of the scatter plot
    """
    df_scatter['conversion_rate[%]'] = (df_scatter['total_react_linear']/df_scatter['audience'])*100
    df_scatter['conversion_rate[%]'] = df_scatter['conversion_rate[%]'].apply(lambda x: round(x,3))
    df_scatter['is_prime_time'] = np.where(df_scatter['is_prime_time']==1, 'prime', 'no_prime')

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        name='Std dev',
        mode='markers',
        x=df_scatter[feature], y=df_scatter['conversion_rate[%]'],
        marker=dict(color='black', size=5, opacity=0.75)
    ))

    fig_scatter.update_layout(
                                title='Conversion Rate Scatter Plot for feature {}'.format(feature),
                                xaxis_title=feature,
                                yaxis=dict(rangemode='nonnegative'),
                                yaxis_title=" Conversion Rate [Reactions/100 aud]",
                                #legend_title="Legend Title",
                                autosize=False,
                                width=700,
                                height=450,
                                font=dict(
                                    #family="Courier New, monospace",
                                    size=12,
                                    color="black"
                                )
                             )

    return fig_scatter
