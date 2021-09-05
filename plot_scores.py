# plots a bar graph of the output outlier scores with labelled outliers in red
# uses a subset of the points when there are too many to plot (to be tweaked)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_labelled(filename):
    # load outliers scores
    path="outlierscores/20D_"+filename+".csv"
    scores_df=pd.read_csv(path)

    # put in increasing order
    scores_df=scores_df.sort_values("score")
    scores_df=scores_df.reset_index(drop=True)

    legend_name="Labelled Outliers"

    if filename=="abide":
        # add labelled outliers from asymHIM
        path_asym="outlierscores/asym_scores.csv"
        asym_df=pd.read_csv(path_asym)
        scores_df["class"]=asym_df["outlier_ind"]    
        legend_name="AsymHIM Outliers"

        width_other_points=0.2
        width_labelled_outliers=1
        
        step_labelled=1 # how many labelled outliers will be plotted? (all of them)
        step_points=1 # how many other points will be plotted? (all of them)
        
    elif filename =="AID362":
        scores_df["class"]=scores_df["class"].map({1:1, -1:0}) 
    
        #set the widths of the bars in the plot
        width_other_points=1 # for AID
        width_labelled_outliers=5 # for AID

        step_labelled=1 # how many labelled outliers will be plotted? (all of them)
        step_points=10 # how many other points will be plotted? (1 out of 10)

    end_points=sum(scores_df["class"]==0)
    end_labelled=sum(scores_df["class"]==1)
    

    if filename=="census":
        #set the widths of the bars in the plot
        width_other_points=100 # for census
        width_labelled_outliers=500 # for census

        # TO SET:
        nb_labelled=50 # how many labelled outliers will be plotted?
        nb_points=300 # how many other points will be plotted?

        # alternatively, set the steps directly
        step_labelled=int(end_labelled/nb_labelled)+1
        step_points=int(end_points/nb_points)+1
       
    # how many labelled outliers do we have?
    print("number of labelled outliers:")
    print(sum(scores_df["class"]==1))
    print("number of labelled inliers :")
    print(sum(scores_df["class"]==0))
    print("number of points total:")
    print(scores_df.shape[0])
    
    #remove some points to be able to plot them
    scores_df=take_subset(scores_df, step_labelled, step_points, end_labelled, end_points)

    # set the colors: red if labelled outlier, blue otherwise
    colors=scores_df["class"].map({0:"blue", 1:"red"}) 

    # set the widths of the bars on the graph
    widths=scores_df["class"].map({0:width_other_points,1: width_labelled_outliers})
    
    # prepare to make the graph
    x=scores_df.index
    y=scores_df["score"]
    
    # make the plot
    make_plot(x,y,filename, legend_name, widths, colors)
    

    print("number of blue points plotted:")
    print(sum(colors=="blue"))
    print("number of reds points plotted:")
    print(sum(colors=="red"))


def take_subset(scores_df, step_labelled, step_points, end_labelled, end_points):
    # select the subset of points to plot
    to_plot=scores_df.loc[scores_df["class"]==0][0:end_points:step_points]
    to_plot=to_plot.append(scores_df.loc[scores_df["class"]==1][0:end_labelled:step_labelled])
    return to_plot
        
def make_plot (x,y,filename, legend_name, widths, colors):
    matplotlib.use("Agg") # prevent Compute Canada from displaying plots
    
    # set the size of the figure
    fig = plt.figure(figsize=(10, 6), dpi=100)

    # plot
    plt.bar(x,y, color=colors, width= widths, snap=False)

    # add labels
    fig.suptitle("Distribution of outlier scores for "+ filename +" data")
    plt.xlabel('Subject')
    plt.ylabel('Score')

    #set legend
    plt.legend([legend_name])
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('red')

    # save figure
    plt.savefig("results/barplot_scores_"+filename+".png")


