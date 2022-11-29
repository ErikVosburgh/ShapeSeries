import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import minmax_scale
from scipy.stats import iqr
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import calinski_harabasz_score
from EXO_2022 import df_exo_2022
from EXO_2021 import df_1, df_2, df_3


max_temp = max(df_exo_2022 ['Temperature'])

min_temp = min(df_exo_2022 ['Temperature'])

# color_list = ['purple', 'red', 'green', 'yellow', 'pink', 'blue', 'black', 'violet', 'teal', 'brown', 'olive', 'cyan','orange']

# =============================================================================
# shape = conductivity_medianshape_library['1']
#
# data=conductivity_data_scaled.interpolate(method='pad')
# param='conductivity'
# units='ms/cm'
# color_1 = color_list[pattern]
# threshold = conductivity_shape_1_thresh
# =============================================================================

### LOAD TEMPLATE SHAPE DATA: ##########################################################################


def shape_scan(shape, data, param, unit, color_1, threshold):
    # data = param_norm(param,data)
    data = data.reset_index()
    data_target = data
    data = data.interpolate(method="pad")
    param = param
    unit = unit

    # NORMALIZE
    data = data_target
    pattern_length = 24  # hrs
    m = pattern_length * 12

    shape_data = shape

    ###store shapes in data frame

    shape_df = pd.DataFrame(columns=["shape"])
    shape_df["shape"] = shape_data

    ### LOAD TARGET DATA: ##########################################################################
    data = data
    # data = data.reset_index()
    data = data.interpolate(method="pad")
    # data['scaled'] = [(num - data_min) / (data_max - data_min) for num in data[param]]

    ###store ts and distance profile in data frame
    target_df = pd.DataFrame(columns=[param, "ed_profile", "dist", "match"])
    target_df[param] = data["scaled"]

    ed_profile = np.ones(len(data)) * np.inf
    for i in range(
        0, len(data) - m
    ):  # assign and standardize sub_stream from template sequence
        sub_stream = target_df.loc[i : i + m - 1, param]
        sub_stream = sub_stream.to_numpy()
        ed_profile[i] = np.linalg.norm(
            sub_stream - shape_df["shape"].to_numpy()
        )

    target_df["ed_profile"] = ed_profile
    target_df["dist"] = ed_profile
    target_df["shape"] = np.nan

    ### IDENTIFY MATCHES ##########################################################################
    bottom = threshold
    target_df["ed_profile"] = target_df["ed_profile"].mask(
        target_df["ed_profile"] > bottom
    )
    condition = target_df["ed_profile"] <= bottom
    ### New TS for TS 1 matches
    for i in range(0, len(data) - m):
        if target_df.loc[i, "ed_profile"] <= bottom:
            target_df.loc[i : i + m, "shape"] = target_df.loc[i : i + m, param]


    # Store and Plot shapes

    x = np.arange(0, m + 1, 1)
    i = 0
    j = 0

    matches = sum(condition)
    matches = pd.DataFrame(index=np.arange(0, m, 1), columns=np.arange(matches))
    match_idx = []

    length = sum(condition)

    while i < len(condition):
        # print(i)
        if math.isnan(target_df.loc[i, "ed_profile"]) == False:

            matches.loc[0 : 0 + m + 1, j] = data.loc[i : i + m - 1, "scaled"]
            match_idx.append(i)
            j += 1

            i += m
        i += 1

    matches = matches.dropna(axis=1, how="all")
    shape_df = shape_df.reset_index()
    shape_df = shape_df.drop(columns=["index"])
    shape_df = pd.concat([shape_df, matches], axis=1)

    cols = [col for col in shape_df.columns]

    fig, axes = plt.subplots(nrows=2, ncols=1)

    # First subplot:

    target_df.plot(
        y=[param],
        ax=axes[0],
        color=["blue"],
        alpha=1,
        use_index=True,
        subplots=True,
    )
    target_df.plot(
        y=["shape"],
        ax=axes[0],
        color=color_1,
        alpha=1,
        use_index=True,
        subplots=True,
    )

    first_legend = axes[0].legend(loc="best", fancybox=True, framealpha=0.1)
    axes[0].add_artist(first_legend)

    axes[0].set_xlabel("index")
    # axes[0].axes.xaxis.set_visible(False)
    axes[0].set_ylabel(unit)
    # axes[0].set_title('{} [{}] - Euclidean Dist. Threshold = {}'.format(param,shape_type, threshold))

    # Second Subplot:

    target_df.plot(
        y=["dist"],
        ax=axes[1],
        color=["grey"],
        alpha=1,
        use_index=True,
        subplots=True,
        sharex=True,
    )
    second_legend = axes[1].legend(loc="best", fancybox=True, framealpha=0.1)
    axes[1].add_artist(second_legend)
    axes[1].set_ylabel("distance")
    axes[1].axhline(y=bottom)

    # Page Format;
    plt.subplots_adjust(
        left=0.1, bottom=0.05, right=0.9, top=0.9, wspace=0.1, hspace=0.1
    )

    axes[1].set_xlim(axes[0].get_xlim())
    # axes[1].set_ylim(axes[0].get_ylim())
    plt.savefig("shape_match", dpi=300, bbox_inches="tight")
    fig.tight_layout()
    plt.show()

    return (
        match_idx,
        target_df["shape"],
        target_df[param],
        target_df["ed_profile"],
        target_df["dist"],
    )


def kmeans_ts(data_scaled, k, slide_len, m, color_list):
    sensor = data_scaled.loc[:, "scaled"]
    sensor = sensor.interpolate(method="pad")
    #sensor.plot()
    sensor = sensor.values  # convert to array

    #%%## EXTRACT ALL SHAPES AND STORE IN DATA FRAME
    ss_df = []
    for start_pos in range(0, len(sensor), slide_len):
        end_pos = start_pos + m
        segment = np.copy(sensor[start_pos:end_pos])
        if len(segment) != m:
            continue
        ss_df.append(segment)

    ss_df = pd.DataFrame(ss_df)

    X = np.c_[ss_df]

    #%% Step 2: set up and create k-means model

    # define the number of clusters
    clusterer = KMeans(n_clusters=k)  # create model instance
    clusterer.fit(X)
    centroids = clusterer.cluster_centers_
    labels = clusterer.labels_

    centroids = pd.DataFrame(centroids)
    centroids = centroids.swapaxes("index", "columns")

    #%% Step 4: evalulate clustering

    # SSE
    SSE = clusterer.inertia_
    CHS = calinski_harabasz_score(X, labels)
    print("---------------")
    print("k = {:}".format(k))
    print("SSE = {:.2f}".format(SSE))

    #%% # Silhouette coefficient

    from sklearn import metrics

    SC = metrics.silhouette_score(X, labels)
    print("SC = {:.2f}".format(SC))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(X, labels)

    #%%
    # plot Silhouette plot
    # -------------------------------------------------
    import matplotlib.cm as cm

    y_lower = 0
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig, ax1 = plt.subplots(figsize=(8, 8))
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])

    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = color_list[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 0.0

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=SC, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    centroids.plot(ax=ax2, color=color_list)
    fig.suptitle(
        "The silhouette coefficient values for kmeans with k = {}".format(k)
    )
    plt.savefig("kmeans.png", dpi=300, bbox_inches="tight")
    plt.show()

    #%% PLOT OF EACH CLUSTER, DATAFRAME OF DISTANCES WITHIN CLUSTERS
    labels = pd.DataFrame(labels)
    # X_dist = clusterer.transform(X)**2
    X_dist = clusterer.transform(X)
    X_dist = pd.DataFrame(X_dist)

    # Series of distances from each centroid to cluster members:
    def cluster_distance(X_dist, labels, cluster_no):
        cluster_dist = X_dist.loc[:, cluster_no]
        cluster_dist = cluster_dist[labels.loc[:, 0] == cluster_no]
        return cluster_dist

    distance_dictionary = {}
    max_dist_dict = {}
    std_dict = {}
    thresh_dict = {}
    median_dict = {}
    for i in range(0, k):
        distance = cluster_distance(X_dist, labels, i)
        distance_dictionary["cluster_{0}_distance".format(i)] = distance
        max_dist_dict["{0}".format(i)] = distance.max()  # + (3*distance.std())
        std_dict["{0}".format(i)] = distance.std()
        max_dist_dict["{0}".format(i)] = distance.max()
        median_dict["{0}".format(i)] = distance.median()
        thresh_dict["{0}".format(i)] = distance.median() + (2 * distance.std())
        # thresh_dict["{0}".format(i)] = distance.max() + distance.std()

    k_dist_dataframe = pd.DataFrame.from_dict(distance_dictionary)
    dct = {k: [v] for k, v in max_dist_dict.items()}
    max_df = pd.DataFrame(dct)

    dct = {k: [v] for k, v in std_dict.items()}
    std_df = pd.DataFrame(dct)

    dct = {k: [v] for k, v in thresh_dict.items()}
    thresh_df = pd.DataFrame(dct)
    thresh_df = thresh_df.fillna(value=0)

    dct = {k: [v] for k, v in median_dict.items()}
    median_df = pd.DataFrame(dct)

    # Plot each centroid w/ matches
    for i in range(0, k):
        cluster_plt = ss_df.loc[labels[0] == i]
        cluster_plt = cluster_plt.reset_index(drop=True)
        cluster_plt = cluster_plt.transpose()

        fig, ax = plt.subplots(nrows=2, ncols=1)
        cluster_plt.plot(
            ax=ax[0],
            legend=False,
            color=color_list[i],
            alpha=0.3,
            title="Cluster {}".format(i),
        )
        centroids.iloc[:, i].plot(ax=ax[0], color=color_list[i], alpha=1)
        max_dist = round(max_df.iloc[0, i], 3)
        median = round(median_df.iloc[0, i], 3)
        std = round(std_df.iloc[0, i], 3)
        k_dist_dataframe.iloc[:, i].hist(
            ax=ax[1],
            bins=10,
            range=[0, max_df.iloc[0, i]],
            color=color_list[i],
            label="max dist. = {}, median = {}, stdev.= {}".format(
                max_dist, median, std
            ),
        )
        ax[1].legend()

        #%%# PRE-PROCESS SENSOR TS TO PLOT CLUSTER MEMBERS ON ORIGINAL TIME SERIES

    segment_class = []

    for i in range(0, len(labels[0])):
        start_pos = i * slide_len
        end_pos = start_pos + m
        cluster = labels.loc[i, 0]
        segment_class.append([cluster] * slide_len)

        if start_pos + slide_len > len(sensor) - m:
            # print(i,end_pos)
            gap = len(sensor) - (start_pos + slide_len)
            segment_class.append([cluster] * gap)

    segment_class = [
        element for nestedlist in segment_class for element in nestedlist
    ]
    segment_class = pd.DataFrame(segment_class)

    def get_kmeans_ts(ts, segment_class, i):
        ts_df = pd.DataFrame(ts, columns=[i]).copy()
        ts_df.iloc[segment_class != i] = np.nan
        ts_df = ts_df.squeeze()
        return ts_df

    # test = get_kmeans_ts(sensor,segment_class,0)

    kmeans_ts_dict = {}
    for i in range(0, k):
        kmeans_ts_dict["{0}".format(i)] = get_kmeans_ts(
            sensor, segment_class, i
        )

    ts_dataframe = pd.DataFrame.from_dict(kmeans_ts_dict)

    #%%# PLOT CLUSTER MEMBERS ON ORIGINAL TIME SERIES

    ts_dataframe.plot(color=color_list, alpha=1, use_index=True)

    # ax.legend(bbox_to_anchor=(1, 1.5), prop={"size":10})
    # ax.get_legend().remove()

    plt.savefig("Kmeans_ts.png", dpi=300, bbox_inches="tight")

    plt.show()

    #%%# Step 5: elbow method

    SSE = np.zeros(X.shape[0])  # initialize

    for K in range(1, 50):
        kmeans = KMeans(n_clusters=K)  # find clusters for K
        kmeans.fit(X)
        SSE[K - 1] = kmeans.inertia_
    # get SSE

    # plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        np.arange(1, SSE.shape[0] + 1),
        SSE,
        marker="o",
        color="b",
        markersize=10,
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.xlim(left=0, right=25)
    plt.title("K")
    plt.savefig("SSE.png", dpi=300, bbox_inches="tight")

    return thresh_df, std_df, centroids, SSE, CHS, SC


def kmedoids_ts(data_scaled, k, slide_len, m, color_list):
    sensor = data_scaled.loc[:, "scaled"]
    sensor = sensor.interpolate(method="pad")

    sensor = sensor.values  # convert to array

    #%%## EXTRACT ALL SHAPES AND STORE IN DATA FRAME
    ss_df = []
    for start_pos in range(0, len(sensor), slide_len):
        end_pos = start_pos + m
        segment = np.copy(sensor[start_pos:end_pos])
        if len(segment) != m:
            continue
        ss_df.append(segment)

    ss_df = pd.DataFrame(ss_df)

    X = np.c_[ss_df]

    #%% Step 2: set up and create k-means model

    # define the number of clusters
    clusterer = KMedoids(n_clusters=k)  # create model instance
    clusterer.fit(X)
    centroids = clusterer.cluster_centers_
    labels = clusterer.labels_

    centroids = pd.DataFrame(centroids)
    centroids = centroids.swapaxes("index", "columns")

    #%% Step 4: evalulate clustering

    # SSE
    SSE = clusterer.inertia_
    CHS = calinski_harabasz_score(X, labels)
    print("---------------")
    print("k = {:}".format(k))
    print("SSE = {:.2f}".format(SSE))

    #%% # Silhouette coefficient

    from sklearn import metrics

    SC = metrics.silhouette_score(X, labels)
    print("SC = {:.2f}".format(SC))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(X, labels)

    #%%
    # plot Silhouette plot
    # -------------------------------------------------
    import matplotlib.cm as cm

    y_lower = 0
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig, ax1 = plt.subplots(figsize=(8, 8))
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])

    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = color_list[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 0.0

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=SC, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    centroids.plot(ax=ax2, color=color_list)
    fig.suptitle(
        "The silhouette coefficient values for kmedoids with k = {}".format(k)
    )
    plt.savefig("kmed.png", dpi=300, bbox_inches="tight")
    plt.show()

    #%% PLOT OF EACH CLUSTER, DATAFRAME OF DISTANCES WITHIN CLUSTERS
    labels = pd.DataFrame(labels)
    # X_dist = clusterer.transform(X)**2
    X_dist = clusterer.transform(X)
    X_dist = pd.DataFrame(X_dist)

    # Series of distances from each centroid to cluster members:
    def cluster_distance(X_dist, labels, cluster_no):
        cluster_dist = X_dist.loc[:, cluster_no]
        cluster_dist = cluster_dist[labels.loc[:, 0] == cluster_no]
        return cluster_dist

    distance_dictionary = {}
    max_dist_dict = {}
    std_dict = {}
    thresh_dict = {}
    median_dict = {}
    for i in range(0, k):
        distance = cluster_distance(X_dist, labels, i)
        distance_dictionary["cluster_{0}_distance".format(i)] = distance
        max_dist_dict["{0}".format(i)] = distance.max()  # + (3*distance.std())
        std_dict["{0}".format(i)] = distance.std()
        max_dist_dict["{0}".format(i)] = distance.max()
        median_dict["{0}".format(i)] = distance.median()
        thresh_dict["{0}".format(i)] = distance.median() + (2 * distance.std())
        # thresh_dict["{0}".format(i)] = distance.max() + distance.std()

    k_dist_dataframe = pd.DataFrame.from_dict(distance_dictionary)
    dct = {k: [v] for k, v in max_dist_dict.items()}
    max_df = pd.DataFrame(dct)

    dct = {k: [v] for k, v in std_dict.items()}
    std_df = pd.DataFrame(dct)

    dct = {k: [v] for k, v in thresh_dict.items()}
    thresh_df = pd.DataFrame(dct)
    thresh_df = thresh_df.fillna(value=0)

    dct = {k: [v] for k, v in median_dict.items()}
    median_df = pd.DataFrame(dct)

    # Plot each centroid w/ matches
    for i in range(0, k):
        cluster_plt = ss_df.loc[labels[0] == i]
        cluster_plt = cluster_plt.reset_index(drop=True)
        cluster_plt = cluster_plt.transpose()

        fig, ax = plt.subplots(nrows=2, ncols=1)
        cluster_plt.plot(
            ax=ax[0],
            legend=False,
            color=color_list[i],
            alpha=0.3,
            title="Cluster {}".format(i),
        )
        centroids.iloc[:, i].plot(ax=ax[0], color=color_list[i], alpha=1)
        max_dist = round(max_df.iloc[0, i], 3)
        median = round(median_df.iloc[0, i], 3)
        std = round(std_df.iloc[0, i], 3)
        k_dist_dataframe.iloc[:, i].hist(
            ax=ax[1],
            bins=10,
            range=[0, max_df.iloc[0, i]],
            color=color_list[i],
            label="max dist. = {}, median = {}, stdev.= {}".format(
                max_dist, median, std
            ),
        )
        ax[1].legend()

        #%%# PRE-PROCESS SENSOR TS TO PLOT CLUSTER MEMBERS ON ORIGINAL TIME SERIES

    segment_class = []

    for i in range(0, len(labels[0])):
        start_pos = i * slide_len
        end_pos = start_pos + m
        cluster = labels.loc[i, 0]
        segment_class.append([cluster] * slide_len)
        if start_pos + slide_len > len(sensor) - m:
            # print(i,end_pos)
            gap = len(sensor) - (start_pos + slide_len)
            segment_class.append([cluster] * gap)

    segment_class = [
        element for nestedlist in segment_class for element in nestedlist
    ]
    segment_class = pd.DataFrame(segment_class)

    def get_kmeans_ts(ts, segment_class, i):
        ts_df = pd.DataFrame(ts, columns=[i]).copy()
        ts_df.iloc[segment_class != i] = np.nan
        ts_df = ts_df.squeeze()
        return ts_df

    # test = get_kmeans_ts(sensor,segment_class,0)

    kmeans_ts_dict = {}
    for i in range(0, k):
        kmeans_ts_dict["{0}".format(i)] = get_kmeans_ts(
            sensor, segment_class, i
        )

    ts_dataframe = pd.DataFrame.from_dict(kmeans_ts_dict)

    #%%# PLOT CLUSTER MEMBERS ON ORIGINAL TIME SERIES

    ts_dataframe.plot(color=color_list, alpha=1, use_index=True)

    # ax.legend(bbox_to_anchor=(1, 1.5), prop={"size":10})
    # ax.get_legend().remove()

    plt.savefig("Kmed_ts.png", dpi=300, bbox_inches="tight")

    plt.show()

    #%%# Step 5: elbow method

    SSE = np.zeros(X.shape[0])  # initialize

    for K in range(1, 50):
        kmeans = KMeans(n_clusters=K)  # find clusters for K
        kmeans.fit(X)
        SSE[K - 1] = kmeans.inertia_
    # get SSE

    # plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        np.arange(1, SSE.shape[0] + 1),
        SSE,
        marker="o",
        color="b",
        markersize=10,
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.xlim(left=0, right=25)
    plt.title("K")
    plt.savefig("SSE_kmed.png", dpi=300, bbox_inches="tight")

    return thresh_df, std_df, centroids, SSE, CHS, SC


def param_norm(param, data):
    if param == "Temperature":
        data["scaled"] = [
            (num - min_temp) / (max_temp - min_temp) for num in data[param]
        ]

    if param == "pH":
        data["scaled"] = [
            (num - min_pH) / (max_pH - min_pH) for num in data[param]
        ]

    if param == "ORP":
        data["scaled"] = [
            (num - min_orp) / (max_orp - min_orp) for num in data[param]
        ]

    if param == "Turbidity":
        data["scaled"] = [
            (num - min_turbidity) / (max_turbidity - min_turbidity)
            for num in data[param]
        ]

    if param == "fDOM":
        data["scaled"] = [
            (num - min_fdom) / (max_fdom - min_fdom) for num in data[param]
        ]

    if param == "Sp. Conductivity":
        data["scaled"] = [
            (num - min_cond) / (max_cond - min_cond) for num in data[param]
        ]

    if param == "Combined Chlorine":
        data["scaled"] = [
            (num - min_cl) / (max_cl - min_cl) for num in data[param]
        ]

    return data


def load_ts_data():

    color_list = [
        "purple",
        "red",
        "green",
        "violet",
        "pink",
        "blue",
        "black",
        "orange",
        "teal",
        "brown",
        "olive",
        "cyan",
        "yellow",
    ]

    #%%## IMPORT AND FORMAT DATA
    param = "Temperature"
    unit = "Deg. C"

    df_online_1 = param_norm(param, df_1)["scaled"]
    df_online_2 = param_norm(param, df_2)["scaled"]
    df_online_3 = param_norm(param, df_3)["scaled"]
    df_offline = param_norm(param, df_exo_2022)["scaled"]

    color_list = [
        "purple",
        "red",
        "green",
        "yellow",
        "pink",
        "blue",
        "black",
        "violet",
        "teal",
        "brown",
        "olive",
        "cyan",
        "orange",
    ]

    df_offline = df_offline.reset_index()
    sensor = df_offline.loc[:, "scaled"]
    sensor = sensor.interpolate(method="pad")
    #sensor.plot()
    sensor = sensor.values  # convert to array

    #%%## SPLIT TS INTO PATTERNS, STORE IN DATA FRAME
    pattern_df = []
    slide_len = 288
    hrs = 24
    m = hrs * 12
    for start_pos in range(0, len(sensor), slide_len):
        end_pos = start_pos + m
        # make a copy so changes to 'segments' doesn't modify the original ekg_data
        segment = np.copy(sensor[start_pos:end_pos])
        # if we're at the end and we've got a truncated segment, drop it
        if len(segment) != m:
            continue
        pattern_df.append(segment)
    pattern_df = pd.DataFrame(pattern_df)

    df_offline = df_offline.set_index(["Date Time"])

    return df_online_1, df_online_2, df_online_3, df_offline, pattern_df
