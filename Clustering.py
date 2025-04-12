import pandas as pd
import numpy
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import probplot

#Inserting image
st.image("ML.jpeg", width=200)

#Import data at the global level
data=pd.read_csv('Country-data.csv')

#selecting the attributes....remove the attribute call country
select_attrib=data[['child_mort','exports','health','imports','income','inflation','life_expec','total_fer','gdpp']]

#Scaling the dataset for analysis. Scaling reduces the bias in the data
scaled=MinMaxScaler(feature_range=(0,1)) #technque for scaling data. There are other techniques. Please explore them
scaled_dataset=scaled.fit_transform(select_attrib) # here you fit and transform the scaling technique on the dataset

# this codes adds column heading to the dataset after scaling
col_header=['child_mort','exports','health','imports','income','inflation','life_expec','total_fer','gdpp']
scaled_data=pd.DataFrame(scaled_dataset,columns=col_header)

# Functions to create pages and place them in selectbox

def page1():
    st.header('Project description')
    '''HELP International is an international humanitarian NGO that is committed to fighting poverty and providing the people of backward countries with basic amenities and relief during the time of disasters and natural calamities. It runs a lot of operational projects from time to time along with advocacy drives to raise awareness as well as for funding purposes. 
After the recent funding programmes, they have been able to raise around $ 10 million. Now the CEO of the NGO needs to decide how to use this money strategically and effectively. The significant issues that come while making this decision are mostly related to choosing the countries that are in the direst need of aid. And this is where I come in as a data analyst. My job is to categorise the countries using some socio-economic and health factors that determine the overall development of the country. Then I need to suggest the countries which the CEO needs to focus on the most. read more here:https://www.kaggle.com/code/gauravduttakiit/categorize-countries-using-k-means-pca#Objectives 
'''
def page2():
    if st.checkbox("Original dataset"): #displaying data when a checkbox is selected
        st.write(data)

    if st.checkbox('Selected attributes'):  #displaying selected attribute data when a checkbox is selected
        st.write(select_attrib)

    if st.checkbox("Scaled data"): #displaying scaled data data when a checkbox is selected
        st.write(scaled_data)

    #importing data from the web
    csv_file=st.file_uploader("Click to upload your Csv file", type=['csv'])

    # Check if file is uploaded
    if csv_file is not None:
        # Read CSV file
        df = pd.read_csv(csv_file)

        # Display dataset
        st.subheader("Uploaded Dataset:")
        st.write(df)

        # Display dataset statistics
        st.subheader("Dataset Statistics:")
        st.write(df.describe())

        # Display dataset information
        st.subheader("Dataset Information:")
        st.write(df.info())
def page3():
    st.header("Exploratory Data Analysis")
    # Statistics for the selected attributes

    st.write("click on the check box to display")
    if st.checkbox('Descriptive of selected attributes'):
        st.write(select_attrib.describe())

    if st.checkbox('Heatmap of selected attributes'):
        cor=scaled_data.corr()

    #Heatmap showing the various correllation
        fig3, ax3=plt.subplots(figsize=(8, 6))
        sns.heatmap(cor, annot=True, cmap='coolwarm', square=True)
        plt.title('Heatmap of selected Attributes')
        st.pyplot(fig3)

    # Pairplots of scatter diagram
    if st.checkbox("Show Pairplots"):
        st.subheader("Pairplot")
        sns.pairplot(scaled_data,corner=True,diag_kind="kde")
        st.pyplot()
    #Boxplot for each attribute
    if st.checkbox("Show Boxplot"):
        st.subheader("BoxPlot")
        fig4, ax4 = plt.subplots(figsize=(10,7))
        sns.boxplot(scaled_data,ax=ax4)
        st.pyplot()

    # Histograms for each attribute
    if st.checkbox("Show Histograms"):
        st.subheader("Histograms")
        fig5, ax5 = plt.subplots()
        sns.histplot(scaled_data['health'], kde=True, ax=ax5)
        st.pyplot(fig5)

    # Scatterplot
    if st.checkbox("Show Scatterplot"):
        st.subheader("Scatterplot")
        x_axis = st.selectbox("Select X-axis feature", select_attrib.columns, index=0)
        y_axis = st.selectbox("Select Y-axis feature", select_attrib.columns, index=1)
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=scaled_data[x_axis], y=data[y_axis], hue=data["child_mort"], palette="viridis", ax=ax6)
        ax6.set_title(f"Scatterplot of {x_axis} vs {y_axis}")
        ax6.set_xlabel('x_axis')
        ax6.set_ylabel('y_axis')
        st.pyplot(fig6)


    # Quantile-Normal Plot (Q-Q Plot)
    if st.checkbox("Show Quantile-Normal Plot (Q-Q Plot)"):
        st.subheader("Quantile-Normal Plot")
        selected_feature = st.selectbox("Select feature for Q-Q Plot", select_attrib.columns)
        fig7, ax7 = plt.subplots(figsize=(10, 7))
        probplot(data[selected_feature], dist="norm", plot=ax7)
        ax7.set_title(f"Q-Q Plot for {selected_feature}")
        st.pyplot(fig7)

        # Skewness and Kurtosis
    # Skewness and Kurtosis
    if st.checkbox("Show Skewness and Kurtosis"):
        st.subheader("Skewness and Kurtosis")
        skewness = select_attrib.apply(lambda col: col.skew(), axis=0)
        kurtosis = select_attrib.apply(lambda col: col.kurtosis(), axis=0)

        # Create a DataFrame
        skew_kurt_table = pd.DataFrame({
            "Skewness": skewness,
            "Kurtosis": kurtosis
        })

        st.write(skew_kurt_table)

        # Line Plot for Trend Analysis
    if st.checkbox("Show Line Plot for Trends"):
        st.subheader("Trend Analysis")
        selected_feature = st.selectbox("Select a feature for line plot", select_attrib.columns)
        sorted_data = data.sort_values(by=selected_feature)
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.plot(sorted_data["country"], sorted_data[selected_feature], marker="o", linestyle="-", color="green")
        ax8.set_title(f"Trend of {selected_feature}")
        ax8.set_xlabel("Country")
        ax8.set_ylabel(selected_feature)
        plt.xticks(rotation=90)
        st.pyplot(fig8)

        # Attribute Distribution
    if st.checkbox("Show Feature Distributions"):
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select a feature for distribution plot", select_attrib.columns)
        fig9, ax9 = plt.subplots(figsize=(8, 6))
        sns.histplot(data[selected_feature], kde=True, ax=ax9, color="blue", bins=20)
        ax9.set_title(f"Distribution of {selected_feature}")
        st.pyplot(fig9)

    # Quartile Ranges (Q1, Q3, and IQR)
    if st.checkbox("Show Quartile Ranges (Q1, Q3, and IQR)"):
        st.subheader("Quartile Ranges and IQR")

        # Compute Q1, Q3, and IQR for all features
        quartiles = select_attrib.apply(lambda col: (col.quantile(0.25), col.quantile(0.75)), axis=0)
        iqr = select_attrib.apply(lambda col: col.quantile(0.75) - col.quantile(0.25), axis=0)

        # Create a DataFrame to display results
        quartile_table = pd.DataFrame({
            "Q1 (25%)": quartiles.apply(lambda x: x[0]),
            "Q3 (75%)": quartiles.apply(lambda x: x[1]),
            "IQR": iqr
        })

        st.write(quartile_table)



def page4():
    st.header("K-Means Clustering")
    cluster_options=[2,3,4,5,6]

    #create a selectbox
    n_clusters=st.selectbox("Select your preferred number of clusters",cluster_options)

    #Kmeans clustering
    kmeansalg=KMeans(n_clusters= n_clusters, random_state=30)
    kmeansalg.fit(scaled_data)

    #diplaying the output. A new column has been added to the origainal data indicating the clusters.
    st.write("Check the last column of this table for the clusters: ")
    data['Kmeans Cluster']=kmeansalg.labels_
    st.write(data)

    #Evaluation of the algorithm
    sil=silhouette_score(scaled_data,kmeansalg.labels_) # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is",sil*100)

    #Visualisation the cluster points...Matplotib and seanborn are both 2D
    st.subheader("Imports VS Exports")
    fig,ax=plt.subplots(figsize=(10,7))
    ax.scatter(scaled_data['income'],scaled_data['inflation'], c=data['Kmeans Cluster'], cmap='viridis')
    ax.set_title('Income vs inflation datapoints')
    ax.set_xlabel('Income')
    ax.set_ylabel('inflation')
    st.pyplot(fig)

    # visualisation the cluster points...Matplotib and seanborn are both 2D
    st.subheader("Income VS Inflation")
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.scatter(scaled_data['imports'], scaled_data['exports'], c=data['Kmeans Cluster'], cmap='viridis')
    ax1.set_title('Import vs Export datapoints')
    ax1.set_xlabel('Imports')
    ax1.set_ylabel('Export')
    st.pyplot(fig1)

def page5():
    st.header("Hierachichal Clustering")

    # Agglomerative Clustering
    st.subheader("Agglomerative Clustering")

    # Define a list of options for the number of clusters
    cluster_options1 = [2, 3, 4, 5, 6]

    # Create a selectbox widget to let the user choose the number of clusters
    n_clusters = st.selectbox("Select your preferred number of clusters", cluster_options1)

    # Compute the pairwise distances between all samples in the scaled data
    distances = pdist(scaled_data, metric='euclidean')

    # Compute the linkage matrix using the 'single' linkage method
    # The linkage matrix represents the hierarchical clustering structure of the data
    #note the following linkage{‘ward’, ‘complete’, ‘average’, ‘single’}
    linkage_matrix = linkage(distances, method='ward')

    # Create an AgglomerativeClustering instance with the chosen number of clusters and 'ward' linkage
    Agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

    # Fit the AgglomerativeClustering instance to the scaled data
    Agg.fit(scaled_data)
    data['Agg Cluster'] = Agg.labels_     # Predict the cluster labels for each sample in the data
    st.write(data)

    # Write a title for the dendrogram plot
    st.write("Dendrogram for Agglomerative Clustering")
    # Create a new figure and axis object for the dendrogram plot
    fig2, ax2 = plt.subplots()
    # Plot the dendrogram using the linkage matrix
    dendrogram(linkage_matrix, ax=ax2)
    # Set the title, x-axis label, and y-axis label for the dendrogram plot
    ax2.set_title("Dendrogram for Agglomerative Clustering")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Distance')
    st.pyplot(fig2)

    # Evaluation of the algorithm
    sil1 = silhouette_score(scaled_data, Agg.labels_)  # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is", sil1 * 100)

    # Divisive
    div = BisectingKMeans(n_clusters=n_clusters)
    div.fit(scaled_data)
    data['Div Cluster'] = div.labels_
    st.write(data)

def page6():
    st.header("Principal Component Analysis")

def page7():
    st.subheader("Association Rules Mining")

#linking pages to the sidebar
pages={
    'Project Description':page1,
    'Loading Dataset':page2,
    'Exploratory Data Analysis':page3,
    'K-means': page4,
    'Hierarchical Clustering': page5,
    'Principal Component Analysis': page6,
    'Association Rules Mining': page7
}

# Sidebar items
select_page=st.sidebar.selectbox("select a page", list(pages.keys()))
st.sidebar.header (f'Unsupervised Learning Project')
st.sidebar.write('Unsupervised learning is a type of machine learning where algorithms are trained on unlabeled data to discover patterns, relationships, or groupings within the data')

#show the page
pages[select_page]()
