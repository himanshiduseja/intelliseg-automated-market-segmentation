# importing all the necessary libraries
import io
import os
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Setting GenAI Configuration
genai.configure(api_key='AIzaSyBucjGX9JSmQo1P2gMBl5l_EIx5nb8hlIU')


# Creating a Streamlit App
class MarketSegmentation:
    
    def __init__(self):
        """
        Initializes the Market Segmentation Class. Sets up initial attributes and initializes a GenerativeModel from the 'genai' module.

        Attributes:
            df (pd.DataFrame): DataFrame to hold the raw data.
            df_scaled (pd.DataFrame): DataFrame to hold scaled data.
            df_encoded (pd.DataFrame): DataFrame to hold encoded data.
            selected_features (list): List of features selected for processing.
            model (genai.GenerativeModel): Instance of the GenerativeModel from the 'genai' module.
        """
        try:
            self.df = None
            self.df_scaled = None
            self.df_encoded = None
            self.selected_features = None
            self.model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        
        except Exception as e:
            st.error(f"An error occurred during initialization: {e}")


    def page_style(self):
        
        """
        Configures the styling for the Streamlit page. Sets the page layout to wide and 
        applies custom CSS styles for various page elements.

        Customizations include:
        - Background image and color
        - Hiding certain default Streamlit components
        - Custom styles for title and text

        Uses Streamlit's markdown function to inject CSS styles and custom HTML into the page.
        """
        try:
            # Set page layout to wide
            st.set_page_config(layout="wide")
            st.title("IntelliSeg")

            # Inject custom CSS for page styling
            st.markdown(
                f"""
                <style>
                    /* Background color and image */
                    .stApp {{
                        
                        background-size: cover;
                        background-position: center;
                        background-repeat: no-repeat;
                        text-color: black;
                    }}

                    .main {{
                        overflow-x: hidden;
                    }}

                    #MainMenu {{visibility: hidden;}}
                    
                    header {{visibility: hidden;}}
                    
                    footer {{visibility: hidden;}}
                    
                    #root > div:nth-child(1) > div > div > div > div > section > div {{
                        padding-top: 1rem; 
                        padding-left: 2rem;
                        padding-right: 2rem; 
                        margin: 1rem;
                    }}
                    
                    .custom-title {{
                        font-family: 'Arial', sans-serif;
                        text-align: center;
                        margin-bottom: 18px;
                        letter-spacing: 1px;
                        text-transform: uppercase;
                        font-size: 45px;
                        font-weight: 700;
                        color:white;
                    }}

                    .custom-text {{
                        font-family: 'Arial', sans-serif;
                        font-size: 35px;
                        font-weight: bold;
                        text-align: center;
                        color: #DCA2F5;
                    }}

                    .st-emotion-cache-1erivf3{{
                        color:white;
                        font-size:16px;
                        font-weight:bold;
                        background: linear-gradient(90deg, hsla(236, 100%, 8%, 1) 0%, hsla(211, 100%, 28%, 1) 100%);
                    }}

                    .st-emotion-cache-19cfm8f{{
                        color:white;
                        font-size:16px;
                        font-weight:bold;
                        background: linear-gradient(90deg, hsla(236, 100%, 8%, 1) 0%, hsla(211, 100%, 28%, 1) 100%);
                    }}

                    .st-gj{{
                        color: white;
                        font-size: 14px;
                        font-weight:bold;
                        background:  linear-gradient(90deg, hsla(236, 100%, 8%, 1) 0%, hsla(211, 100%, 28%, 1) 100%);
                    }}


                </style>
                """,
                unsafe_allow_html=True
            )

            # Creating a title
            st.markdown("""<center class='custom-title'>Market Segmentation</center>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while applying page style: {e}")


    def load_data(self):
        """
        Allows the user to upload a CSV file through Streamlit's file uploader widget.
        Reads the uploaded file into a DataFrame if a file is provided.

        Attributes:
            uploaded_file (UploadedFile): The uploaded file object from Streamlit.
            df (pd.DataFrame): DataFrame to hold the data read from the CSV file.
        """
        try:
            # Create a file uploader widget for CSV files
            self.uploaded_file = st.file_uploader("Upload a CSV file for Segmentation", type="csv")
            
            if self.uploaded_file is not None:
                # Read the uploaded file into a DataFrame
                self.df = pd.read_csv(self.uploaded_file)
            else:
                st.warning("No file uploaded. Please upload a CSV file.")

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except pd.errors.ParserError:
            st.error("There was an error parsing the CSV file. Please ensure the file is correctly formatted.")
        except Exception as e:
            st.error(f"An error occurred while loading data: {e}")


    def display_data(self):
        """
        Displays the first and last 5 rows of the DataFrame in separate tabs using Streamlit.
        If the DataFrame is not loaded, displays a warning message.

        Attributes:
            df (pd.DataFrame): DataFrame containing the data to be displayed.
        """
        # Display a custom header
        st.markdown("""<span class="custom-text">Segmentation Data</span>""", unsafe_allow_html=True)

        try:
            if self.df is not None:
                # Create tabs for displaying the head and tail of the DataFrame
                tabs = st.tabs(["Head", "Tail"])

                # Display the first 5 rows in the 'Head' tab
                with tabs[0]:
                    st.table(self.df.head(5))

                # Display the last 5 rows in the 'Tail' tab
                with tabs[1]:
                    st.table(self.df.tail(5))
            else:
                st.warning("No data available. Please upload a CSV file first.")

        except Exception as e:
            st.error(f"An error occurred while displaying data: {e}")


    def data_information(self):
        """
        Displays information about the dataset, including metrics for rows, features, null values, 
        and duplicate values. Also provides a summary of data types and statistical descriptions.

        Attributes:
            df (pd.DataFrame): DataFrame containing the data to be analyzed.
        """
        st.subheader(" ")
        st.markdown("""<span class="custom-text">Dataset Information</span>""", unsafe_allow_html=True)
                
        try:
            if self.df is not None:
                # Create columns for displaying metrics
                columns = st.columns(4)
                
                with columns[0]:
                    st.metric(label="Total Rows", value=self.df.shape[0])

                with columns[1]:
                    st.metric(label="Total Features", value=self.df.shape[1])

                with columns[2]:
                    st.metric(label="Total Null Values", value=self.df.isnull().sum().sum())
                
                with columns[3]:
                    st.metric(label="Total Duplicate Values", value=self.df.duplicated().sum())

                st.text("")
                
                # Display data types information
                st.markdown(f"""<h5>Data types: </h5>""", unsafe_allow_html=True)
                info_df = pd.DataFrame({
                    "Non-Null Count": self.df.count(),
                    "Dtype": self.df.dtypes
                })
                st.table(info_df.T)
                st.text("")

                # Display statistical description of the dataset
                st.markdown(f"""<h5>Description: </h5>""", unsafe_allow_html=True)
                st.table(self.df.describe().iloc[1:,:])
            else:
                st.warning("No data available. Please upload a CSV file first.")

        except Exception as e:
            st.error(f"An error occurred while displaying data information: {e}")


    def type_casting(self):
        """
        Provides an option to convert selected columns in the DataFrame to numeric data types.
        The user can select columns from a multiselect widget, and the method attempts to
        convert these columns to numeric, coercing errors where necessary.

        Attributes:
            df (pd.DataFrame): DataFrame containing the data to be type-cast.
        """
        st.subheader(" ")
        st.markdown("""<span class="custom-text">Type Casting</span>""", unsafe_allow_html=True)
            
        try:
            # Checkbox to enable or disable type casting
            flag = st.checkbox("Want To Perform Type Casting")
            
            if flag:
                # Multiselect widget to choose columns for conversion
                num_columns = st.multiselect("Select columns to convert to numeric", self.df.columns)

                if num_columns:
                    # Convert selected columns to numeric
                    for col in num_columns:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    st.info("Selected columns have been converted to numeric.")
                    
                    # Display the first 5 rows of the updated DataFrame
                    st.table(self.df.head(5))
                else:
                    st.warning("No columns selected for conversion.")
            else:
                st.info("Type casting is not enabled.")

        except Exception as e:
            st.error(f"An error occurred while performing type casting: {e}")


    def data_cleaning(self):
        """
        Handles data cleaning tasks for the DataFrame, including:
        - Handling null values in categorical and numerical features.
        - Removing duplicate rows from the DataFrame.

        Attributes:
            df (pd.DataFrame): DataFrame containing the data to be cleaned.
        """
        st.subheader(" ")
        st.markdown("""<span class="custom-text">Data Cleaning</span>""", unsafe_allow_html=True)
            
        try:
            st.markdown("""<h5 style='color:#D88DF8'>1. Handling Null Data</h5>""", unsafe_allow_html=True)

            if self.df.isnull().sum().sum() > 0:
                # Get columns with null values
                columns_with_null = self.df.columns[self.df.isnull().any()]

                # Separate numerical and categorical columns with null values
                numerical_columns = self.df.select_dtypes(include=['number']).columns.intersection(columns_with_null)
                categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.intersection(columns_with_null)

                # Categorical Features Handling
                st.markdown("""<h6 style='color:white'>Categorical Features:</h6>""", unsafe_allow_html=True)
                if len(categorical_columns) > 0:
                    choice_cat = st.selectbox("Select How You Want to Handle Null Data in Categorical Features", ["Imputing Values", "Dropping Data"])

                    if choice_cat:
                        selected_cat_columns = st.multiselect("Select Categorical Columns to Apply", categorical_columns)

                        if choice_cat == "Dropping Data":
                            if selected_cat_columns:
                                self.df = self.df.dropna(subset=selected_cat_columns)
                                st.info("Selected categorical columns with null values are dropped.")
                            else:
                                self.df = self.df.dropna()
                                st.info("All rows with any null values are dropped.")

                        elif choice_cat == "Imputing Values":
                            if selected_cat_columns:
                                impute_categorical = st.selectbox(
                                    "Select How to Impute Selected Categorical Columns", 
                                    ["Mode"]
                                )
                                self.df[selected_cat_columns] = self.df[selected_cat_columns].fillna(self.df[selected_cat_columns].mode().iloc[0])
                                st.info("Null values in selected categorical columns are imputed.")
                else:
                    st.info("No Categorical Columns with Null Values")

                # Numerical Features Handling
                st.write("")
                st.markdown("""<h6 style='color:white'>Numerical Features:</h6>""", unsafe_allow_html=True)
                if len(numerical_columns) > 0:
                    choice_num = st.selectbox("Select How You Want to Handle Null Data in Numerical Features", ["Dropping Data", "Imputing Values"])
                
                    if choice_num:
                        selected_num_columns = st.multiselect("Select Numerical Columns to Apply", numerical_columns)

                        if choice_num == "Dropping Data":
                            if selected_num_columns:
                                self.df = self.df.dropna(subset=selected_num_columns)
                                st.info("Selected numerical columns with null values are dropped.")
                            else:
                                self.df = self.df.dropna()
                                st.info("All rows with any null values are dropped.")

                        elif choice_num == "Imputing Values":
                            if selected_num_columns:
                                impute_numerical = st.selectbox(
                                    "Select How to Impute Selected Numerical Columns", 
                                    ["Mean", "Median", "Mode", "Zero"]
                                )
                                if impute_numerical == "Mean":
                                    self.df[selected_num_columns] = self.df[selected_num_columns].fillna(self.df[selected_num_columns].mean())
                                elif impute_numerical == "Median":
                                    self.df[selected_num_columns] = self.df[selected_num_columns].fillna(self.df[selected_num_columns].median())
                                elif impute_numerical == "Mode":
                                    self.df[selected_num_columns] = self.df[selected_num_columns].fillna(self.df[selected_num_columns].mode().iloc[0])
                                elif impute_numerical == "Zero":
                                    self.df[selected_num_columns] = self.df[selected_num_columns].fillna(0)
                                st.info("Null values in selected numerical columns are imputed.")
                else:
                    st.info("No Numerical Columns with Null Values")

                # Display cleaned DataFrame
                st.write("")

            else:
                st.info("No Null Values in the Data")
            
            # Handling Duplicate Data
            st.markdown("""<h5 style="color:#D88DF8">2. Handling Duplicate Data</h5>""", unsafe_allow_html=True)

            if self.df.duplicated().sum() > 0:
                drop_duplicates = st.checkbox("Drop Duplicate Rows")
                if drop_duplicates:
                    self.df = self.df.drop_duplicates()
                    st.success("All duplicate rows have been dropped.")
                else:
                    st.info("Duplicate rows were not removed.")
            else:
                st.info("No Duplicate Rows Found")

        except Exception as e:
            st.error(f"An error occurred while cleaning data: {e}")
            # Handle specific exceptions if needed
            # e.g., KeyError, ValueError, etc.


    def feature_encoding(self):
        """
        Applies feature encoding techniques to the DataFrame and displays the correlation matrix.
        Supports two encoding methods: One-Hot Encoding and Label Encoding.

        Attributes:
            df (pd.DataFrame): DataFrame containing the data to be encoded.
            df_encoded (pd.DataFrame): DataFrame containing the encoded data.
        """
        st.subheader(" ")
        st.markdown("""<span class="custom-text">Feature Encoding</span>""", unsafe_allow_html=True)

        try:
            # Select encoding method
            encoding_method = st.selectbox("Select Encoding Method", ["One-Hot Encoding", "Label Encoding"])

            if encoding_method == "One-Hot Encoding":
                # Apply One-Hot Encoding
                self.df_encoded = pd.get_dummies(self.df)
                st.info("One Hot Encoding Applied")
            
            elif encoding_method == "Label Encoding":
                # Apply Label Encoding
                self.df_encoded = self.df.copy()
                label_encoder = LabelEncoder()
                for column in self.df_encoded.select_dtypes(include=['object']).columns:
                    self.df_encoded[column] = label_encoder.fit_transform(self.df_encoded[column])
                st.info("Label Encoding Applied")

                # Display Correlation Matrix
                st.subheader(" ")
                st.markdown("""<span class="custom-text">Correlation Matrix</span>""", unsafe_allow_html=True)
                corr_matrix = self.df_encoded.corr()
                st.table(corr_matrix)
        
        except Exception as e:
            st.error(f"An error occurred while encoding features: {e}")


    def feature_scaling(self):
        """
        Applies feature scaling techniques to the encoded DataFrame. Supports two scaling methods:
        Min-Max Scaling and Standard Scaling.

        Attributes:
            df_encoded (pd.DataFrame): DataFrame containing the encoded data.
            df_scaled (pd.DataFrame): DataFrame containing the scaled data.
        """
        try:
            st.subheader(" ")
            st.markdown("""<span class="custom-text">Feature Scaling</span>""", unsafe_allow_html=True)
            
            # Select scaling method
            scaling_method = st.selectbox("Select Scaling Method", ["Min-Max Scaling", "Standard Scaling"])
            self.df_scaled = self.df_encoded.copy()
            
            # Apply selected scaling method
            if scaling_method == "Standard Scaling":
                scaler = StandardScaler()
                numeric_cols = self.df_scaled.select_dtypes(include=['number']).columns
                self.df_scaled[numeric_cols] = scaler.fit_transform(self.df_scaled[numeric_cols])
                st.info("Standard Scaling Applied")

            elif scaling_method == "Min-Max Scaling":
                scaler = MinMaxScaler()
                numeric_cols = self.df_scaled.select_dtypes(include=['number']).columns
                self.df_scaled[numeric_cols] = scaler.fit_transform(self.df_scaled[numeric_cols])
                st.info("Min Max Scaling Applied")
            
            # Notify if no numeric columns were found
            if not numeric_cols.any():
                st.warning("No numeric columns found for scaling.")
        
        except Exception as e:
            st.error(f"An error occurred while scaling features: {e}")


    def clustering_plots(self):
        """
        Generates and displays clustering plots for K-Means clustering, including Elbow and Silhouette plots.
        The user can select features for clustering, choose the number of clusters, and view the WCSS and Silhouette scores.

        Attributes:
            df_scaled (pd.DataFrame): DataFrame containing scaled data for clustering.
            selected_features (list): List of selected features for clustering.
        """
        try:
            st.subheader(" ")
            st.markdown("""<span class="custom-text">K-Means Clustering</span>""", unsafe_allow_html=True)

            # Get all column names and use them as the default selected options
            all_features = self.df_scaled.columns.tolist()

            # Use multiselect with all features pre-selected by default
            self.selected_features = st.multiselect("Select Features for Clustering (Remove unwanted features)", all_features, default=all_features)
            k_clusters = st.select_slider("Select K Clusters", options=list(range(2, 15)), value=6)

            if self.selected_features:
                wcss = []
                ss = []

                pca = PCA(2)
                pca_encoded = pca.fit_transform(self.df_scaled[self.selected_features])

                for i in range(2, k_clusters + 1):
                    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
                    kmeans.fit(self.df_scaled[self.selected_features])
                    wcss.append(kmeans.inertia_)
                    
                    y = kmeans.predict(self.df_scaled[self.selected_features])
                    s = silhouette_score(pca_encoded, y, random_state=42)
                    ss.append(round(s, 5))

                wcss_ss = pd.DataFrame({
                    'Number of Clusters': list(range(2, k_clusters + 1)),
                    'WCSS': wcss,
                    "SS": ss
                })

                elbow, silhouette = st.columns(2)

                with elbow:
                    # Plot WCSS using Plotly Express with enhanced appearance
                    fig = px.line(
                        wcss_ss,
                        x='Number of Clusters',
                        y='WCSS',
                        title='Elbow Method for Optimal K',
                        labels={'Number of Clusters': 'Number of Clusters', 'WCSS': 'Within-Cluster Sum of Squares'},
                        markers=True
                    )

                    # Update layout to enhance appearance
                    fig.update_traces(
                        line=dict(width=3, color='#FFB942'),  # Thicken the line
                        marker=dict(size=12, color='#E67300')  # Increase marker size for better visibility
                    )

                    fig.update_layout(
                        xaxis=dict(
                            tickmode='linear',
                            tick0=2,
                            dtick=1
                        ),
                        yaxis=dict(
                            zeroline=False,
                            showgrid=False
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',  # Remove background color
                        paper_bgcolor='rgba(0,0,0,0)',  # Remove outer background color
                        font=dict(size=18),  # Increase font size for readability
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with silhouette:
                    # Plot Silhouette Scores using Plotly Express with enhanced appearance
                    fig = px.line(
                        wcss_ss,
                        x='Number of Clusters',
                        y='SS',
                        title='Silhouette Plot',
                        labels={'Number of Clusters': 'Number of Clusters', 'SS': 'Silhouette Score'},
                        markers=True
                    )

                    # Update layout to enhance appearance
                    fig.update_traces(
                        line=dict(width=3, color='lightgreen'),  # Thicken the line
                        marker=dict(size=12, color='darkgreen')  # Increase marker size for better visibility
                    )

                    fig.update_layout(
                        xaxis=dict(
                            tickmode='linear',
                            tick0=2,
                            dtick=1
                        ),
                        yaxis=dict(
                            zeroline=False,
                            showgrid=False
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',  # Remove background color
                        paper_bgcolor='rgba(0,0,0,0)',  # Remove outer background color
                        font=dict(size=18),  # Increase font size for readability
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("No features selected for clustering.")
        
        except Exception as e:
            st.error(f"An error occurred while generating clustering plots: {e}")


    def clustering(self):
        """
        Applies K-Means clustering to the dataset, performs PCA for dimensionality reduction, and generates
        2D and 3D scatter plots to visualize the clustering results. Users can select the number of clusters
        and the color scale for the plots.

        Attributes:
            df_encoded (pd.DataFrame): DataFrame containing encoded features for clustering.
            selected_features (list): List of selected features for clustering.
        """
        color_scale = [ 
            'amp','viridis', 'aggrnyl', 'agsunset', 'algae', 'armyrose', 'balance', 'blackbody', 'bluered', 
            'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 
            'curl', 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 
            'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 
            'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 
            'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 
            'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 
            'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 
            'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 
            'temps', 'thermal', 'tropic', 'turbo', 'twilight'
        ]

        try:
            cols = st.columns(2)
            with cols[0]:
                k = st.number_input("Select Number of Clusters", value=3, min_value=2, step=1)
            with cols[1]:
                color_ = st.selectbox("Select Cluster Color", color_scale)

            # Applying KMeans with k Clusters
            Clustering = KMeans(n_clusters=k, random_state=42).fit(self.df_encoded[self.selected_features])
            labels = Clustering.labels_
            self.df['Cluster'] = labels  # Adding these labels to the initial dataset for profiling

            # Perform PCA for 2D and 3D
            pca_2d = PCA(n_components=2)
            pca_3d = PCA(n_components=3)

            pca_2d_df = pca_2d.fit_transform(self.df_encoded[self.selected_features])
            pca_3d_df = pca_3d.fit_transform(self.df_encoded[self.selected_features])

            # Transform the cluster centers using the same PCA transformation
            centers_2d = pca_2d.transform(Clustering.cluster_centers_)
            centers_3d = pca_3d.transform(Clustering.cluster_centers_)

            # Create DataFrames for Plotly
            pca_2d_df = pd.DataFrame(pca_2d_df, columns=['PCA Component 1', 'PCA Component 2'])
            pca_3d_df = pd.DataFrame(pca_3d_df, columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
            pca_2d_df['Cluster'] = labels
            pca_3d_df['Cluster'] = labels

            st.text("")
            st.markdown(f"""<center><h3>KMeans Clustering With {k} Clusters</h3></center>""", unsafe_allow_html=True)

            # Create columns for side-by-side display
            col1, col2 = st.columns(2)

            # 2D Plot with Plotly
            with col1:
                fig_2d = px.scatter(
                    pca_2d_df,
                    x='PCA Component 2',
                    y='PCA Component 1',
                    color='Cluster',
                    color_continuous_scale=color_,
                    labels={'PCA Component 1': 'PCA Component 1', 'PCA Component 2': 'PCA Component 2'}
                )

                # Add centroids to the 2D plot
                centroids_2d_df = pd.DataFrame(centers_2d, columns=['PCA Component 1', 'PCA Component 2'])
                centroids_2d_df['Cluster'] = range(k)

                fig_2d.add_scatter(
                    x=centroids_2d_df['PCA Component 2'],
                    y=centroids_2d_df['PCA Component 1'],
                    mode='markers+text',
                    marker=dict(color='lightgreen', size=50, symbol='star', line=dict(width=1, color='Black')),
                    text=['Centroid'] * k,
                    textposition='top center',
                    showlegend=False
                )

                # Update layout for the 2D plot
                fig_2d.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',  # Remove plot background color
                    paper_bgcolor='rgba(0,0,0,0)',  # Remove figure background color
                    legend_title_text='Cluster',
                    legend_title_font_size=18,
                    legend_font_size=20,
                    xaxis_title='PCA Component 2',
                    yaxis_title='PCA Component 1',
                    height=600,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                )

                fig_2d.update_traces(marker=dict(size=13))  # Adjust the marker size here
                st.plotly_chart(fig_2d, use_container_width=True)

            # 3D Plot with Plotly
            with col2:
                fig_3d = px.scatter_3d(
                    pca_3d_df,
                    x='PCA Component 1',
                    y='PCA Component 2',
                    z='PCA Component 3',
                    color='Cluster',
                    color_continuous_scale=color_,
                    labels={
                        'PCA Component 1': 'PCA Component 1',
                        'PCA Component 2': 'PCA Component 2',
                        'PCA Component 3': 'PCA Component 3'
                    }
                )

                # Add centroids to the 3D plot
                centroids_3d_df = pd.DataFrame(centers_3d, columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
                centroids_3d_df['Cluster'] = range(k)

                fig_3d.add_scatter3d(
                    x=centroids_3d_df['PCA Component 1'],
                    y=centroids_3d_df['PCA Component 2'],
                    z=centroids_3d_df['PCA Component 3'],
                    mode='markers+text',
                    marker=dict(color='lightgreen', size=8, symbol='diamond', line=dict(width=1, color='Black')),
                    text=['Centroid'] * k,
                    textposition='top center',
                    showlegend=False
                )

                # Update layout for the 3D plot
                fig_3d.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',  # Remove plot background color
                    paper_bgcolor='rgba(0,0,0,0)',  # Remove figure background color
                    legend_title_text='Cluster',
                    legend_title_font_size=18,
                    legend_font_size=20,
                    height=600,
                )

                st.plotly_chart(fig_3d, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
            # Handle specific exceptions if needed
            # e.g., ValueError, IndexError, etc.

    
    def profiling_segments(self):
        """
        Profiles the segmented clusters by analyzing their characteristics based on categorical and numerical data.
        Generates a detailed report using the language model, highlighting unique features, behavioral, demographic,
        geographic, and psychographic segmentations, and provides a summary of key findings.

        Attributes:
            df (pd.DataFrame): DataFrame containing the dataset with clusters.
            model (object): Language model used for generating content based on the prompt.
        """
        
        try:
            st.subheader(" ")
            st.markdown("""<center><h1 style="color:#FFDD00">Profiling Segments</h1></center>""", unsafe_allow_html=True)
        
            # Identifying categorical columns (including Cluster)
            categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            categorical_columns.insert(0, 'Cluster')

            # Grouping data by the categorical columns and aggregating numerical features
            grouped_df = self.df.groupby(categorical_columns).agg({
                col: ['mean', 'median', 'min', 'max'] 
                for col in self.df.select_dtypes(include=['number']).columns
            })

            prompt = f"""
                Analyze the segmented clusters in the provided DataFrame, emphasizing each cluster's unique characteristics and differentiating factors.

                **Data:** {grouped_df.to_json()}

                using this data, perform profiling segments and return data (don't mention mean and median directly refer to the value) in the format mentioned below and don't mention about yourself.

                **Output Format:**
                #### <span style='color:lightgreen'>Clusters Information
                1. cluster 0 to cluster n-1 each one, denoting their features

                #### <span style='color:lightgreen'>Behavioral Segmentation (Show only if there is any)
                * Details for each cluster (Segments based on their buying behavior, usage rates, brand loyalty, purchase occasion, and benefits sought.)

                #### <span style='color:lightgreen'>Demographic Segmentation (Show only if there is any)
                * Details for each cluster (Divides based on measurable characteristics like age, gender, income, education, occupation, family size, etc.)

                #### <span style='color:lightgreen'>Geographic Segmentation (Show only if there is any)
                * Details for each cluster (Categorizes based on their geographic location, such as country, state, city, neighborhood, or climate.)

                #### <span style='color:lightgreen'>Psychographic Segmentation (Show only if there is any)
                * Details for each cluster (Groups based on their values, lifestyles, interests, attitudes, opinions, and personality traits.)

                #### Summarizing the segmentation
                * Summarize the key findings of the segmentation analysis, highlighting the most significant differences between clusters
                """
            cols = st.columns([5, 2, 5])
            with cols[1]:
                btn = st.button("Perform Profiling")
            
            if btn:
                response = self.model.generate_content(prompt)
                st.markdown(response.text, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred during profiling segments: {e}")
            

    def main(self):
        self.page_style()
        self.load_data()
        if self.uploaded_file is not None:
            self.display_data()
            self.data_information()
            self.type_casting()
            self.data_cleaning()
            self.feature_encoding()
            self.feature_scaling()
            self.clustering_plots()
            self.clustering()
            self.profiling_segments()


if __name__=="__main__":
    app = MarketSegmentation()
    app.main()
