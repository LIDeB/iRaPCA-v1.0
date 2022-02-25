# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:26:28 2021

@author: LIDeB UNLP
"""

# iRaPCA WebApp

#%%
# Needed packages
import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from rdkit import Chem
import numpy as np
from statistics import mean, stdev
from multiprocessing import freeze_support
from mordred import Calculator, descriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from validclust import dunn
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as plt1
import random
from datetime import date
from molvs import Standardizer


#%%

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='LIDEB Tools - Clustering ',
    layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

from PIL import Image
image = Image.open('cropped-header-irapca.png')
st.image(image)

st.write("[![Website](https://img.shields.io/badge/website-LIDeB-blue)](https://lideb.biol.unlp.edu.ar)[![Twitter Follow](https://img.shields.io/twitter/follow/LIDeB_UNLP?style=social)](https://twitter.com/intent/follow?screen_name=LIDeB_UNLP)")
st.subheader(":pushpin:" "About Us")
st.markdown("We are a drug discovery team with an interest in the development of publicly available open-source customizable cheminformatics tools to be used in computer-assisted drug discovery. We belong to the Laboratory of Bioactive Compounds Research and Development (LIDeB) of the National University of La Plata (UNLP), Argentina. Our research group is focused on computer-guided drug repurposing and rational discovery of new drug candidates to treat epilepsy and neglected tropical diseases.")


# Introduction
#---------------------------------#

st.write("""
# LIDeB Tools - iRaPCA v1.0

iRaPCA Clustering is a clustering strategy based on an iterative combination of the random subspace approach (feature bagging),
dimensionality reduction through Principal Component Analysis (PCA) and the k-means algorithm. The optimal number of clusters k
and the best subset of descriptors are selected from plots of silhouette coefficient against different k values and subsets.
Different validation metrics can be downloaded once the process has finished. A number of graphs may be built and readily downloaded
through a simple click. 

The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Mordred](https://github.com/mordred-descriptor/mordred), [Scikit-learn](https://scikit-learn.org/stable/), [Plotly](https://plotly.com/python/), [MolVS](https://github.com/mcs07/MolVS), [validclust](https://github.com/crew102/validclust)

The next workflow summarizes the steps performed by this method:
    
""")

image = Image.open('clustering_irapca_workflow.png')
st.image(image, caption='Clustering Workflow')

#st.subheader(":rocket:" "**Fast Tutorial** " "[iRaPCA](https://www.youtube.com/watch?v=6HoBdFDY9Ic&ab_channel=LIDeBUNLP)")
#st.markdown(" ")

#st.markdown("""
#         **To cite the application, please reference XXXXXXXXX**
#         """)

#%%

########### OPTIONS #######
# SIDEBAR

st.sidebar.header('Molecular descriptors')
molecular_descriptors = st.sidebar.checkbox('Check ONLY if you have previously calculated the molecular descriptors')
if molecular_descriptors == True:
    uploaded_file_1 = st.sidebar.file_uploader("Upload your molecular descriptors in a TXT file. Your file should have a column called 'NAME'", type=["txt"])
    st.sidebar.markdown("""
    [Example TXT molecular descriptor file](https://raw.githubusercontent.com/Capigol/iRaPCA_v1/main/example_descriptors.txt)
    """)

else:
    st.sidebar.header('Upload your SMILES')
    uploaded_file_1 = st.sidebar.file_uploader("Upload a CSV file with one SMILES per line", type=["csv"])
    st.sidebar.write("SMILES must be contained in the first column, or in a column named SMILES in any position")
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/Capigol/iRaPCA_v1/main/example_molecules.csv)
    """)

clustering_setting = st.sidebar.checkbox('Check to change the default configuration')
if clustering_setting == True:
    st.sidebar.header('Dimensionality reduction')    
    threshold_variance = st.sidebar.slider('Threshold variance', 0.00, 0.20, 0.05, 0.01)
    random_subspace_seed = st.sidebar.checkbox('Random seed',value=False)
    num_subsets = st.sidebar.slider('Nº of subsets', 50, 250, 100, 50)
    num_descriptores = st.sidebar.slider('Nº subset descriptors', 50, 300, 200, 50)    
    coef_correlacion = st.sidebar.selectbox("correlation coefficient", ("pearson", "kendall","spearman"),0)
    limite_correlacion = st.sidebar.slider('Threshold correlation', 0.0, 1.0, 0.4, 0.05)
    min_desc_subset = st.sidebar.slider('Min nº of descriptors for subset', 4, 10, 4, 1)
    max_desc_subset = st.sidebar.slider('Max nº of descriptors for subset', 10, 50, 25, 5)
    min_n_clusters = st.sidebar.slider('Min nº of clusters by round', 2, 10, 2, 1)
    max_n_clusters = st.sidebar.slider('Max nº of clusters by round', 20, 50, 25, 5)
    range_n_clusters = list(range(min_n_clusters,max_n_clusters,1))
    maximo_porcentaje_del_total = st.sidebar.slider('Max relation "cluster/total"', 0.0, 1.0, 0.3, 0.1)
    vueltas_maximas = st.sidebar.slider('Max nº of rounds', 1, 10, 5, 1)
    num_pca = st.sidebar.slider('PCAs', 2, 3, 2, 1)
    smiles_standardization = st.sidebar.checkbox('Standardization of SMILES',value=True)

    ignore_error = st.sidebar.checkbox('Ignore error in SMILES',value=True)
    st.sidebar.header('Type of Plots')
    graficar_sunburnt = st.sidebar.checkbox('Sunburn',value=True)
    graficar_silhouette = st.sidebar.checkbox('Silhouette vs k',value=True)
    graficar_scatter = st.sidebar.checkbox('Scatter',value=True)

# Default configuration:    
else:   
    smiles_standardization = True
    threshold_variance = 0.05
    random_subspace_seed = False
    num_subsets = 100
    num_descriptores = 200
    coef_correlacion = "pearson"
    limite_correlacion = 0.4
    min_desc_subset = 4
    max_desc_subset = 25
    min_n_clusters = 2
    max_n_clusters= 25
    range_n_clusters = list(range(min_n_clusters,max_n_clusters,1))
    maximo_porcentaje_del_total = 0.3
    vueltas_maximas = 5
    num_pca = 2
    ignore_error = True
    # Plots
    graficar_sunburnt = True
    graficar_silhouette = True
    graficar_scatter = True

    
st.sidebar.title(":speech_balloon: Contact Us")
st.sidebar.info(
"""
If you are looking to contact us, please
[:e-mail:](mailto:lideb@biol.unlp.edu.ar) or [Twitter](https://twitter.com/LIDeB_UNLP)
""")
    
    
#%%

### Reading/calculating molecular descriptors ###

def calcular_descriptores(uploaded_file_1,descriptores_calculados):
    
    if molecular_descriptors == True:
        descriptores = pd.read_csv(uploaded_file_1, sep='\t', delimiter=None, header='infer', names=None)
        molecules_names = descriptores['NAME'].tolist()
        descriptores.drop(['NAME'], axis=1,inplace=True)
        lista_nombres = []
        for i,name in enumerate(molecules_names):
            nombre = f'Molecule_{i+1}'
            lista_nombres.append(nombre)
        descriptores['NAME'] = lista_nombres
        descriptores.set_index("NAME",inplace=True)
        descriptores = descriptores.reindex(sorted(descriptores.columns), axis=1)
        descriptores.replace([np.inf, -np.inf], np.nan, inplace=True)
        descriptores = descriptores.apply(pd.to_numeric, errors = 'coerce')
        descriptores = descriptores.dropna(axis=0,how="all")
        descriptores = descriptores.dropna(axis=1) 
        
        st.write("The initial dataset has " + str(descriptores.shape[0]) + " molecules and " + str(descriptores.shape[1]) + " descriptors")
        return descriptores
    else:
        data1x = pd.DataFrame()
        suppl = []
        st.markdown("**Step 1: Standarization and descriptor calculation**")
        df_initial = pd.read_csv(uploaded_file_1)
        if "SMILES" in df_initial:
            list_of_smiles = df_initial["SMILES"]
        else:
            list_of_smiles = df_initial.iloc[:, 0]
        s = Standardizer()

        i = 0
        t = st.empty()
        problematic_smiles = []
        rows_to_retain = []

        for molecule in list_of_smiles:
            if smiles_standardization == True:
                t.markdown("Standardizing molecules: " + str(i+1) +"/" + str(len(list_of_smiles)))
            i = i+1
            try:
                mol = Chem.MolFromSmiles(molecule)
                if smiles_standardization == True:
                # estandarizada = s.super_parent(mol)
                    mol_sin_fragmento = s.fragment_parent(mol) #Return the fragment parent of a given molecule, the largest organic covalent unit in the molecule
                    mol_sin_estereo = s.stereo_parent(mol_sin_fragmento, skip_standardize= True) #Return The stereo parentof a given molecule, has all stereochemistry information removed from tetrahedral centers and double bonds.
                    mol_sin_carga = s.charge_parent(mol_sin_estereo, skip_standardize= True) #Return the charge parent of a given molecule,  the uncharged version of the fragment parent
                    estandarizada = s.isotope_parent(mol_sin_carga, skip_standardize= True) #Return the isotope parent of a given molecule, has all atoms replaced with the most abundant isotope for that element.
                    suppl.append(estandarizada)
                    rows_to_retain.append(i -1)
                else:
                    suppl.append(mol)

            except:
                problematic_smiles.append(i)
        if ignore_error == False and len(problematic_smiles) > 0:
            st.error("*Oh no! There is a problem with descriptor calculation of some SMILES.*  :confused:")
            st.markdown(f"*Please check your SMILES number: {str(problematic_smiles)}*")
            st.stop()
        
        else:
            if len(problematic_smiles) > 0 and smiles_standardization == True:
                st.markdown("Lines " + str(problematic_smiles) + " have problematic (or empty) SMILES. We have omitted them.")
            else:
                pass
            
        calc = Calculator(descriptors, ignore_3D=True) 
        t = st.empty()

        for i,mol in enumerate(suppl):
            if __name__ == "__main__":
                    if mol != None:
                        try:
                            freeze_support()
                            descriptor1 = calc(mol)
                            resu = descriptor1.asdict()
                            solo_nombre = {'NAME' : f'SMILES_{i+1}'}
                            solo_nombre.update(resu)

                            solo_nombre = pd.DataFrame.from_dict(data=solo_nombre,orient="index")
                            data1x = pd.concat([data1x, solo_nombre],axis=1, ignore_index=True)
                           
                            t.markdown("Calculating descriptors " + str(i+1) +"/" + str(len(suppl))) 
                            if smiles_standardization == False:
                                rows_to_retain.append(i)
                        except:
                            st.error("**Oh no! There is a problem with descriptor calculation of some SMILES.**  :confused:")
                            st.markdown("**Please check your SMILES number: **" + str(i+1))
                            st.stop()
                    else:
                        pass
        previuos_data = df_initial.iloc[rows_to_retain]

        t.markdown("Descriptor calculation have FINISHED")
        data1x = data1x.T
        descriptores = data1x.set_index('NAME',inplace=False).copy()
        descriptores = descriptores.reindex(sorted(descriptores.columns), axis=1)   
        descriptores.replace([np.inf, -np.inf], np.nan, inplace=True)
        descriptores = descriptores.apply(pd.to_numeric, errors = 'coerce') 
        descriptores = descriptores.dropna(axis=0,how="all")
        descriptores = descriptores.dropna(axis=1)
        st.write(str(descriptores.shape[0]) + " molecules were loaded correctly and " + str(descriptores.shape[1]) + " descriptors were calculated")
        
        return descriptores,previuos_data 
    
        

#%%

### Removing low variance descriptors ###

def descriptores_baja_variancia(descriptores, vuelta, threshold_variance: float):
    selector = VarianceThreshold(threshold_variance)       
    selector.fit(descriptores)                              
    descriptores_ok = descriptores[descriptores.columns[selector.get_support(indices=True)]]
    if vuelta == 1:
        st.write(str(descriptores_ok.shape[1]) + " descriptors have passed the variance threshold")
        st.markdown("**Step 2: Clustering**")
        
    return descriptores_ok

#%%

### Subsetting ###

def generar_subset(descriptores_ok, num_subsets: int, coef_correlacion: str, limite_correlacion: float, vuelta):
    subsets_ok=[]
    i=0
    t = st.empty()
    while (i < num_subsets): 
        if random_subspace_seed == True:
            subset= descriptores_ok.sample(num_descriptores,axis=1)
        else:
            subset= descriptores_ok.sample(num_descriptores,axis=1,random_state=i)  
        corr_matrix = subset.corr(coef_correlacion).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > limite_correlacion)]
        curado=subset.drop(subset[to_drop], axis=1)
        total_molec_subset = curado.shape[0]
        t.markdown("Generating descriptor subsets: " + str(i+1) +"/" + str(num_subsets))
        i = i+1
        subsets_ok.append(curado)
    return subsets_ok, total_molec_subset

#%%

### Normalization ###

def normalizar_descriptores(subset):
    descriptores_sin_normalizar = pd.DataFrame(subset)
    scaler = MinMaxScaler()
    descriptores_normalizados = pd.DataFrame(scaler.fit_transform(descriptores_sin_normalizar)) 
    return descriptores_normalizados

#%%
### Clustering ###

def PCA_clustering(descriptores_normalizados, range_n_clusters, num_pca: float, siluetas):

    sil_coef_grafica = []
    for n_clusters in range_n_clusters:
        pca = PCA(n_components = num_pca)
        pcas = pd.DataFrame(pca.fit_transform(descriptores_normalizados))
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(pcas)
        silhouette_avg = silhouette_score(pcas, cluster_labels)
        sil_coef_grafica.append(silhouette_avg)

    siluetas.append(sil_coef_grafica)

    return siluetas

def clustering(subsets_ok, min_desc_subset: int, max_desc_subset: int, range_n_clusters, num_pca: int):
    siluetas = []
    subsets_seleccionados = []
    t = st.empty()
    for i, subset in enumerate(subsets_ok):
        t.markdown("Clustering subsets: " + str(i+1) +"/" + str(num_subsets))
        if min_desc_subset < len(subset.columns) < max_desc_subset:
            descriptores_normalizados = normalizar_descriptores(subset)
            if max_n_clusters > len(descriptores_normalizados.index):
                
                range_n_clusters = list(range(min_n_clusters,len(descriptores_normalizados.index),1))
            siluetas = PCA_clustering(descriptores_normalizados, range_n_clusters, num_pca, siluetas)
            subsets_seleccionados.append(i)

    tabla_final = pd.DataFrame(siluetas).T
    tabla_final.columns = subsets_seleccionados
    tabla_final.index = range_n_clusters
    return tabla_final, subsets_seleccionados

#%%

### Plot Silhouette coefficient vs K for each subset ###

def grafica_silhouette(subsets_seleccionados,tabla_final,num_pca: int, range_n_clusters, limite_correlacion: float):
    
    if graficar_silhouette:
        fig = go.Figure()
        for num in subsets_seleccionados:
            fig.add_trace(go.Scatter(x=range_n_clusters, y=tabla_final[num], 
                            mode='lines+markers', name= f'Subset {num}', 
                            hovertemplate = "Subset = %s<br>Clusters = %%{x}<br>Silhouette = %%{y} <extra></extra>" % num))
        
        fig.update_layout(title = 'Number of clusters for K-means vs Silhouette coefficient',
                          plot_bgcolor = 'rgb(256,256,256)',
                          title_font = dict(size=25, family='Calibri', color='black'),
                          legend_title_text = "Subsets", 
                          legend_title_font = dict(size=18, family='Calibri', color='black'),
                          legend_font = dict(size=15, family='Calibri', color='black'))
        fig.update_xaxes(title_text='K (number of clusters)', range = [1.5, 20.5],
                         showline=True, linecolor='black', gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                         tickfont=dict(family='Arial', size=16, color='black'),
                         title_font = dict(size=20, family='Calibri', color='black'))
        fig.update_yaxes(title_text='SIL coefficient',
                         showline=True, linecolor='black', gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                         tickfont=dict(family='Arial', size=16, color='black'),
                         title_font = dict(size=20, family='Calibri', color='black'))

        st.plotly_chart(fig)
    return
#%%

### Clusters adittional information ###

def moleculas_en_cluster_PCA_clustering(subset_seleccionado, num_pca: int, cluster_mejor: int, subset_mejor: int, clusters_padre, vuelta, descriptores):
    
    subset_seleccionado_normalizado = normalizar_descriptores(subset_seleccionado)
    pca = PCA(n_components = num_pca)
    pcas = pd.DataFrame(pca.fit_transform(subset_seleccionado_normalizado))
    pcas = pcas.set_index(subset_seleccionado.index)
    for j,i in enumerate(range(num_pca),start=1):
        pcas.rename(columns = {i: "PCA_" + str(j)}, inplace=True)
    kmeans_new = KMeans(n_clusters=cluster_mejor, random_state=10).fit(pcas)
    df_molecula_cluster_actual = pd.DataFrame(kmeans_new.fit_predict(pcas))
    df_molecula_cluster_actual.rename(columns={0: 'CLUSTER'},inplace = True)
    df_molecula_cluster_actual['CLUSTER'] = df_molecula_cluster_actual['CLUSTER'] + 1    
    df_molecula_cluster_actual.index = subset_seleccionado.index.tolist()
    
    # ordeno los cluster por tamaño para que el mas grande sea el 1
    cluster_ordenados = []
    df_contado = pd.DataFrame(df_molecula_cluster_actual['CLUSTER'].value_counts())
    df_contado['cluster_nuevo'] = list(range(1, len(df_contado)+1))
    for j in range(len(df_molecula_cluster_actual)):
        for i in range(1, len(df_contado)+1):
            if df_molecula_cluster_actual['CLUSTER'][j] == i:
                cluster_ordenados.append(df_contado['cluster_nuevo'][i])
    df_molecula_cluster_actual['CLUSTER'] = cluster_ordenados    
    
    if vuelta == 1:
        df_cluster_padre = pd.DataFrame(pd.Series([cluster_actual for cluster_actual in df_molecula_cluster_actual['CLUSTER']]))
    else:
        lista_nombre_cluster_padre = [[str(clusters_padre), str(cluster_actual)] for cluster_actual in df_molecula_cluster_actual['CLUSTER']]
        df_cluster_padre = pd.DataFrame(pd.Series(['.'.join(nombre_cluster_padre) for nombre_cluster_padre in lista_nombre_cluster_padre]))
    df_cluster_padre.rename(columns={0: 'Cluster, padre'},inplace = True)
    df_cluster_padre.index = df_molecula_cluster_actual.index.values

    df_cluster_con_cluster_padre = pd.merge(df_molecula_cluster_actual, df_cluster_padre, left_index = True, right_index= True)
    df_subset_PCA = pd.merge(subset_seleccionado, pcas, left_index = True, right_index= True)
    moleculas_cluster = pd.merge(df_subset_PCA, df_cluster_con_cluster_padre, left_index = True, right_index= True)

    final_conteo = pd.DataFrame(moleculas_cluster['Cluster, padre'].value_counts())
    final_conteo.rename(columns = {'Cluster, padre':'Molecules'}, inplace = True)
    final_conteo.index.names = ['Cluster']
    final_conteo['Relacion'] = final_conteo['Molecules']/descriptores.shape[0]
    return pcas, moleculas_cluster, final_conteo

#%%

### Scatter plot with PCAs for each selected subset and K ###

def grafica_scatter(moleculas_cluster,subset_mejor,cluster_mejor):
    if graficar_scatter:
        tabla_final_moleculas = moleculas_cluster.copy()
        tabla_final_moleculas.rename(columns = {'PCA_1': 'PC_1', 'PCA_2': 'PC_2', 'Cluster, padre': 'Cluster'}, inplace = True)
        tabla_final_moleculas['Cluster'] = tabla_final_moleculas['Cluster'].astype(str)
        
        fig2 = plt1.scatter(tabla_final_moleculas, x = 'PC_1', y = 'PC_2', color = 'Cluster',
                           hover_name = tabla_final_moleculas.index, 
                           title = f'Scatter Plot of PC 1 vs PC 2 for subset {subset_mejor} and K {cluster_mejor}')
        fig2.update_layout(legend_title="Cluster", plot_bgcolor = 'rgb(256,256,256)',
                           title_font = dict(size=25, family='Calibri', color='black'),
                           legend_title_font = dict(size=18, family='Calibri', color='black'),
                           legend_font = dict(size=15, family='Calibri', color='black'))
        fig2.update_traces(marker=dict(size=15, line=dict(width=1)))
        fig2.update_xaxes(title_text="PC 1", showline=True, linecolor='black', 
                          gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                          tickfont=dict(family='Arial', size=16, color='black'),
                          title_font = dict(size=20, family='Calibri', color='black'))
        fig2.update_yaxes(title_text="PC 2", showline=True, linecolor='black', 
                          gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                          tickfont=dict(family='Arial', size=16, color='black'),
                          title_font = dict(size=20, family='Calibri', color='black'))     
        st.plotly_chart(fig2)
    return
      
#%%

### Random cluster evaluations ###

def cluster_random(pcas, molec_name,cluster_mejor):
    compilado_silhoutte = []
    compilado_db = []
    compilado_ch = []
    compilado_dunn = []
    
    for i in range(500):
        random.seed(a=i, version=2)
        random_clusters = []
        for x in molec_name:
            random_clusters.append(random.randint(0,cluster_mejor-1))
        silhouette_random = silhouette_score(pcas, np.ravel(random_clusters))
        compilado_silhoutte.append(silhouette_random)
        db_random = davies_bouldin_score(pcas, np.ravel(random_clusters))
        compilado_db.append(db_random)
        ch_random = calinski_harabasz_score(pcas, np.ravel(random_clusters))
        compilado_ch.append(ch_random)
        dist_dunn = pairwise_distances(pcas)
        dunn_randome = dunn(dist_dunn, np.ravel(random_clusters))
        compilado_dunn.append(dunn_randome)

    sil_random = round(mean(compilado_silhoutte),4)
    sil_random_st = str(round(stdev(compilado_silhoutte),4))
    db_random = round(mean(compilado_db),4)
    db_random_st = str(round(stdev(compilado_db),4))
    ch_random = round(mean(compilado_ch),4)
    ch_random_st = str(round(stdev(compilado_ch),4))
    dunn_random = round(mean(compilado_dunn),4)
    dunn_random_st = str(round(stdev(compilado_dunn),4))

    return sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st


### Clustering performance determination ###

def coeficientes_clustering(pcas, df_molecula_cluster_actual, cluster_mejor, molec_name,vuelta):
    from sklearn.mixture import GaussianMixture

    sil_random, sil_random_st, db_random, db_random_st, ch_random, ch_random_st, dunn_random, dunn_random_st = cluster_random(pcas, molec_name,cluster_mejor)
    silhouette_avg = round(silhouette_score(pcas, np.ravel(df_molecula_cluster_actual)),4)
    gmm = GaussianMixture(n_components=cluster_mejor, init_params='kmeans')
    gmm.fit(pcas)
    # bic_score = gmm.bic(pcas)
    db_score = round(davies_bouldin_score(pcas, np.ravel(df_molecula_cluster_actual)),4)
    ch_score = round(calinski_harabasz_score(pcas, np.ravel(df_molecula_cluster_actual)),4)
    dist_dunn = pairwise_distances(pcas)
    dunn_score = round(dunn(dist_dunn, np.ravel(df_molecula_cluster_actual)),4)
    if vuelta == 1:
        st.markdown(f'**The Silhouette score is: {silhouette_avg}**')
        st.write(f'The Silhouette Score for random cluster is: {sil_random}') 
    validation_round = [vuelta,silhouette_avg, sil_random, sil_random_st, db_score, db_random, db_random_st,ch_score, ch_random, ch_random_st,dunn_score, dunn_random, dunn_random_st]
    st.write("-"*50)
    return validation_round

#%%

### Indexes ###

def getIndexes(df, value):
    ''' Get index positions of value in dataframe as a tuple
    first the subset,then the cluster '''

    result = df.isin([value])
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            posicion = (row, col)
    return posicion

#%%

### Hierarchical Clustering ###

def clusters_con_mayor_porcentaje(lista_final_conteo, maximo_porcentaje_del_total):
    lista_cluster_para_seguir = []
    lista_cluster_padres = []
    for final_conteo_ in lista_final_conteo:
        clusters_para_seguir = []
        for index, row in final_conteo_.iterrows():
            if row['Relacion'] > maximo_porcentaje_del_total:
                clusters_para_seguir.append(index)
                lista_cluster_padres.append(index)
        lista_cluster_para_seguir.append(clusters_para_seguir)
    return lista_cluster_para_seguir, lista_cluster_padres

#%%

### Hierarchical Clustering ###

def asignar_moleculas_para_RDCPCA(lista_cluster_para_seguir, lista_cluster_moleculas, moleculas_compiladas, vuelta):
    lista_nuevas_moleculas = []
    for p, cluster_para_seguir_ in enumerate(lista_cluster_para_seguir):
        if cluster_para_seguir_ is not None:
            for cluster_ in cluster_para_seguir_:
                nuevas_moleculas = []
                for index, row in lista_cluster_moleculas[p].iterrows():
                    if row['Cluster, padre'] == cluster_:
                        nuevas_moleculas.append(index)
                        if vuelta == vueltas_maximas:
                            moleculas_compiladas[index] = row['Cluster, padre']
                lista_nuevas_moleculas.append(nuevas_moleculas)

    for cluster_moleculas_ in lista_cluster_moleculas:
        for index, row in cluster_moleculas_.iterrows():
            agregar_o_no = any([index in nuevas_moleculas_ for nuevas_moleculas_ in lista_nuevas_moleculas])
            if agregar_o_no == False:
                moleculas_compiladas[index] = row['Cluster, padre']
    return lista_nuevas_moleculas, moleculas_compiladas

#%%

### Sunburn plot of all the molecules ###

def sunburn_plot(sunburnt):
    sunburnt.insert(loc = 0, column = 'All', value = 'All')
    sunburnt = sunburnt.fillna(' ')
    sunburnt['Molecules'] = 1

    fig3 = plt1.sunburst(sunburnt, path = sunburnt.iloc[:,0:-1], values = 'Molecules')
    fig3.update_layout(title = "Sunburst Plot", title_x=0.5,
                       title_font = dict(size=25, family='Calibri', color='black'))
    fig3.update_layout(margin = dict(t=60,r=20,b=20,l=20), autosize = True)
    
    st.plotly_chart(fig3)
    return

#%%
 
### Bar plot of molecule distribution ###

def bar_plot_counts(dataframe_final_1):
    
    fig4 = plt1.bar(dataframe_final_1, x = dataframe_final_1.index.get_level_values(0), y = 'Molecules', 
                   color = dataframe_final_1.index.get_level_values(0))
    
    fig4.update_layout(legend_title="Cluster", plot_bgcolor = 'rgb(256,256,256)',
                       legend_title_font = dict(size=18, family='Calibri', color='black'),
                       legend_font = dict(size=15, family='Calibri', color='black'))
    fig4.update_xaxes(title_text='Cluster', showline=True, linecolor='black', 
                      gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                      tickfont=dict(family='Arial', size=16, color='black'),
                      title_font = dict(size=20, family='Calibri', color='black'))
    fig4.update_yaxes(title_text='Amount of molecules', showline=True, linecolor='black', 
                      gridcolor='lightgrey', zerolinecolor = 'lightgrey',
                      tickfont=dict(family='Arial', size=16, color='black'),
                      title_font = dict(size=20, family='Calibri', color='black'))
    st.plotly_chart(fig4)
    return fig4

#%%
### Settings file ###

def setting_info(vuelta,dataframe_final_1):

    today = date.today()
    fecha = today.strftime("%d/%m/%Y")
    settings = []
    settings.append(["Date clustering was performed: " , fecha])
    settings.append(["Seetings:",""])
    settings.append(["Threshold variance:", str(threshold_variance)])
    settings.append(["Random seed:", str(random_subspace_seed)])
    settings.append(["Number of subsets:", str(num_subsets)])
    settings.append(["Number of descriptors by subset:", str(num_descriptores)])
    settings.append(["Correlation coefficient:", str(coef_correlacion)])
    settings.append(["Correlation threshold:", str(limite_correlacion)])
    settings.append(["Min number of descriptors by subset:", str(min_desc_subset)])
    settings.append(["Max number of descriptors by subset:", str(max_desc_subset)])
    settings.append(["Min number of clusters by round:", str(min_n_clusters)])
    settings.append(["Max number of clusters by round:", str(max_n_clusters)])
    settings.append(["Max relation 'cluster/total':", str(maximo_porcentaje_del_total)])
    settings.append(["Max number of rounds:", str(vueltas_maximas)])
    settings.append(["PCAs:", str(num_pca)])
    settings.append(["",""])
    settings.append(["Results:",""])
    settings.append(["Total rounds :", str(vuelta)])
    settings.append(["Total clusters :", str(len(dataframe_final_1))])
    settings.append(["",""])
    settings.append(["To cite the application, please reference: ","XXXXXXXXXXX"])   
    settings_df = pd.DataFrame(settings)
    return settings_df


#%%
### Exporting files ###

def filedownload(df):
    if molecular_descriptors:
        csv = df.to_csv(index=True,header=True)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="cluster_assignations.csv">Download CSV File with the cluster assignations</a>'
    else:
        csv = df.to_csv(index=False,header=True)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="cluster_assignations.csv">Download CSV File with the cluster assignations</a>'
    return href

def filedownload1(df):

    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="cluster_distributions.csv">Download CSV File with the cluster distributions</a>'
    return href

def filedownload2(df):
    csv = df.to_csv(index=False,header=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="clustering_settings.csv">Download CSV File with your clustering settings</a>'
    return href

def filedownload3(df):
    txt = df.to_csv(sep="\t",index=True,header=True)
    b64 = base64.b64encode(txt.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/txt;base64,{b64}" download="Descriptors.txt">Download TXT File with your descriptors</a>'
    return href

def filedownload4(df):
    csv = df.to_csv(index=False,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Validations.csv">Download CSV File with the validations</a>'
    return href

#%%


### Running ###
def clustering_final_function(uploaded_file_1):
    lista_nuevas_moleculas = [1]
    vuelta = 1
    moleculas_compiladas = {}
    todos_silhouette = []
    lista_cluster_padres = ['']
    lista_cluster_moleculas = []
    lista_descriptores = []
    validation_all = []
    if molecular_descriptors:
        descriptores = calcular_descriptores(uploaded_file_1,vuelta)
    else:
        descriptores, previuos_data = calcular_descriptores(uploaded_file_1,vuelta)
    lista_descriptores.append(descriptores)

    while len(lista_nuevas_moleculas)>0 and vuelta <= vueltas_maximas:

        lista_subsets_ok = []
        lista_tablas_finales = []
        lista_final_conteo = []
        lista_subsets_seleccionados = []
        lista_total_molec_subset =[]
        sunburnt_nuevos = pd.Series(dtype="float64")

        for descriptores_ in lista_descriptores:
            descriptores_ok = descriptores_baja_variancia(descriptores_, vuelta, threshold_variance)
            subsets_ok, total_molec_subset = generar_subset(descriptores_ok, num_subsets, coef_correlacion, limite_correlacion,vuelta)
            lista_subsets_ok.append(subsets_ok)
            lista_total_molec_subset.append(total_molec_subset)
            tabla_final, subsets_seleccionados = clustering(subsets_ok, min_desc_subset, max_desc_subset, range_n_clusters, num_pca)
            lista_tablas_finales.append(tabla_final)
            lista_subsets_seleccionados.append(subsets_seleccionados)
                
        lista_cluster_moleculas = []
        for j, tabla_final_ in enumerate(lista_tablas_finales):
            try:
                silhouette_max = tabla_final_.values.max()
                todos_silhouette.append(silhouette_max)
                cluster_mejor, subset_mejor = getIndexes(tabla_final_, silhouette_max)
                subset_mejor_sil = lista_subsets_ok[j][subset_mejor]
                pcas, cluster_moleculas, final_conteo = moleculas_en_cluster_PCA_clustering(subset_mejor_sil, num_pca, cluster_mejor, subset_mejor, lista_cluster_padres[j], vuelta, descriptores)
            except ValueError:
                if vuelta == 1:
                    st.error(f'For the selected Threshold correlation filter ({limite_correlacion}) none of the subsets have between {min_desc_subset} and {max_desc_subset} descriptors in round {vuelta}')
                    st.stop()
                else:
                    for i, cluster_moleculas_ in enumerate(lista_cluster_moleculas):
                        for index, row in cluster_moleculas_.iterrows():
                            moleculas_compiladas[index] = row['Cluster, padre']
                    st.error(f'For the selected Threshold correlation filter ({limite_correlacion}) none of the subsets have between {min_desc_subset} and {max_desc_subset} descriptors in round {vuelta}')
                    st.stop()
          
            st.markdown(f"**Round: {vuelta}**")
            st.write("- Subsets with a number of descriptors between the limits: " + str(len(lista_subsets_seleccionados[j])))
            if round != 1:
                st.write("- The subset has: " + str(lista_total_molec_subset[j]) + " molecules")
            st.write("- The average number of descriptors by subset is: " + str(round(mean([x.shape[1] for x in lista_subsets_ok[j]]),2)))
            grafica_silhouette(lista_subsets_seleccionados[j],tabla_final_, num_pca, range_n_clusters, limite_correlacion)
            grafica_scatter(cluster_moleculas,subset_mejor,cluster_mejor)
            st.write(f'Maximum coefficient of silhouette obtained was obtained in the subset {subset_mejor} with {cluster_mejor} clusters\n')
    
            if vuelta == 1:
                sunburnt = pd.DataFrame(cluster_moleculas['Cluster, padre'])
            else:
                sunburnt_agregar = cluster_moleculas['Cluster, padre']
                sunburnt_nuevos = sunburnt_nuevos.append(sunburnt_agregar)       
            validation_round = coeficientes_clustering(pcas, cluster_moleculas['CLUSTER'], cluster_mejor, cluster_moleculas.index,vuelta)
            validation_all.append(validation_round)
            lista_cluster_moleculas.append(cluster_moleculas)
            lista_final_conteo.append(final_conteo)

        if vuelta != 1:
            sunburnt_nuevos = sunburnt_nuevos.to_frame()
            sunburnt_nuevos.rename(columns={0: f'Cluster, padre, V{vuelta}'},inplace = True)
            sunburnt = pd.concat([sunburnt,sunburnt_nuevos], axis = 1)
    
        lista_cluster_para_seguir, lista_cluster_padres = clusters_con_mayor_porcentaje(lista_final_conteo, maximo_porcentaje_del_total)
    
        if len(lista_cluster_para_seguir) != 0:
            lista_nuevas_moleculas, moleculas_compiladas = asignar_moleculas_para_RDCPCA(lista_cluster_para_seguir, lista_cluster_moleculas, moleculas_compiladas,vuelta)
        else:
            for i, cluster_moleculas_ in enumerate(lista_cluster_moleculas):
                for index, row in cluster_moleculas_.iterrows():
                    moleculas_compiladas[index] = row['Cluster, padre']
            break
            
        lista_descriptores = []
        for nuevas_moleculas_ in lista_nuevas_moleculas:
            descriptores_nuevas_molec = []
            for molec in nuevas_moleculas_:
                row = descriptores.loc[molec]
                descriptores_nuevas_molec.append(row)
            descriptores_nuevas_molec = pd.DataFrame(descriptores_nuevas_molec)
            lista_descriptores.append(descriptores_nuevas_molec)
    
        vuelta += 1
        
    dataframe_final = pd.DataFrame.from_dict(moleculas_compiladas, orient = 'index')
    dataframe_final.rename(columns = {0: 'CLUSTER'}, inplace = True)
    dataframe_final['key'] = dataframe_final.index
    dataframe_final['key'] = dataframe_final['key'].str.split('_').str[1].astype(int)
    dataframe_final = dataframe_final.sort_values('key', ascending=True).drop('key', axis=1)
    
    if molecular_descriptors:
        dataframe_final.index.rename("NAME", inplace = True)
    else:
        previuos_data.reset_index(drop = True, inplace = True)
        dataframe_final.reset_index(drop = True, inplace = True)
        dataframe_final = previuos_data.join(dataframe_final, how = 'right')    

    dataframe_final_1 = dataframe_final['CLUSTER'].value_counts().to_frame()
    dataframe_final_1.rename(columns = {'CLUSTER': 'Molecules'}, inplace = True)
     
    validation_final = pd.DataFrame(validation_all)
    validation_final.columns = ["Round","SIL score", "SIL random", "SD SIL random", "DB score", "DB random", "SD DB random","CH score", "CH random", "SD CH random", "Dunn score", "Dunn random", "SD Dunn random"]

    
    if len(lista_nuevas_moleculas) == 0:
        vuelta-=1
        st.success(f'The {descriptores.shape[0]} molecules were distributed in {len(dataframe_final_1)} clusters \n\nThere is no more cluster with a relationship greater than selected value: {maximo_porcentaje_del_total}\n')
    else:
        if vuelta == vueltas_maximas+1:
            vuelta-=1
            st.success(f'The {descriptores.shape[0]} molecules were distributed in {len(dataframe_final_1)} clusters \n\nThe maximum number of rounds was reached {vueltas_maximas}\n')
            
    if graficar_sunburnt == True:
        st.markdown(":point_down: **Here you can see the Sunburst Plot**", unsafe_allow_html=True)
        sunburn_plot(sunburnt)
        

    st.markdown(":point_down: **Here you can see the cluster distribution**", unsafe_allow_html=True)
    plot = bar_plot_counts(dataframe_final_1)
    
    st.markdown(":point_down: **Here you can download the cluster assignations**", unsafe_allow_html=True)
    st.markdown(filedownload(dataframe_final), unsafe_allow_html=True)

    st.markdown(":point_down: **Here you can download a table with the cluster distibutions**", unsafe_allow_html=True)
    st.markdown(filedownload1(dataframe_final_1), unsafe_allow_html=True)
    
    st.markdown(":point_down: **Here you can download a table with the validation metrics**", unsafe_allow_html=True)
    st.markdown(filedownload4(validation_final), unsafe_allow_html=True)
    
    settings_df = setting_info(vuelta,dataframe_final_1)
    st.markdown(":point_down: **Here you can download your settings**", unsafe_allow_html=True)
    st.markdown(filedownload2(settings_df), unsafe_allow_html=True)
    st.balloons()  
    return

if uploaded_file_1 is not None:
    run = st.button("RUN")
    if run == True:
        clustering_final_function(uploaded_file_1)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to run with the Example Dataset'):
        uploaded_file_1 = open("example_molecules.csv","r")
        st.markdown("**Running with the example dataset**", unsafe_allow_html=True)
        clustering_final_function(uploaded_file_1)

  
#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; ' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed with ❤️ by <a style='display: ; text-align: center' href="https://twitter.com/capigol" target="_blank">Lucas Alberca</a> and <a style='display: ; text-align: center' href="https://twitter.com/denis_prada" target="_blank">Denis Prada</a> for <a style='display:; text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
