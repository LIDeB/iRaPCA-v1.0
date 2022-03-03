# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:26:28 2021

@author: LIDeB UNLP
"""

# iRaPCA WebApp


#%%
# Needed packages
import pandas as pd
import sys
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
import plotly
import plotly.graph_objects as go
import plotly.express as plt1
import random
from datetime import date
from molvs import Standardizer
import warnings


#%%

# Name of the dataset
dataset = "F7"

# Folder with files
directory = str(Path(r"C:\Users\adm\Desktop\Pruebas\Dataset"))

# If the molecular descriptors were previously calculated 
available_molecular_descriptors = False

if available_molecular_descriptors:
    # molecular descriptors in a TXT file. Your file should have a column called 'NAME'
    descriptors_dataset = "F7_descriptors_Mordred.txt" 
    uploaded_file_1 = pd.read_csv(directory + "\\" + descriptors_dataset, sep='\t', delimiter=None, header='infer', names=None)
else:
    # CSV file with one SMILES per line
    SMILES_dataset = "Smiles_F7.csv"
    uploaded_file_1 = pd.read_csv(directory + "\\" + SMILES_dataset)
    
    # If the SMILES should be standardized or not
    smiles_standardization = True      
    # Ignore error in SMILES
    ignore_error = True

# Threshold variance
threshold_variance = 0.05

# If you want a random seed
random_subspace_seed= False

# Nº of subsets
num_subsets = 100 
# Nº descriptors per subset
num_descriptors = 200 

# Correlation coefficient
coef_correlacion = "pearson"  # it can be "pearson", "kendall" y "spearman"
# Threshold correlation
threshold_correlation = 0.4          

# Min Nº of descriptors for subset
min_desc_subset = 4
# Max Nº of descriptors for subset
max_desc_subset = 25

# Min Nº of clusters by round
min_n_clusters = 2 
# Max Nº of clusters by round
max_n_clusters = 25 
range_n_clusters = list(range(2,25,1)) # range(start, stop, step)

# Max relation "cluster/total"
max_ratio_cluster_total = 0.30
# Max Nº of rounds
max_round = 5

# Nº of Principal Component Analysis 
num_pca = 2

# Plots 
plot_silhouette = False
plot_scatter = False
plot_sunburnt = True
plot_bar = True

# plot configuration
plot_format = "svg"     # it can be png, svg, jpeg, webp
plot_height = 800       # plot height in pixeles
plot_width = 800        # plot width in pixeles
plot_scale = 2          # Multiply title/legend/axis/canvas sizes by this factor

config = {'toImageButtonOptions': {'format': plot_format, 
    'height': plot_height,'width': plot_width,'scale': plot_scale}}
    
#%%

### Reading/calculating molecular descriptors ###

def calcular_descriptores(uploaded_file_1,available_molecular_descriptors):
    
    if available_molecular_descriptors:
        descriptores = uploaded_file_1
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
        print("**Step 1: Uploading molecular descriptor**")        
        return descriptores
    else:
        data1x = pd.DataFrame()
        suppl = []
        print("**Step 1: Standarization and descriptor calculation**")
        df_initial = uploaded_file_1
        if "SMILES" in df_initial:
            list_of_smiles = df_initial["SMILES"]
        else:
            list_of_smiles = df_initial.iloc[:, 0]
        s = Standardizer()

        problematic_smiles = []
        rows_to_retain = []

        for i,molecule in enumerate(list_of_smiles, start = 1):
            try:
                mol = Chem.MolFromSmiles(molecule)
                if smiles_standardization == True:
                    estandarizada = s.super_parent(mol) 
                    suppl.append(estandarizada)
                    rows_to_retain.append(i -1)
                else:
                    suppl.append(mol)
                    
            except:
                problematic_smiles.append(i)

        if ignore_error == False and len(problematic_smiles) > 0:
            print("Oh no! There is a problem with descriptor calculation of some SMILES.")
            print(f"Please check your SMILES number: {str(problematic_smiles)}")
            sys.exit()

        else:
            if len(problematic_smiles) > 0:
                print(f"Lines {str(problematic_smiles)} have problematic (or empty) SMILES. We have omitted them.")
            else:
                print("All SMILES have been submitted correctly")
            
        calc = Calculator(descriptors, ignore_3D=True) 

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
                            print("Calculating descriptors " + str(i+1) +"/" + str(len(suppl)))   
                            if smiles_standardization == False:
                                rows_to_retain.append(i)
                        except:
                            print("Oh no! There is a problem with descriptor calculation of some SMILES.")
                            print("Please check your SMILES number: " + str(i+1))
                            sys.exit()
                    else:
                        pass
                    
        previuos_data = df_initial.iloc[rows_to_retain]
                
        print("Descriptor calculation have FINISHED")
        data1x = data1x.T
        descriptores = data1x.set_index('NAME',inplace=False).copy()
        descriptores = descriptores.reindex(sorted(descriptores.columns), axis=1)   
        descriptores.replace([np.inf, -np.inf], np.nan, inplace=True)
        descriptores = descriptores.apply(pd.to_numeric, errors = 'coerce') 
        descriptores = descriptores.dropna(axis=0,how="all")
        descriptores = descriptores.dropna(axis=1)
                
        return descriptores,previuos_data, rows_to_retain 
    
        

#%%

### Removing low variance descriptors ###

def descriptores_baja_variancia(descriptores, vuelta, threshold_variance: float):
    selector = VarianceThreshold(threshold_variance)       
    selector.fit(descriptores)                              
    descriptores_ok = descriptores[descriptores.columns[selector.get_support(indices=True)]]
    if vuelta == 1:
        print(f'{str(descriptores_ok.shape[1])} descriptors have passed the variance threshold')
        print("-"*50)
        print("**Step 2: Clustering**")
          
    return descriptores_ok

#%%

### Subsetting ###

def generar_subset(descriptores_ok, num_subsets: int, coef_correlacion: str, threshold_correlation: float, vuelta):
    subsets_ok=[]
    i=0
    while (i < num_subsets): 
        if random_subspace_seed == True:
            subset= descriptores_ok.sample(num_descriptors,axis=1)
        else:
            subset= descriptores_ok.sample(num_descriptors,axis=1,random_state=i)  
        corr_matrix = subset.corr(coef_correlacion).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold_correlation)]
        curado=subset.drop(subset[to_drop], axis=1)
        total_molec_subset = curado.shape[0]
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
    for i, subset in enumerate(subsets_ok):
        
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

def grafica_silhouette(subsets_seleccionados,tabla_final,num_pca: int, range_n_clusters, vuelta, threshold_correlation: float):
    
    if plot_silhouette:
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

        plotly.offline.plot(fig, filename= f'{directory}\\SIL_round_{vuelta}_PCAs_{num_pca}_coef_{threshold_correlation}.html', config=config)
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

def grafica_scatter(moleculas_cluster,subset_mejor,cluster_mejor, vuelta):
    if plot_scatter:
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
        plotly.offline.plot(fig2, filename= f'{directory}\\Scatterplot_PCs_round_{vuelta}_cluster_{cluster_mejor}_subset_{subset_mejor}.html', config=config)
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
    db_score = round(davies_bouldin_score(pcas, np.ravel(df_molecula_cluster_actual)),4)
    ch_score = round(calinski_harabasz_score(pcas, np.ravel(df_molecula_cluster_actual)),4)
    dist_dunn = pairwise_distances(pcas)
    dunn_score = round(dunn(dist_dunn, np.ravel(df_molecula_cluster_actual)),4)
    if vuelta == 1:
       print(f'The Silhouette score is: {silhouette_avg}')
       print(f'The Silhouette Score for random cluster is: {sil_random}') 
    validation_round = [vuelta,silhouette_avg, sil_random, sil_random_st, db_score, db_random, db_random_st,ch_score, ch_random, ch_random_st,dunn_score, dunn_random, dunn_random_st]
    print("-"*50)
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

def clusters_con_mayor_porcentaje(lista_final_conteo, max_ratio_cluster_total):
    lista_cluster_para_seguir = []
    lista_cluster_padres = []
    for final_conteo_ in lista_final_conteo:
        clusters_para_seguir = []
        for index, row in final_conteo_.iterrows():
            if row['Relacion'] > max_ratio_cluster_total:
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
                        if vuelta == max_round:
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
    if plot_sunburnt: 
        warnings.simplefilter(action='ignore', category=FutureWarning)
        sunburnt.insert(loc = 0, column = 'All', value = 'All')
        sunburnt = sunburnt.fillna(' ')
        sunburnt['Molecules'] = 1
        
        fig3 = plt1.sunburst(sunburnt, path = sunburnt.iloc[:,0:-1], values = 'Molecules')
        fig3.update_layout(title = "Sunburst Plot", title_x=0.5,
                           title_font = dict(size=25, family='Calibri', color='black'))
        fig3.update_layout(margin = dict(t=60,r=20,b=20,l=20), autosize = True)
        
        plotly.offline.plot(fig3, filename= f'{directory}\\Sunburst_{dataset}.html', config=config)
    return

#%%
 
### Bar plot of molecule distribution ###

def bar_plot_counts(dataframe_final_1):
    if plot_bar: 
        df_bar = dataframe_final_1.copy()
        df_bar['name_cluster'] = dataframe_final_1.index.astype(str)
        fig4 = plt1.bar(df_bar, x = 'name_cluster', y = 'Molecules', color = 'name_cluster')0))
        
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
        plotly.offline.plot(fig4, filename= f'{directory}\\Barplot_{dataset}.html', config=config)
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
    settings.append(["Number of descriptors by subset:", str(num_descriptors)])
    settings.append(["Correlation coefficient:", str(coef_correlacion)])
    settings.append(["Correlation threshold:", str(threshold_correlation)])
    settings.append(["Min number of descriptors by subset:", str(min_desc_subset)])
    settings.append(["Max number of descriptors by subset:", str(max_desc_subset)])
    settings.append(["Min number of clusters by round:", str(min_n_clusters)])
    settings.append(["Max number of clusters by round:", str(max_n_clusters)])
    settings.append(["Max relation 'cluster/total':", str(max_ratio_cluster_total)])
    settings.append(["Max number of rounds:", str(max_round)])
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


### Running ###

lista_nuevas_moleculas = [1]
vuelta = 1
moleculas_compiladas = {}
todos_silhouette = []
lista_cluster_padres = ['']
lista_cluster_moleculas = []
lista_descriptores = []
validation_all = []

if available_molecular_descriptors:
    descriptores = calcular_descriptores(uploaded_file_1,available_molecular_descriptors)
else:
    descriptores, previuos_data, rows_to_retain = calcular_descriptores(uploaded_file_1,available_molecular_descriptors)
    
lista_descriptores.append(descriptores)

while len(lista_nuevas_moleculas)>0 and vuelta <= max_round:

    lista_subsets_ok = []
    lista_tablas_finales = []
    lista_final_conteo = []
    lista_subsets_seleccionados = []
    lista_total_molec_subset =[]
    sunburnt_nuevos = pd.Series(dtype = 'float64')

    for descriptores_ in lista_descriptores:
        descriptores_ok = descriptores_baja_variancia(descriptores_, vuelta, threshold_variance)
        subsets_ok, total_molec_subset = generar_subset(descriptores_ok, num_subsets, coef_correlacion, threshold_correlation,vuelta)
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
                print(f'For the selected Threshold correlation filter ({threshold_correlation}) none of the subsets have between {min_desc_subset} and {max_desc_subset} descriptors in round {vuelta}')
                sys.exit()
            else:
                for i, cluster_moleculas_ in enumerate(lista_cluster_moleculas):
                    for index, row in cluster_moleculas_.iterrows():
                        moleculas_compiladas[index] = row['Cluster, padre']
                print(f'For the selected Threshold correlation filter ({threshold_correlation}) none of the subsets have between {min_desc_subset} and {max_desc_subset} descriptors in round {vuelta}')
                sys.exit()
      
        print(f"**Round: {vuelta}**")
        print("- Subsets with a number of descriptors between the limits: " + str(len(lista_subsets_seleccionados[j])))
        if vuelta != 1:
            print("- The subset has: " + str(lista_total_molec_subset[j]) + " molecules")
        print("- The average number of descriptors by subset is: " + str(round(mean([x.shape[1] for x in lista_subsets_ok[j]]),2)))
        grafica_silhouette(lista_subsets_seleccionados[j],tabla_final_, num_pca, range_n_clusters,vuelta, threshold_correlation)
        grafica_scatter(cluster_moleculas,subset_mejor,cluster_mejor, vuelta)
        print(f'Maximum coefficient of silhouette obtained was obtained in the subset {subset_mejor} with {cluster_mejor} clusters\n')

        if vuelta == 1:
            sunburnt = pd.DataFrame(cluster_moleculas['Cluster, padre'])
        else:
            sunburnt_agregar = cluster_moleculas['Cluster, padre']
            sunburnt_nuevos = pd.concat([sunburnt_nuevos, sunburnt_agregar], axis = 0)
        validation_round = coeficientes_clustering(pcas, cluster_moleculas['CLUSTER'], cluster_mejor, cluster_moleculas.index,vuelta)
        validation_all.append(validation_round)
        lista_cluster_moleculas.append(cluster_moleculas)
        lista_final_conteo.append(final_conteo)

    if vuelta != 1:
        sunburnt_nuevos = sunburnt_nuevos.to_frame()
        sunburnt_nuevos.rename(columns={0: f'Cluster, padre, V{vuelta}'},inplace = True)
        sunburnt = pd.concat([sunburnt,sunburnt_nuevos], axis = 1)

    lista_cluster_para_seguir, lista_cluster_padres = clusters_con_mayor_porcentaje(lista_final_conteo, max_ratio_cluster_total)

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

if available_molecular_descriptors:
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
    print(f'The {descriptores.shape[0]} molecules were distributed in {len(dataframe_final_1)} clusters \n\nThere is no more cluster with a relationship greater than selected value: {max_ratio_cluster_total}\n')
else:
    if vuelta == max_round+1:
        vuelta-=1
        print(f'The {descriptores.shape[0]} molecules were distributed in {len(dataframe_final_1)} clusters \n\nThe maximum number of rounds was reached {max_round}\n')
        
sunburn_plot(sunburnt)
    
bar_plot_counts(dataframe_final_1)

if available_molecular_descriptors:
    dataframe_final.to_csv(f'{directory}\\Cluster_assignation_{dataset}.csv', index=True, header = True)
else:
    dataframe_final.to_csv(f'{directory}\\Cluster_assignation_{dataset}.csv', index=False, header = True)

dataframe_final_1.to_csv(f'{directory}\\Cluster_distributions_{dataset}.csv', index=True,header=True)

validation_final.to_csv(f'{directory}\\Validation_indexes_{dataset}.csv',index=False,header=True)

settings_df = setting_info(vuelta,dataframe_final_1)
settings_df.to_csv(f'{directory}\\Clustering_setting_{dataset}.csv',index=False,header=False)
  
 

