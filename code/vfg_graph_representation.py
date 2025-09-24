import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans

import gen_hypergraphs
import tensor_operations

import tools_wavelet
import tools_regularization
from hgrc import hgrc


# V, E = gen_hypergraphs.path_hypergraph(7)
# name = "path_hypergraph7_gr"
# seed = 2

V, E = gen_hypergraphs.path_hypergraph(7)
name = "path_hypergraph7b_gr"
seed = 2
E.append([1,2])

# V, E = gen_hypergraphs.path_hypergraph(21)
# name = "path_hypergraph21_gr"
# seed = 4

# V, E = gen_hypergraphs.path_hypergraph(181)
# name = "path_hypergraph181_gr"
# seed = 0

# V, E = gen_hypergraphs.cyclic_hypergraph(24)
# name = "cycle_hypergraph24_gr"
# seed = 2

# V, E = gen_hypergraphs.squid_hypergraph(13) # or 40
# name = "squid_hypergraph13_gr"
# seed = 2

# V, E, pos = gen_hypergraphs.random_geometric_hypergraph(64, 0.20, seed=4155)
# name = "random_geometric_hypergraph_gr"
# seed = 2

# V, E = gen_hypergraphs.path_hypergraph_order4(70)
# name = "path_hypergraph70_order4_gr"
# seed = 2

# V, E, pos = gen_hypergraphs.hypergraph_H1()
# name = "hypergraph_H1"

N = len(V)

shifting = False
GFT = False
GFT_heatmap = False
iGFT = False
WGFT = False; plot_window = False

spectral_clustering = False; n_clusters = 3

translation = False; new_node = 4
modulation = False; new_module = 7

build_Rihaczek_energy_distribution = False

wavelet = False; n_filters = 7
available_filters = dict()
available_filters[0] = "spectral"
available_filters[1] = "uniform_translates"
available_filters[2] = "spectrum-adapted"

filters_name = available_filters[0]

regularization = True; gamma = 1

plot_degree = False
compute_eigenvector_centrality = False
compute_hgrc = False; gamma_hgrc = 0.5

x = np.zeros((N,))

signal = "delta"
# signal = "eigenvector"
# signal = "eigenvectors"
# signal = "exponential"
# signal = "manually"; x[0] = 1; x[4] = -1


if signal == "delta":
    center = 1

if signal == "eigenvector":
    eigenvector = 2

if name in ["path_hypergraph7_gr","path_hypergraph7b_gr","path_hypergraph21_gr",
            "cycle_hypergraph24_gr"]:
    tau = 0.5
    tau_x = 0.5

if name == "path_hypergraph181_gr":
    tau = 300
    tau_x = 1

if name == "squid_hypergraph_gr":
    tau = 2
    tau_x = 2

if name == "path_hypergraph70_order4_gr":
    tau = 10

if name == "random_geometric_hypergraph_gr":
    tau = 3

G = tensor_operations.graph_representation(V,E)

if name in ["path_hypergraph7_gr","path_hypergraph7b_gr","path_hypergraph21_gr",
            "cycle_hypergraph24_gr","squid_hypergraph13_gr"]:
    pos = nx.spring_layout(G, seed=seed)

if name in ["path_hypergraph181_gr","path_hypergraph70_order4_gr"]:
    pos = nx.spiral_layout(G)


start_time = time.process_time()

L = nx.laplacian_matrix(G)
L = L.toarray()

eigenvalues, eigenvectors = np.linalg.eig(L)

order = np.argsort(eigenvalues)


#%% Graph Spectral Clustering

if name == "random_geometric_hypergraph_gr" and signal == "eigenvectors":
    spectral_clustering = True

if spectral_clustering:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(eigenvectors[:,order[:n_clusters]])
    
    kmeans_labels = kmeans.labels_
    
    if n_clusters == 3:
        set0 = list(np.where(np.array(kmeans_labels) == 0)[0])
        set1 = list(np.where(np.array(kmeans_labels) == 1)[0])
        set2 = list(np.where(np.array(kmeans_labels) == 2)[0])
        
        sets = [set0, set1, set2]
        
        sorted_sets = sorted(sets, key=len)
        
        set0, set1, set2 = sorted_sets
        
        for i in set0:
            kmeans_labels[i] = 0
        for i in set1:
            kmeans_labels[i] = 1
        for i in set2:
            kmeans_labels[i] = 2
    
    plt.figure()
    nx.draw_networkx(G,pos,node_color=kmeans_labels,node_size=100,vmin=0,vmax=8,
                      with_labels=False,cmap="Set1")
    plt.savefig("results/"+name+"_spectral_clustering"+str(n_clusters)+".pdf", bbox_inches='tight')
    plt.show()

#%% Signal

if signal == "delta":
    x[center-1] = 1

if signal == "eigenvector":
    x = eigenvectors[:,order[eigenvector-1]]

if signal == "eigenvectors":
    if name == "path_hypergraph181_gr":
        x[:60] = eigenvectors[:60,order[10]]
        x[60:120] = eigenvectors[60:120,order[60]]
        x[120:] = eigenvectors[120:,order[30]]
    
    if name == "path_hypergraph70_order4_gr":
        x[:20] = eigenvectors[:20,order[10]]
        x[20:40] = eigenvectors[20:40,order[27]]
        x[40:] = eigenvectors[40:,order[5]]
    
    if name == "random_geometric_hypergraph_gr":
        if n_clusters == 3:
            for i in set0:
                x[i] = eigenvectors[i,order[10]]
            for i in set1:
                x[i] = eigenvectors[i,order[27]]
            for i in set2:
                x[i] = eigenvectors[i,order[5]]

if signal == "exponential":
    x_hat = np.zeros((N,))
    for l in range(N):
        x_hat[l] = np.exp(-tau_x*eigenvalues[order[l]])
    
    x = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            x[n] += x_hat[l] * eigenvectors[n,order[l]]


if signal == "delta":
    vmax = 1
    vmin = -1

if signal in ["eigenvector","eigenvectors"]:
    vmin = np.min(eigenvectors)
    vmax = np.max(eigenvectors)
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

if signal in ["exponential","manually"]:
    vmin = np.min(x)
    vmax = np.max(x)
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

cmap = "seismic"


plt.figure()
nx.draw_networkx(G,pos,node_color=x,node_size=100,vmin=vmin,vmax=vmax,with_labels=False, cmap=cmap)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
sm._A = []
plt.colorbar(sm, ax=plt.gca())
if name == "path_hypergraph181_gr":
    plt.axis("equal")
if signal == "eigenvector":
    plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+".pdf", bbox_inches='tight')
if signal == "eigenvectors":
    plt.savefig("results/"+name+"_eigenvectors.pdf", bbox_inches='tight')
if signal == "delta":
    plt.savefig("results/"+name+"_delta"+str(center)+".pdf", bbox_inches='tight')
if signal == "exponential":
    plt.savefig("results/"+name+"_exponential.pdf", bbox_inches='tight')
if signal == "manually":
    plt.savefig("results/"+name+"_manually.pdf", bbox_inches='tight')
plt.show()


#%% Shifting Operator

if shifting:
    x_shifted = L @ x
    
    vmin_shifted = np.min(x_shifted)
    vmax_shifted = np.max(x_shifted)
    
    vmax_shifted = max(np.abs(vmin_shifted),vmax_shifted)
    vmin_shifted = - vmax_shifted
    
    plt.figure()
    nx.draw_networkx(G,pos,node_color=x_shifted,node_size=100,vmin=vmin_shifted,vmax=vmax_shifted,with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_shifted,vmax=vmax_shifted))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_shifted.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_shifted.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_shifted.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_shifted.pdf", bbox_inches='tight')
    plt.show()


#%% Graph Fourier transform

if translation or modulation or wavelet:
    GFT = True

if GFT:
    x_hat = np.zeros((N,))
    
    for l in range(N):
        for n in range(N):
            x_hat[l] += x[n] * eigenvectors[n,order[l]]
    
    plt.figure()
    plt.scatter(eigenvalues[order], np.abs(x_hat))
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_GFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_GFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_GFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_GFT.pdf", bbox_inches='tight')
    plt.show()
    
    if GFT_heatmap:
        df = pd.DataFrame(np.abs(x_hat), columns=[1], index=range(1,N+1))
        
        plt.figure()
        ax = sns.heatmap(df, cmap="Blues")
        ax.invert_yaxis()
        plt.yticks(rotation=0)
        if signal == "eigenvector":
            plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_GFT_heatmap.pdf", bbox_inches='tight')
        if signal == "eigenvectors":
            plt.savefig("results/"+name+"_eigenvectors_GFT_heatmap.pdf", bbox_inches='tight')
        if signal == "delta":
            plt.savefig("results/"+name+"_delta"+str(center)+"_GFT_heatmap.pdf", bbox_inches='tight')
        if signal == "exponential":
            plt.savefig("results/"+name+"_exponential_GFT_heatmap.pdf", bbox_inches='tight') 
        plt.show()

if iGFT:
    x_rebuilt = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            x_rebuilt[n] += x_hat[l] * eigenvectors[n,order[l]]
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=x_rebuilt, node_size=100, vmin=vmin, vmax=vmax,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_iGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_iGFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_iGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_iGFT.pdf", bbox_inches='tight')
    plt.show()


#%% Translation Operator

if translation:
    Tx = np.zeros((N,))
    
    for n in range(N):
        for l in range(N):
            Tx[n] += x_hat[l] * eigenvectors[new_node,order[l]] * eigenvectors[n,order[l]]
    
    Tx *= np.sqrt(N)
    
    vmin_Tx = np.min(Tx)
    vmax_Tx = np.max(Tx)
    
    vmax_Tx = max(np.abs(vmin_Tx),vmax_Tx)
    vmin_Tx = - vmax_Tx
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=Tx, node_size=100, vmin=vmin_Tx, vmax=vmax_Tx,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_Tx,vmax=vmax_Tx))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_translation"+str(new_node)+".pdf", bbox_inches='tight')
    plt.show()


#%% Modulation Operator

if modulation:
    Mx = np.zeros((N,))
    
    for n in range(N):
        Mx[n] = x[n] * eigenvectors[n,order[new_module-1]]
    
    Mx *= np.sqrt(N)
    
    vmin_Mx = np.min(Mx)
    vmax_Mx = np.max(Mx)
    
    vmax_Mx = max(np.abs(vmin_Mx),vmax_Mx)
    vmin_Mx = - vmax_Mx
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=Mx, node_size=100, vmin=vmin_Mx, vmax=vmax_Mx,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_Mx,vmax=vmax_Mx))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    plt.show()

    Mx_hat = np.zeros((N,))
    
    for l in range(N):
        for n in range(N):
            Mx_hat[l] += Mx[n] * eigenvectors[n,order[l]]
    
    plt.figure()
    plt.scatter(eigenvalues[order], np.abs(Mx_hat))
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    plt.show()


#%% Windowed Hypergraph Fourier Transform (spectrogram)

if WGFT:
    g_hat = np.zeros((N,))
    for l in range(N):
        g_hat[l] = np.exp(-tau*eigenvalues[order[l]])
    
    if plot_window:
        plt.figure()
        plt.scatter(eigenvalues[order], g_hat)
        plt.savefig("results/"+name+"_window"+str(tau)+"_GFT.pdf", bbox_inches='tight')
        plt.show()
        
        g = np.zeros((N,))
        for n in range(N):
            for l in range(N):
                g[n] += g_hat[l] * eigenvectors[n,order[l]]
        
        vmin_g = np.min(g)
        vmax_g = np.max(g)
        
        vmax_g = max(np.abs(vmin_g),vmax_g)
        vmin_g = - vmax_g
        
        plt.figure()
        nx.draw_networkx(G, pos, node_color=g, node_size=100, vmin=vmin_g, vmax=vmax_g,
                          with_labels=False, cmap=cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_g, vmax=vmax_g))
        sm._A = []
        plt.colorbar(sm, ax=plt.gca())
        plt.savefig("results/"+name+"_window"+str(tau)+".pdf", bbox_inches='tight')
        plt.show()
    
    spectrogram = np.zeros((N,N))
    
    for k in range(N):
        for i in range(N):
            gik = np.zeros((N,))
            for n in range(N):
                for l in range(N):
                    gik[n] += g_hat[l] * eigenvectors[i,order[l]] * eigenvectors[n,order[l]]
                
                gik[n] *= eigenvectors[n,order[k]]
            
            gik *= N
            
            for n in range(N):
                spectrogram[i,k] += x[n] * gik[n]
    
    spectrogram = spectrogram.T
    
    xticklabels = range(1,N+1)
    yticklabels = range(1,N+1)
    
    if name == "random_geometric_hypergraph_gr" and spectral_clustering == True:
        spectrogram = spectrogram[:,set0 + set1 + set2]
        
        xticklabels = ["r" for i in set0]
        xticklabels = xticklabels + ["b" for i in set1]
        xticklabels = xticklabels + ["g" for i in set2]
    
    df = pd.DataFrame(spectrogram*spectrogram, columns=xticklabels, index=yticklabels)
    
    plt.figure()
    ax = sns.heatmap(df,cmap="jet")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_spectrogram.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_spectrogram.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_spectrogram.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_spectrogram.pdf", bbox_inches='tight')
    plt.show()

#%% Rihaczek Energy Distribution

if build_Rihaczek_energy_distribution:
    energy_distribution = np.zeros((N,N))
    
    for k in range(N):
        for i in range(N):
            for n in range(N):
                energy_distribution[i,k] += x[i] * x[n] * eigenvectors[i,order[k]] * eigenvectors[n,order[k]]
    
    energy_distribution = energy_distribution.T
    
    xticklabels = range(1,N+1)
    yticklabels = range(1,N+1)
    
    if name == "random_geometric_hypergraph":
        energy_distribution = energy_distribution[:,set0 + set1 + set2]
        
        xticklabels = ["r" for i in set0]
        xticklabels = xticklabels + ["b" for i in set1]
        xticklabels = xticklabels + ["g" for i in set2]
    
    df = pd.DataFrame(energy_distribution*energy_distribution, columns=xticklabels, index=yticklabels)
    
    plt.figure()
    ax = sns.heatmap(df,cmap="jet")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_energy_Rihaczek.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_energy_Rihaczek.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_energy_Rihaczek.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_energy_Rihaczek.pdf", bbox_inches='tight')
    plt.show()
    
    # Marginal properties
    
    vertex = np.zeros((N,))
    spectral = np.zeros((N,))
    
    for i in range(N):
        for k in range(N):
            vertex[i] += energy_distribution[k,i]
    
    for k in range(N):
        for i in range(N):
            spectral[k] += energy_distribution[k,i]
    
    plt.figure()
    plt.scatter(range(1,N+1), vertex)
    plt.vlines(range(1,N+1),[0 for i in range(N)],vertex)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    plt.show()
    
    plt.figure()
    plt.scatter(range(1,N+1), spectral)
    plt.vlines(range(1,N+1),[0 for i in range(N)],spectral)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    plt.show()

#%% Spectral Graph Wavelet

if wavelet:
    lmax = np.max(eigenvalues)
    
    if filters_name == "spectral":
        filters = tools_wavelet.spectral(n_filters,lmax)
    
    if filters_name == "uniform_translates":
        filters = tools_wavelet.uniform_translates(n_filters,lmax)
    
    if filters_name == "spectrum-adapted":
        approx_spectrum = tools_wavelet.spectrum_cdf_approx(N,lmax,L)
            
        filters = tools_wavelet.spectrum_adapted(n_filters, lmax, approx_spectrum)
    
    wav = np.zeros((N,n_filters))
    for n in range(N):
        for p in range(n_filters):
            for l in range(N):
                wav[n,p] += filters[p](eigenvalues[order[l]]) * x_hat[l] * eigenvectors[n,order[l]]
                
    wav = wav.T
    
    xticklabels = range(1,N+1)
    yticklabels = range(1,n_filters+1)
    
    if name == "random_geometric_hypergraph":
        wav = wav[:,set0 + set1 + set2]
        
        xticklabels = ["r" for i in set0]
        xticklabels = xticklabels + ["b" for i in set1]
        xticklabels = xticklabels + ["g" for i in set2]
    
    df = pd.DataFrame(wav*wav, columns=xticklabels, index=yticklabels)
    
    plt.figure()
    ax = sns.heatmap(df,cmap="jet")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_"+filters_name+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_"+filters_name+".pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_"+filters_name+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_"+filters_name+".pdf", bbox_inches='tight')
    plt.show()
    
    samples = np.arange(0, lmax-0.0001, 0.01)
    samples = np.append(samples, lmax)
    n_samples = len(samples)
    
    z = np.zeros((n_samples,))
    
    linewidth = 2
    
    plt.figure()
    for j in range(n_filters):
        for l in range(n_samples):
            z[l] = filters[j](samples[l])
            
        plt.plot(samples,z, linewidth=linewidth)
    
    x_spectrum = eigenvalues
        
    plt.plot(x_spectrum,np.zeros(len(x_spectrum)),'kx', mew=1.5)
    plt.savefig("results/"+name+"_kernels_"+filters_name+".pdf", bbox_inches='tight')
    plt.show()


#%% Graph Tikhonov Regularization

if regularization:
    
    gr = tools_regularization.hypergraph_regularization(L.astype('f'), gamma)
    
    x_smooth = gr.transform(x)

    vmin = np.min(x_smooth)
    vmax = np.max(x_smooth)
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

    cmap = "seismic"
    
    plt.figure()
    nx.draw_networkx(G,pos,node_color=x_smooth,node_size=100,vmin=vmin,vmax=vmax,with_labels=False,cmap=cmap)
    if name == "random_geometric_hypergraph":
        for edge in E:
            if len(edge) == 3:
                X = np.zeros([3,2])
                X[0,:] = pos[edge[0]]
                X[1,:] = pos[edge[1]]
                X[2,:] = pos[edge[2]]
                pol = plt.Polygon(X)
                plt.gca().add_patch(pol)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_gamma"+str(gamma)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_gamma"+str(gamma)+".pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_gamma"+str(gamma)+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_gamma"+str(gamma)+".pdf", bbox_inches='tight')
    if signal == "manually":
        plt.savefig("results/"+name+"_manually_gamma"+str(gamma)+".pdf", bbox_inches='tight')
    plt.show()
    
    
    if signal == "manually":
        plt.figure()
        nx.draw_networkx(G,pos,node_color=np.sign(x_smooth),node_size=100,with_labels=False,cmap=cmap)
        if name == "random_geometric_hypergraph":
            for edge in E:
                if len(edge) == 3:
                    X = np.zeros([3,2])
                    X[0,:] = pos[edge[0]]
                    X[1,:] = pos[edge[1]]
                    X[2,:] = pos[edge[2]]
                    pol = plt.Polygon(X)
                    plt.gca().add_patch(pol)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1,vmax=1))
        sm._A = []
        plt.colorbar(sm, ax=plt.gca())
        if signal == "eigenvector":
            plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "eigenvectors":
            plt.savefig("results/"+name+"_eigenvectors_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "delta":
            plt.savefig("results/"+name+"_delta"+str(center)+"_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "exponential":
            plt.savefig("results/"+name+"_exponential_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "manually":
            plt.savefig("results/"+name+"_manually_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        plt.show()


#%% Graph Regularization Centrality

if plot_degree:
    centralities = dict()
    for i in range(N):
        centralities[i] = L[i,i]
    
    vmin = np.min(list(centralities.values()))
    vmax = np.max(list(centralities.values()))
    
    reds = plt.get_cmap('Reds')
    
    newcolors = reds(np.linspace(0.25, 1.0, 256))
    cmap = ListedColormap(newcolors)
    
    plt.figure()
    nx.draw_networkx(G, pos, with_labels=False, node_size=100,
                     node_color=list(centralities.values()), cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    plt.savefig("results/"+name+"_degree.pdf", bbox_inches='tight')
    plt.show()
    
    sorted_centrality = {}
    sorted_keys = sorted(centralities, key=centralities.get)
    
    for w in sorted_keys:
        sorted_centrality[w] = centralities[w]
    
    sorted_nodes = []
    for node in sorted_centrality.keys():
        sorted_nodes.append(node)
    
    height = list(sorted_centrality.values())
    
    plt.figure()
    if name == "path_hypergraph181" or name == "random_geometric_hypergraph":
        plt.xticks([])
    else:
        plt.xticks(range(len(G)), sorted_nodes)
    plt.plot(range(len(G)), height, 'k', lw=1.2, zorder=1)
    plt.scatter(range(len(G)), height, s=100, c=height, cmap=cmap, zorder=2)
    plt.savefig('results/'+name+'_degree_signature.pdf', bbox_inches='tight')
    plt.show()

if compute_eigenvector_centrality:
    centralities = dict()
    for i in range(N):
        centralities[i] = np.abs(eigenvectors[i,order[-1]])
    
    vmin = np.min(list(centralities.values()))
    vmax = np.max(list(centralities.values()))
    
    reds = plt.get_cmap('Reds')
    
    newcolors = reds(np.linspace(0.25, 1.0, 256))
    cmap = ListedColormap(newcolors)
    
    plt.figure()
    nx.draw_networkx(G, pos, with_labels=False, node_size=100,
                     node_color=list(centralities.values()), cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    plt.savefig("results/"+name+"_eigenvector_centrality.pdf", bbox_inches='tight')
    plt.show()
    
    sorted_centrality = {}
    sorted_keys = sorted(centralities, key=centralities.get)
    
    for w in sorted_keys:
        sorted_centrality[w] = centralities[w]
    
    sorted_nodes = []
    for node in sorted_centrality.keys():
        sorted_nodes.append(node)
    
    height = list(sorted_centrality.values())
    
    plt.figure()
    if name == "path_hypergraph181" or name == "random_geometric_hypergraph":
        plt.xticks([])
    else:
        plt.xticks(range(len(G)), sorted_nodes)
    plt.plot(range(len(G)), height, 'k', lw=1.2, zorder=1)
    plt.scatter(range(len(G)), height, s=100, c=height, cmap=cmap, zorder=2)
    plt.savefig('results/'+name+'_eigenvector_centrality_signature.pdf', bbox_inches='tight')
    plt.show()
    

if compute_hgrc:
    
    centralities = hgrc(L.astype('f'), gamma_hgrc)
    
    vmin = np.min(list(centralities.values()))
    vmax = np.max(list(centralities.values()))
    
    reds = plt.get_cmap('Reds')
    
    newcolors = reds(np.linspace(0.25, 1.0, 256))
    cmap = ListedColormap(newcolors)
    
    plt.figure()
    nx.draw_networkx(G, pos, with_labels=False, node_size=100,
                     node_color=list(centralities.values()), cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    plt.savefig("results/"+name+"_hgrc.pdf", bbox_inches='tight')
    plt.show()
    
    sorted_centrality = {}
    sorted_keys = sorted(centralities, key=centralities.get)
    
    for w in sorted_keys:
        sorted_centrality[w] = centralities[w]
    
    sorted_nodes = []
    for node in sorted_centrality.keys():
        sorted_nodes.append(node)
    
    height = list(sorted_centrality.values())
    
    plt.figure()
    if name == "path_hypergraph181" or name == "random_geometric_hypergraph":
        plt.xticks([])
    else:
        plt.xticks(range(len(G)), sorted_nodes)
    plt.plot(range(len(G)), height, 'k', lw=1.2, zorder=1)
    plt.scatter(range(len(G)), height, s=100, c=height, cmap=cmap, zorder=2)
    plt.savefig('results/'+name+'_hgrc_signature.pdf', bbox_inches='tight')
    plt.show()

elapsed_time = time.process_time() - start_time
print("Elapsed time:",elapsed_time)











