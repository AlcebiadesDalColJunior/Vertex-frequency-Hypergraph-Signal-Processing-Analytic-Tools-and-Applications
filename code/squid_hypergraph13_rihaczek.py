import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.fft import fft, fft2, fftn

from sklearn.cluster import KMeans

import gen_hypergraphs
import tensor_operations

import tools_wavelet
import tools_regularization
from hgrc import hgrc


# V, E = gen_hypergraphs.path_hypergraph(7)
# name = "path_hypergraph7"
# seed = 2

# V, E = gen_hypergraphs.path_hypergraph(7)
# name = "path_hypergraph7b"
# seed = 2
# E.append([1,2])

# V, E = gen_hypergraphs.path_hypergraph(21)
# name = "path_hypergraph21"
# seed = 4

# V, E = gen_hypergraphs.path_hypergraph(181)
# name = "path_hypergraph181"
# seed = 0

# V, E = gen_hypergraphs.cyclic_hypergraph(24)
# name = "cycle_hypergraph24"
# seed = 2

V, E = gen_hypergraphs.squid_hypergraph(13)
name = "squid_hypergraph13"
seed = 2

# V, E, pos = gen_hypergraphs.random_geometric_hypergraph(64, 0.20, seed=4155)
# name = "random_geometric_hypergraph"
# seed = 2

# V, E = gen_hypergraphs.path_hypergraph_order4(70)
# name = "path_hypergraph70_order4"
# seed = 2

# V, E, pos = gen_hypergraphs.hypergraph_H1()
# name = "hypergraph_H1"


N = len(V)

graph_representation = False

shifting = False
HGFT = False
HGFT_heatmap = False
iHGFT = False
WHGFT = True; plot_window = False

frontal_slice = 0
plot_frontal_slice = False
plot_frontal_slices = False

spectral_clustering = False; n_clusters = 3

translation = False; new_node = 4
modulation = False; new_module = 7

build_Rihaczek_energy_distribution = True

wavelet = False; n_filters = 7
available_filters = dict()
available_filters[0] = "spectral"
available_filters[1] = "uniform_translates"
available_filters[2] = "spectrum_adapted"

filters_name = available_filters[0]

regularization = False; gamma = 1

plot_degree = False
compute_eigenvector_centrality = False
compute_hgrc = False; gamma_hgrc = 1

t_shifting = False
t_HGFT = False

x = np.zeros((N,))

# signal = "delta"
signal = "eigenvector"
# signal = "eigenvectors"
# signal = "exponential"
# signal = "manually"

if signal == "delta":
    center = 52

if signal == "eigenvector":
    eigenvector = 1

if signal == "manually":
    x[0] = 1; x[4] = -1
    #x[2] = 1; x[4] = -1

if name in ["path_hypergraph7","path_hypergraph7b","path_hypergraph21",
            "cycle_hypergraph24"]:
    tau = 1
    tau_x = 1

if name == "path_hypergraph181":
    tau = 300
    tau_x = 1

if name == "squid_hypergraph13":
    tau = 2
    tau_x = 2

if name == "path_hypergraph70_order4":
    tau = 50

if name == "random_geometric_hypergraph":
    tau = 3

if name == "hypergraph_H1":
    tau = 1

if name in ["cameraman_5_5_neighborhood_hypergraph","cameraman_5_5_ianh"]:
    tau = 1

G = tensor_operations.graph_representation(V,E)

if name in ["path_hypergraph7","path_hypergraph7b","path_hypergraph21",
            "cycle_hypergraph24","squid_hypergraph13"]:
    pos = nx.spring_layout(G, seed=seed)

if name in ["path_hypergraph181","path_hypergraph70_order4"]:
    pos = nx.spiral_layout(G)


start_time = time.process_time()

if graph_representation:
    plt.figure()
    nx.draw_networkx(G, pos)
    if name == "path_hypergraph181":
        plt.axis("equal")
    plt.savefig("results/"+name+"_graph_representation.pdf", bbox_inches='tight')
    plt.show()

A = tensor_operations.adjacency_tensor(V,E)

D = tensor_operations.degree_tensor(V,E,A)

L = D - A

M = len(A.shape)

if M == 3:
    L_sym = tensor_operations.sym(L)
if M == 4:
    L_sym = tensor_operations.sym4(L)
if M == 5:
    L_sym = tensor_operations.sym5(L)

if shifting or wavelet or regularization or compute_hgrc:
    if M == 3:
        L_sym_hat = np.real(fft(L_sym))
    if M == 4:
        L_sym_hat = np.real(fft2(L_sym))
    if M == 5:
        L_sym_hat = np.real(fftn(L_sym, axes=(-3,-2,-1)))

eigenvalues_hat, eigenvectors_hat = tensor_operations.t_eigendecomposition(L_sym)

eigenvalues_hat = np.real(eigenvalues_hat)
eigenvectors_hat = np.real(eigenvectors_hat)

if M == 3:
    eigenvalues_vec = np.diag(eigenvalues_hat[:,:,frontal_slice])
if M == 4:
    eigenvalues_vec = np.diag(eigenvalues_hat[:,:,frontal_slice,frontal_slice])
if M == 5:
    eigenvalues_vec = np.diag(eigenvalues_hat[:,:,frontal_slice,frontal_slice,frontal_slice])

order = np.argsort(eigenvalues_vec)


#%% Hypergraph Spectral Clustering

if name == "random_geometric_hypergraph" and signal == "eigenvectors":
    spectral_clustering = True

if spectral_clustering:
    if M == 3:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(eigenvectors_hat[:,order[:n_clusters],frontal_slice])
    
    if M == 4:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(eigenvectors_hat[:,order[:n_clusters],frontal_slice,frontal_slice])
    
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
    if name == "random_geometric_hypergraph":
        for edge in E:
            if len(edge) == 3:
                X = np.zeros([3,2])
                X[0,:] = pos[edge[0]]
                X[1,:] = pos[edge[1]]
                X[2,:] = pos[edge[2]]
                pol = plt.Polygon(X)
                plt.gca().add_patch(pol)
    plt.savefig("results/"+name+"_spectral_clustering"+str(n_clusters)+".pdf", bbox_inches='tight')
    plt.show()

#%% Signal

if signal == "delta":
    x[center] = 1

if signal == "eigenvector":
    if M == 3:
        x = eigenvectors_hat[:,order[eigenvector],frontal_slice]
    if M == 4:
        x = eigenvectors_hat[:,order[eigenvector],frontal_slice,frontal_slice]
    if M == 5:
        x = eigenvectors_hat[:,order[eigenvector],frontal_slice,frontal_slice,frontal_slice]

if signal == "eigenvectors":
    if name == "path_hypergraph181":
        x[:60] = eigenvectors_hat[:60,order[10],frontal_slice]
        x[60:120] = eigenvectors_hat[60:120,order[60],frontal_slice]
        x[120:] = eigenvectors_hat[120:,order[30],frontal_slice]
    
    if name == "path_hypergraph70_order4":
        x[:20] = eigenvectors_hat[:20,order[10],frontal_slice,frontal_slice]
        x[20:40] = eigenvectors_hat[20:40,order[27],frontal_slice,frontal_slice]
        x[40:] = eigenvectors_hat[40:,order[5],frontal_slice,frontal_slice]
    
    if name == "random_geometric_hypergraph":
        if n_clusters == 3:
            for i in set0:
                x[i] = eigenvectors_hat[i,order[10],frontal_slice]
            for i in set1:
                x[i] = eigenvectors_hat[i,order[27],frontal_slice]
            for i in set2:
                x[i] = eigenvectors_hat[i,order[5],frontal_slice]

if signal == "exponential":
    x_hat = np.zeros((N,))
    for l in range(N):
        if M == 3:
            x_hat[l] = np.exp(-tau_x * eigenvalues_hat[order[l],order[l],frontal_slice])
        if M == 4:
            x_hat[l] = np.exp(-tau_x * eigenvalues_hat[order[l],order[l],frontal_slice,frontal_slice])
    
    x = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            if M == 3:
                x[n] += x_hat[l] * eigenvectors_hat[n,order[l],frontal_slice]
            if M == 4:
                x[n] += x_hat[l] * eigenvectors_hat[n,order[l],frontal_slice,frontal_slice]

if signal == "delta":
    vmax = 1
    vmin = -1

if signal in ["eigenvector","eigenvectors"]:
    if M == 3:
        vmin = np.min(eigenvectors_hat[:,:,frontal_slice])
        vmax = np.max(eigenvectors_hat[:,:,frontal_slice])
    if M == 4:
        vmin = np.min(eigenvectors_hat[:,:,frontal_slice,frontal_slice])
        vmax = np.max(eigenvectors_hat[:,:,frontal_slice,frontal_slice])
    if M == 5:
        vmin = np.min(eigenvectors_hat[:,:,frontal_slice,frontal_slice,frontal_slice])
        vmax = np.max(eigenvectors_hat[:,:,frontal_slice,frontal_slice,frontal_slice])
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

if signal in ["exponential","manually"]:
    vmin = np.min(x)
    vmax = np.max(x)
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax
    
cmap = "seismic"


plt.figure()
nx.draw_networkx(G,pos,node_color=x,node_size=100,vmin=vmin,vmax=vmax,with_labels=False,cmap=cmap)
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
if name == "path_hypergraph181":
    plt.axis("equal")
if signal == "delta":
    plt.savefig("results/"+name+"_delta"+str(center+1)+".pdf", bbox_inches='tight')
if signal == "eigenvector":
    plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+".pdf", bbox_inches='tight')
if signal == "eigenvectors":
    plt.savefig("results/"+name+"_eigenvectors.pdf", bbox_inches='tight')
if signal == "exponential":
    plt.savefig("results/"+name+"_exponential.pdf", bbox_inches='tight')
if signal == "manually":
    plt.savefig("results/"+name+"_manually.pdf", bbox_inches='tight')
plt.show()


#%% Shifting Operator

if shifting:
    if M == 3:
        x_shifted = L_sym_hat[:,:,frontal_slice] @ x
    if M == 4:
        x_shifted = L_sym_hat[:,:,frontal_slice,frontal_slice] @ x
    
    plt.figure()
    nx.draw_networkx(G,pos,node_color=x_shifted,node_size=100,vmin=vmin,vmax=vmax,with_labels=False,cmap=cmap)
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
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_shifted.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_shifted.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_shifted.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_shifted.pdf", bbox_inches='tight')
    plt.show()


#%% Hypergraph Fourier Transform

if translation or modulation or wavelet:
    HGFT = True

if HGFT:
    x_hat = np.zeros((N,))

    for l in range(N):
        for n in range(N):
            if M == 3:
                x_hat[l] += x[n] * eigenvectors_hat[n,order[l],frontal_slice]
            if M == 4:
                x_hat[l] += x[n] * eigenvectors_hat[n,order[l],frontal_slice,frontal_slice]

    plt.figure()
    plt.scatter(eigenvalues_vec[order], np.abs(x_hat))
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_HGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_HGFT.pdf", bbox_inches='tight')     
    plt.show()
    
    if HGFT_heatmap:
        df = pd.DataFrame(np.abs(x_hat), columns=[1], index=range(1,N+1))
        
        plt.figure()
        ax = sns.heatmap(df, cmap="Blues")
        ax.invert_yaxis()
        plt.yticks(rotation=0)
        if signal == "delta":
            plt.savefig("results/"+name+"_delta"+str(center+1)+"_HGFT_heatmap.pdf", bbox_inches='tight')
        if signal == "eigenvector":
            plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_HGFT_heatmap.pdf", bbox_inches='tight')
        if signal == "eigenvectors":
            plt.savefig("results/"+name+"_eigenvectors_HGFT_heatmap.pdf", bbox_inches='tight')
        if signal == "exponential":
            plt.savefig("results/"+name+"_exponential_HGFT_heatmap.pdf", bbox_inches='tight') 
        plt.show()

if iHGFT:
    x_rebuilt = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            if M == 3:
                x_rebuilt[n] += x_hat[l] * eigenvectors_hat[n,order[l],frontal_slice]
            if M == 4:
                x_rebuilt[n] += x_hat[l] * eigenvectors_hat[n,order[l],frontal_slice,frontal_slice]
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=x_rebuilt, node_size=100, vmin=vmin, vmax=vmax,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_iHGFT.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_iHGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_iHGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_iHGFT.pdf", bbox_inches='tight')
    plt.show()
    

#%% Translation Operator

if translation:
    Tx = np.zeros((N,))
    
    for n in range(N):
        for l in range(N):
            Tx[n] += x_hat[l] * eigenvectors_hat[new_node-1,order[l],frontal_slice] * eigenvectors_hat[n,order[l],frontal_slice]
    
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
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_translation"+str(new_node)+".pdf", bbox_inches='tight')
    plt.show()
    
    Tx_hat = np.zeros((N,))
    
    for l in range(N):
        for n in range(N):
            Tx_hat[l] += Tx[n] * eigenvectors_hat[n,order[l],frontal_slice]
    
    plt.figure()
    plt.scatter(eigenvalues_vec[order], np.abs(Tx_hat))
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    plt.show()


#%% Modulation operator

if modulation:
    Mx = np.zeros((N,))
    
    for n in range(N):
        Mx[n] = x[n] * eigenvectors_hat[n,order[new_module-1],frontal_slice]
    
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
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    plt.show()

    Mx_hat = np.zeros((N,))
    
    for l in range(N):
        for n in range(N):
            Mx_hat[l] += Mx[n] * eigenvectors_hat[n,order[l],frontal_slice]
    
    plt.figure()
    plt.scatter(eigenvalues_vec[order], np.abs(Mx_hat))
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    plt.show()


#%% Windowed Hypergraph Fourier Transform (spectrogram)

if WHGFT:
    g_hat = np.zeros((N,))
    for l in range(N):
        if M == 3:
            g_hat[l] = np.exp(-tau*eigenvalues_hat[order[l],order[l],frontal_slice])
        if M == 4:
            g_hat[l] = np.exp(-tau*eigenvalues_hat[order[l],order[l],frontal_slice,frontal_slice])
        if M == 5:
            g_hat[l] = np.exp(-tau*eigenvalues_hat[order[l],order[l],frontal_slice,frontal_slice,frontal_slice])
    
    if plot_window:
        plt.figure()
        plt.scatter(eigenvalues_vec[order], g_hat)
        plt.savefig("results/"+name+"_window"+str(tau)+"_HGFT.pdf", bbox_inches='tight')
        plt.show()
        
        plt.figure()
        if M == 3:
            lmax = np.max(np.diag(eigenvalues_hat[:,:,frontal_slice]))
        if M == 4:
            lmax = np.max(np.diag(eigenvalues_hat[:,:,frontal_slice,frontal_slice]))
        
        samples = np.arange(0, lmax-0.0001, 0.01)
        if name == "path_hypergraph181":
            samples = np.arange(0, lmax-0.000001, 0.0001)
        samples = np.append(samples, lmax)
        
        z = np.exp(-tau*samples)
        
        linewidth = 2
        
        if M == 3:
            x_spectrum = np.diag(eigenvalues_hat[:,:,frontal_slice])
        if M == 4:
            x_spectrum = np.diag(eigenvalues_hat[:,:,frontal_slice,frontal_slice])
        plt.plot(x_spectrum,np.zeros(len(x_spectrum)),'kx', mew=1.5)
        plt.plot(samples, z, linewidth=linewidth)
        plt.savefig("results/"+name+"_window"+str(tau)+"_HGFT_filled.pdf", bbox_inches='tight')
        plt.show()
    
        g = np.zeros((N,))
        for n in range(N):
            for l in range(N):
                if M == 3:
                    g[n] += g_hat[l] * eigenvectors_hat[n,order[l],frontal_slice]
                if M == 4:
                    g[n] += g_hat[l] * eigenvectors_hat[n,order[l],frontal_slice,frontal_slice]
        
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
                    if M == 3:
                        gik[n] += g_hat[l] * eigenvectors_hat[i,order[l],frontal_slice] * eigenvectors_hat[n,order[l],frontal_slice]
                    if M == 4:
                        gik[n] += g_hat[l] * eigenvectors_hat[i,order[l],frontal_slice,frontal_slice] * eigenvectors_hat[n,order[l],frontal_slice,frontal_slice]
                    if M == 5:
                        gik[n] += g_hat[l] * eigenvectors_hat[i,order[l],frontal_slice,frontal_slice,frontal_slice] * eigenvectors_hat[n,order[l],frontal_slice,frontal_slice,frontal_slice]
                if M == 3:
                    gik[n] *= eigenvectors_hat[n,order[k],frontal_slice]
                if M == 4:
                    gik[n] *= eigenvectors_hat[n,order[k],frontal_slice,frontal_slice]
                if M == 5:
                    gik[n] *= eigenvectors_hat[n,order[k],frontal_slice,frontal_slice,frontal_slice]
            
            gik *= N
            
            for n in range(N):
                spectrogram[i,k] += x[n] * gik[n]
    
    spectrogram = spectrogram.T    
    
    xticklabels = range(1,N+1)
    yticklabels = range(1,N+1)
    
    if name == "random_geometric_hypergraph" and spectral_clustering == True:
        spectrogram = spectrogram[:,set0 + set1 + set2]
        
        xticklabels = ["r" for i in set0]
        xticklabels = xticklabels + ["b" for i in set1]
        xticklabels = xticklabels + ["g" for i in set2]
    
    df = pd.DataFrame(spectrogram*spectrogram, columns=xticklabels, index=yticklabels)
    
    plt.figure()
    ax = sns.heatmap(df,cmap="jet")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if name == "path_hypergraph21":
        plt.xticks(rotation=90)
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_spectrogram.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_spectrogram.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_spectrogram.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_spectrogram.pdf", bbox_inches='tight')
    plt.show()


#%% Rihaczek Energy Distribution

if build_Rihaczek_energy_distribution:
    energy_distribution = np.zeros((N,N))
    
    for k in range(N):
        for i in range(N):
            for n in range(N):
                if M == 3:
                    energy_distribution[i,k] += x[i] * x[n] * eigenvectors_hat[i,order[k],frontal_slice] * eigenvectors_hat[n,order[k],frontal_slice]
                if M == 4:
                    energy_distribution[i,k] += x[i] * x[n] * eigenvectors_hat[i,order[k],frontal_slice,frontal_slice] * eigenvectors_hat[n,order[k],frontal_slice,frontal_slice]
    
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
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_energy_Rihaczek.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_energy_Rihaczek.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_energy_Rihaczek.pdf", bbox_inches='tight')
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
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_energy_Rihaczek_marginal_vertex.pdf", bbox_inches='tight')
    plt.show()
    
    plt.figure()
    plt.scatter(range(1,N+1), spectral)
    plt.vlines(range(1,N+1),[0 for i in range(N)],spectral)
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_energy_Rihaczek_marginal_spectral.pdf", bbox_inches='tight')
    plt.show()


#%% Frontal Slices

if plot_frontal_slice or plot_frontal_slices:
    xticklabels = range(1,N+1)
    yticklabels = range(1,N+1)

if plot_frontal_slice:
    if M == 3:
        df = pd.DataFrame(eigenvectors_hat[:,order,frontal_slice], columns=xticklabels, index=yticklabels)
    if M == 4:
        df = pd.DataFrame(eigenvectors_hat[:,order,frontal_slice,frontal_slice], columns=xticklabels, index=yticklabels)
        
    plt.figure()
    ax = sns.heatmap(df, cmap="Reds")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    plt.savefig("results/"+name+"_first_frontal_slice.pdf", bbox_inches='tight')
    plt.show()

if plot_frontal_slices:
    for i in range(N+1):
        if M == 3:
            eigenvalues_vec_i = np.diag(eigenvalues_hat[:,:,i])
        if M == 4:
            eigenvalues_vec_i = np.diag(eigenvalues_hat[:,:,i,i])
        if M == 5:
            eigenvalues_vec_i = np.diag(eigenvalues_hat[:,:,i,i,i])

        order_i = np.argsort(eigenvalues_vec_i)
        
        if M == 3:
            df = pd.DataFrame(eigenvectors_hat[:,order_i,i], columns=xticklabels, index=yticklabels)
        if M == 4:
            df = pd.DataFrame(eigenvectors_hat[:,order_i,i,i], columns=xticklabels, index=yticklabels)
            
        plt.figure(dpi=50)
        #plt.title("Slice "+str(i+1))
        ax = sns.heatmap(df, cmap="Reds")
        ax.invert_yaxis()
        plt.savefig("results/"+name+"_frontal_slice"+str(i+1)+".pdf", bbox_inches='tight')
        plt.show()


#%% Spectral Hypergraph Wavelet

if wavelet:
    if M == 3:
        lmax = np.max(np.diag(eigenvalues_hat[:,:,frontal_slice]))
    if M == 4:
        lmax = np.max(np.diag(eigenvalues_hat[:,:,frontal_slice,frontal_slice]))
    
    if filters_name == "spectral":
        filters = tools_wavelet.spectral(n_filters,lmax)
    
    if filters_name == "uniform_translates":
        filters = tools_wavelet.uniform_translates(n_filters,lmax)
    
    if filters_name == "spectrum_adapted":
        if M == 3:
            approx_spectrum = tools_wavelet.spectrum_cdf_approx(N,lmax,L_sym_hat[:,:,frontal_slice])
        if M == 4:
            approx_spectrum = tools_wavelet.spectrum_cdf_approx(N,lmax,L_sym_hat[:,:,frontal_slice,frontal_slice])
            
        filters = tools_wavelet.spectrum_adapted(n_filters, lmax, approx_spectrum)
    
    wav = np.zeros((N,n_filters))
    for n in range(N):
        for p in range(n_filters):
            for l in range(N):
                if M == 3:
                    wav[n,p] += filters[p](eigenvalues_hat[order[l],order[l],frontal_slice]) * x_hat[l] * eigenvectors_hat[n,order[l],frontal_slice]
                if M == 4:
                    wav[n,p] += filters[p](eigenvalues_hat[order[l],order[l],frontal_slice,frontal_slice]) * x_hat[l] * eigenvectors_hat[n,order[l],frontal_slice,frontal_slice]
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
    if name == "path_hypergraph21":
        plt.xticks(rotation=90)
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_"+filters_name+".pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_"+filters_name+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_"+filters_name+".pdf", bbox_inches='tight')
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
    
    if M == 3:
        x_spectrum = np.diag(eigenvalues_hat[:,:,frontal_slice])
    if M == 4:
        x_spectrum = np.diag(eigenvalues_hat[:,:,frontal_slice,frontal_slice])
        
    plt.plot(x_spectrum,np.zeros(len(x_spectrum)),'kx', mew=1.5)
    plt.savefig("results/"+name+"_kernels_"+filters_name+".pdf", bbox_inches='tight')
    plt.show()

#%% Hypergraph Tikhonov Regularization

if regularization:
    
    if M == 3:
        gr = tools_regularization.hypergraph_regularization(L_sym_hat[:,:,frontal_slice], gamma)
    if M == 4:
        gr = tools_regularization.hypergraph_regularization(L_sym_hat[:,:,frontal_slice,frontal_slice], gamma)
    if M == 5:
        gr = tools_regularization.hypergraph_regularization(L_sym_hat[:,:,frontal_slice,frontal_slice,frontal_slice], gamma)
    
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
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_gamma"+str(gamma)+".pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_gamma"+str(gamma)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_gamma"+str(gamma)+".pdf", bbox_inches='tight')
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
        if signal == "delta":
            plt.savefig("results/"+name+"_delta"+str(center+1)+"_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "eigenvector":
            plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "eigenvectors":
            plt.savefig("results/"+name+"_eigenvectors_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "exponential":
            plt.savefig("results/"+name+"_exponential_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        if signal == "manually":
            plt.savefig("results/"+name+"_manually_gamma"+str(gamma)+"_sign.pdf", bbox_inches='tight')
        plt.show()


#%% Hypergraph Regularization Centrality

if plot_degree:
    centralities = dict()
    for i in range(N):
        if M == 3:
            centralities[i+1] = D[i,i,i]
        if M == 4:
            centralities[i+1] = D[i,i,i,i]
    
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
        if M == 3:
            centralities[i+1] = np.abs(eigenvectors_hat[i,order[-1],frontal_slice])
        if M == 4:
            centralities[i+1] = np.abs(eigenvectors_hat[i,order[-1],frontal_slice,frontal_slice])
    
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
    
    if M == 3:
        centralities = hgrc(L_sym_hat[:,:,frontal_slice], gamma_hgrc)
    if M == 4:
        centralities = hgrc(L_sym_hat[:,:,frontal_slice,frontal_slice], gamma_hgrc)
    
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


#%% t-Hypergraph Fourier Transform

if t_shifting or t_HGFT:
    if M == 3:
        X = np.outer(x,x)
        
        X_expand = tensor_operations.t_expand(X)
        
        X_expand_sym = tensor_operations.sym(X_expand)
    if M == 4:
        X = tensor_operations.outer(np.outer(x,x),x)
        
        X_expand = tensor_operations.t_expand4(X)
        
        X_expand_sym = tensor_operations.sym4(X_expand)
    
    Ns = 2*N+1

    t_xticklabels = range(1,Ns+1)
    t_yticklabels = range(1,N+1)
    
if t_shifting:
    X_expand_sym_shift = tensor_operations.t_product(L_sym, X_expand_sym)
    X_expand_sym_shift = np.real(X_expand_sym_shift)
    
    df = pd.DataFrame(X_expand_sym_shift[:,0,:], columns=t_xticklabels, index=t_yticklabels)

    plt.figure()
    ax = sns.heatmap(df, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_t_shifted.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_t_shifted.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_t_shifted.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_t_shifted.pdf", bbox_inches='tight') 
    plt.show()
    
if t_HGFT:
    eigenvalues, eigenvectors = tensor_operations.t_decomposition(L_sym)
    
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    eigenvectors_transpose = tensor_operations.t_transpose(eigenvectors[:,order,:])
    
    X_expand_sym_Fourier = tensor_operations.t_product(eigenvectors_transpose, X_expand_sym)
    X_expand_sym_Fourier = np.real(X_expand_sym_Fourier)
    
    df = pd.DataFrame(X_expand_sym_Fourier[:,0,:], columns=t_xticklabels, index=t_yticklabels)
    
    plt.figure()
    ax = sns.heatmap(df, cmap="Blues")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center+1)+"_tHGFT.pdf", bbox_inches='tight')
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector+1)+"_tHGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_tHGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_tHGFT.pdf", bbox_inches='tight') 
    plt.show()
    

elapsed_time = time.process_time() - start_time
print("Elapsed time:",np.round(elapsed_time,4))













