import os
import networkx as nx
import numpy as np
from networkx.algorithms.centrality import degree_centrality,closeness_centrality, betweenness_centrality
from itertools import groupby
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from collections import defaultdict
from scipy.stats import  histogram
from gensim.models import Word2Vec

egoList=[0, 107, 1684, 1912 , 3437, 348, 3980, 414, 686, 698]
pathhack = os.path.dirname(os.path.realpath(__file__))

 
graph_list=[]
for i in egoList:
    edge_file = ("./data/ego-facebook/%d.fullEdges"  % i)
    G=nx.read_edgelist(edge_file)
    graph_list.append(G)
    nodes= G.nodes()
    
def DegreeDistribution(G):
     
    degs = defaultdict(int)
    for i in G.degree().values(): degs[i]+=1
    items = sorted ( degs.items () )
     
    return items
     
     
     
def sim_degree(g1, g2):
     
    degree_list1=[]
    degree_list2=[]
    degree_dist1= DegreeDistribution(g1)
    degree_dist2= DegreeDistribution(g2)
    num_node1= len(g1.nodes())
    num_node2= len(g2.nodes())
     
    for key, value in degree_dist1:
         
        degree_list1.append(value+1)
        #print('degree of node:', key, 'frequency:',value, value+1)
     
     
    n1= len(degree_list1) 
     
     
    for key, value in degree_dist2:
        
        degree_list2.append(value+1)
           
    n2= len(degree_list2)  
     
     
     
    if n1>n2:
        n= n1-n2
        zero_list= [1] * n 
        degree_list2= degree_list2+zero_list
         
        KL= entropy( degree_list1, degree_list2)
     
    if n1<n2:
        n=n2-n1
        zero_list= [1] * n 
        degree_list1= degree_list1+zero_list
     
        KL= entropy( degree_list1, degree_list2)
         
         
    return KL
 
   
def sim_closeness(g1, g2):
     
    closness_list1=[]
    closness_list2=[]
    closeness_cent1= nx.closeness_centrality(g1)
    closeness_cent2= nx.closeness_centrality(g2)
     
    for key, value in closeness_cent1.items():
        closness_list1.append(value)
     
    n1= len(g1.nodes())
    n2= len(g2.nodes())
     
    numbin= 50
 
    
    hist1= histogram(closness_list1 ,numbins=numbin)
     
    for i in range(len(hist1[0])):
        hist1[0][i]+=1 
         
             
    print(hist1[0])
     
    for key, value in closeness_cent2.items():
        closness_list2.append(value)
           
     
    hist2= histogram(closness_list2,numbins=numbin) 
     
    for i in range(len(hist2[0])):
        hist2[0][i]+=1 
         
    print(hist2[0])
     
     
    KL= entropy(hist1[0] , hist2[0])
     
         
    return KL
 
     
def sim_betweeness(g1, g2):
    betweeness_list1=[]
    betweeness_list2=[]
    betweeness_cent1= nx.betweenness_centrality(g1)
    betweeness_cent2= nx.betweenness_centrality(g2)
     
     
    n1= len(g1.nodes())
    n2=len(g2.nodes())
     
    numbin= 50
 
 
    for key, value in betweeness_cent1.items():
         
        betweeness_list1.append(value)
         
    hist1= histogram(betweeness_list1 ,numbins=numbin)
     
    for i in range(len(hist1[0])):
        hist1[0][i]+=1    
     
    for key, value in betweeness_cent2.items():
          
        betweeness_list2.append(value)
         
               
    hist2= histogram(betweeness_list2,numbins=numbin)
     
    for i in range(len(hist2[0])):
        hist2[0][i]+=1     
    
    KL= entropy(hist1[0] , hist2[0])
            
    return KL 
      
      
def sim_eigen(g1, g2):  
    eigen_list1=[]
    eigen_list2=[]
    eigen_cent1 = nx.eigenvector_centrality(g1)
    eigen_cent2 = nx.eigenvector_centrality(g2)
     
    n1= len(g1.nodes())
    n2=len(g2.nodes())
     
    numbin= 50
     
    for key, value in eigen_cent1.items():
        eigen_list1.append(value)
     
     
    hist1= histogram(eigen_list1 ,numbins=numbin)
      
    for i in range(len(hist1[0])):
        hist1[0][i]+=1  
    
      
    for key, value in eigen_cent2.items():
        eigen_list2.append(value)
           
    hist2= histogram(eigen_list2,numbins=numbin) 
    for i in range(len(hist2[0])):
        hist2[0][i]+=1     
    
         
    KL= entropy(hist1[0] , hist2[0])
     
    return KL
 
 
  
sim_vector=[]
X_train=[]
  
  
for i in range(len(graph_list)):
     for j in range(len(graph_list)):
          
         if i!=j:
              sim_vector=[]
              
              print('ego:',egoList[i],'   ego:',egoList[j])
               
              degree1= sim_degree(graph_list[i],graph_list[j])
              sim_vector.append(degree1)
              print('degree_similaity:',degree1)
               
             
             
              closeness1= sim_closeness (graph_list[i],graph_list[j])
              sim_vector.append(closeness1)
              print('closeness_similaity:',closeness1)
               
                
              betweeness1= sim_betweeness(graph_list[i],graph_list[j])
              sim_vector.append(betweeness1)
              print('betweenness_similaity:',betweeness1)
               
               
              eigen1= sim_eigen(graph_list[i],graph_list[j])
              sim_vector.append(eigen1)
              print('eigenvector_similaity:',eigen1)
               
              print(sim_vector)
               
              print('---------------------------------------------------------------------')
               
              X_train.append(sim_vector) 
  
       
    
           
  
  
X1_train= np.array(X_train) 
print(X1_train.shape) 
       
import csv
with open('train_data.csv', 'w', newline='') as fp:
     a = csv.writer(fp, delimiter=',')
     a.writerows(X1_train)
    
######################################## Load Embeddings

glo_file = 'glo128.emb'
model = Word2Vec.load_word2vec_format(glo_file, binary=False)

n2vec_file = 'node2vec256_pq2.emb'
model = Word2Vec.load_word2vec_format(n2vec_file, binary=False)

gf_file = open('gf.emb', "r")
gf={}
 
for line in gf_file:
    a= line.rstrip('\n').split(' ')
    a1=a[0]
    a2= a[1:]   
    vec= [float(i) for i in a2]
    gf.update({a1:vec})
   
  
print(gf)  

spectral_file = open('ss.emb', "r")
spectral={}

for line in spectral_file:
   a= line.rstrip('\n').split(' ')
   a1=a[0]
   a2= a[1:]   
   vec= [float(i) for i in a2]
   spectral.update({a1:vec})
  
 


f3 = open('ego.pairs' ,'w')   
f4 = open('ego.sim' ,'w')


for i in range(len(egoList)):
    distances=[]
    r=1
    for j in range(len(egoList)):
        if i!= j:
            
            print(egoList[i],egoList[j])
            f3.write(str(egoList[i])+' '+str(egoList[j]))
            f3.write("\n") 
            
            d=np.dot(model[str(egoList[i])], model[str(egoList[j])] )
            
            distances.append((egoList[i],egoList[j], d))
            
    sorted_dist = sorted(distances, key=lambda tup: tup[2], reverse= True)
    
    print('####################################################################')
    
    for k in range(len(sorted_dist)):  
        f4.write(str(sorted_dist[k][0])+' '+str(sorted_dist[k][1])+' '+str(sorted_dist[k][2]))
        print(sorted_dist[k][0], sorted_dist[k][1], sorted_dist[k][2], r)
        r+=1
        f4.write("\n")   
     
f4.close()
f3.close()

    