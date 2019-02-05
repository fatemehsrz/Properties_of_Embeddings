import csv
import os

egoId = []
file2 = open('ego.pairs', "r")

for line in file2:
   a1= line.rstrip('\n').split(' ') 
   a2= [int(i) for i in a1]
   egoId.append(a2)


with open('ego_pair_train.csv', 'w', newline='') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(egoId)
                         

egoList=[0, 107, 1684, 1912 , 3437, 348, 3980, 414, 686, 698]
pathhack = os.path.dirname(os.path.realpath(__file__))


ranking=[]       
pairs1=[] 
pairs2 =[]     
csvfile1= open('ego_pair_train.csv') 
readCSV1 = csv.readCSV1 = csv.reader(csvfile1, delimiter=',')
for row1 in readCSV1:
    pairs1.append((row1[0],row1[1]))

rank_dic={}
csvfile2= open('rank_label.csv')    
readCSV2 = csv.reader(csvfile2, delimiter=',')
for row2 in readCSV2:   
    pairs2.append((row2[0],row2[1]))
    rank_dic.update({(row2[0],row2[1]):row2[2]})
    
for i in range(len(pairs1)):
    for j in range(len(pairs2)):
         if pairs1[i]==pairs2[j]:
            ego_rank =[] 
            
            print(pairs2[j][0],pairs2[j][1],'ranking:',rank_dic[pairs2[j]])
          
            ego_rank.append(rank_dic[pairs2[j]])
          
    ranking.append(ego_rank) 

                   

with open('ego_rank_label2.csv', 'w', newline='') as f1:
    a = csv.writer(f1, delimiter=',')
    a.writerows(ranking)
                         
                    
