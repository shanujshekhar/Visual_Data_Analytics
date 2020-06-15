import json
from flask import Flask, render_template, request, redirect, Response, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
from sklearn.manifold import MDS

def readData(file):
	data = pd.read_csv(file)
	return data

def randomSampling(data):
	randomData = data.sample(frac=0.25)
	randomData.to_csv('diamonds_randomSampled.csv', index=False)
	return randomData
	
def stratifiedSampling(data):
	# Elbow Method to find Optimal Number of Clusters K
	dataElbow = deepcopy(data)
	models = []
	distortions = []

	clusters = range(1,10)
	for k in clusters:
		model = KMeans(n_clusters=k).fit(dataElbow)
		predictions = model.predict(dataElbow)

		models.append(predictions)
		distortions.append(sum(np.min(cdist(dataElbow, model.cluster_centers_, 'euclidean'), axis=1)) / dataElbow.shape[0])

	# print ("Printing the Elbow Plot....")
	
	# plt.plot(clusters, distortions, 'bo-')
	# plt.xlabel('Number of Clusters k')
	# plt.ylabel('Distortion')
	# plt.title('The Elbow Method showing the optimal k')
	# plt.show()

	optimalClusters = 2
	clusterNos = models[optimalClusters-1]

	columns = dataElbow.columns

	dataElbow['clusterNo'] = clusterNos

	cluster1 = dataElbow[dataElbow['clusterNo']==0]
	cluster2 = dataElbow[dataElbow['clusterNo']==1]

	cluster1 = cluster1.sample(n = (int(len(data) * 0.25)//optimalClusters))

	cluster2 = cluster2.sample(n = (int(len(data) * 0.25)//optimalClusters))

	clusters = [cluster1, cluster2]

	stratifiedData = pd.concat(clusters)

	stratifiedData.to_csv('diamonds_stratifiedSampled.csv', index=False)
	return stratifiedData

def PCALoadings(loadings, columns, heading):

	attr_loads = []

	pcs = 2
	for i in range(loadings.shape[0]):
		sm = 0
		for j in range(pcs):
			t = loadings['PC' + str(j+1)][i]
			sm += (t * t)
		attr_loads.append([columns[i], sm])
	
	return attr_loads

def find_pca(heading, data, columns):
	
	scaled_data = preprocessing.scale(data)
	components = 10
	pca = PCA(n_components=components)
	final_data = pca.fit_transform(scaled_data)

	per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
	labels = [x for x in range(1, len(per_var)+1)]
	

	attributes = ['PC' + str(i+1) for i in range(components)]

	scatter = pd.DataFrame(final_data, columns=attributes)
	scatter = scatter[['PC1', 'PC2']]
	scatter.to_csv('scatter_'+ heading + '.csv', index=False)

	if(heading=="stratified"):
		pca_components = pca.components_.T[:len(pca.components_.T)-1]
	else:
		pca_components = pca.components_.T

	loadings = pd.DataFrame(pca_components, columns=attributes, index=columns)

	attribute_loadings = PCALoadings(loadings, columns, heading)
	attribute_loadings.sort(key = lambda x:(-x[1],x[0])) 

	pca_csv = pd.DataFrame(list(zip(labels, per_var, attribute_loadings)), columns=['PC_Number','Variance_Explained', 'Attribute_Loadings'])
	pca_csv.to_csv('pca_'+heading + '.csv', index=False)

def find_mds(data, randomData, stratifiedData):

	embedding = MDS(n_components=2, dissimilarity='precomputed')
	

	print ('Original')
	data = preprocessing.scale(data)
	dmatrix_euc = pairwise_distances(data, metric='euclidean')
	dmatrix_cor = pairwise_distances(data, metric='correlation')
	
	mds_euc = embedding.fit_transform(dmatrix_euc)
	mds_euc = pd.DataFrame(mds_euc, columns=['MDS1_euc', 'MDS2_euc'])

	mds_cor = embedding.fit_transform(dmatrix_cor)
	mds_cor = pd.DataFrame(mds_cor, columns=['MDS1_cor', 'MDS2_cor'])

	mds_orig = pd.concat([mds_euc, mds_cor], axis=1)

	print ('Random')
	randomData = preprocessing.scale(randomData)
	dmatrix_euc = pairwise_distances(randomData, metric='euclidean')
	dmatrix_cor = pairwise_distances(randomData, metric='correlation')

	mds_euc = embedding.fit_transform(dmatrix_euc)
	mds_euc = pd.DataFrame(mds_euc, columns=['MDS1_euc', 'MDS2_euc'])

	mds_cor = embedding.fit_transform(dmatrix_cor)
	mds_cor = pd.DataFrame(mds_cor, columns=['MDS1_cor', 'MDS2_cor'])

	mds_random = pd.concat([mds_euc, mds_cor], axis=1)

	print ('Stratified')
	stratifiedData = preprocessing.scale(stratifiedData)
	dmatrix_euc = pairwise_distances(stratifiedData, metric='euclidean')
	dmatrix_cor = pairwise_distances(stratifiedData, metric='correlation')
	
	mds_euc = embedding.fit_transform(dmatrix_euc)
	mds_euc = pd.DataFrame(mds_euc, columns=['MDS1_euc', 'MDS2_euc'])

	mds_cor = embedding.fit_transform(dmatrix_cor)
	mds_cor = pd.DataFrame(mds_cor, columns=['MDS1_cor', 'MDS2_cor'])


	mds_stratified = pd.concat([mds_euc, mds_cor], axis=1)

	mds_orig.to_csv('MDS_original.csv', index=False)
	mds_random.to_csv('MD_random.csv', index=False)
	mds_stratified.to_csv('MDS_stratified.csv', index=False)

def makeCSVs(data):
	randomData = randomSampling(data)
	stratifiedData = stratifiedSampling(data)

	find_mds(data, randomData, stratifiedData)

	find_pca("original", data, data.columns)
	find_pca("random", randomData, data.columns)
	find_pca("stratified", stratifiedData, data.columns)

app = Flask(__name__)
@app.route("/", methods = ['POST', 'GET'])
def index():

	full_data = pd.read_csv('diamonds_LabelEncode.csv')
	full_data = full_data[:1000]
	random_data = pd.read_csv('diamonds_randomSampled.csv')
	strat_data = pd.read_csv('diamonds_stratifiedSampled.csv')
	original = pd.read_csv('pca_original.csv')
	random = pd.read_csv('pca_random.csv')
	stratified = pd.read_csv('pca_stratified.csv')
	scatter_original = pd.read_csv('scatter_original.csv')
	scatter_random = pd.read_csv('scatter_random.csv')
	scatter_stratified = pd.read_csv('scatter_stratified.csv')
	mds_orig = pd.read_csv('MDS_original.csv')
	mds_random = pd.read_csv('MD_random.csv')
	mds_stratified = pd.read_csv('MDS_stratified.csv')

	data  = {}
	data['full_data'] = full_data.to_dict(orient='records')
	data['random_data'] = random_data.to_dict(orient='records')
	data['strat_data'] = strat_data.to_dict(orient='records')
	data['original'] = original.to_dict(orient='records')
	data['random'] = random.to_dict(orient='records')
	data['stratified'] = stratified.to_dict(orient='records')
	data['scatter_original'] = scatter_original.to_dict(orient='records')
	data['scatter_random'] = scatter_random.to_dict(orient='records')
	data['scatter_stratified'] = scatter_stratified.to_dict(orient='records')
	data['mds_orig'] = mds_orig.to_dict(orient='records')
	data['mds_random'] = mds_random.to_dict(orient='records')
	data['mds_stratified'] = mds_stratified.to_dict(orient='records')
	
	json_data = json.dumps(data)
	final_data = {'chart_data': json_data}
	return render_template("index.html", data=final_data)

if __name__ == "__main__":
	# data = readData('diamonds_LabelEncode.csv')
	# data = data[:1000]
	# makeCSVs(data)
	
	app.run(debug=True)
