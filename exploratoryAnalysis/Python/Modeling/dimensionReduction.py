from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=4)

twoDim = pca.fit_transform(modelingDF)

n_pcs= pca.components_.shape[0]

most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

topContribution = modelingDF.iloc[:,most_important]