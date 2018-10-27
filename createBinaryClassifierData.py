import numpy as np
import pandas as pd
from dataGeneratorClass import dataGenerator, plotData # this module is on my GitHub page
# Generate data 

catg_lvls = [5, 8, 38] # 3 categorical variables with corresponding # of levels
n_cat = len(catg_lvls)
n_beta = 2 # number of beta variables
n_uniform = 1 # of uniform variables
n = n_cat + n_beta + n_uniform
N = 200000 # number of observations
responseMean = 0.05 # Bernoulli response mean
p_width = 0.1
np.random.seed(123)
trainPercent = 2./3 # fraction of data for training
testPercent = 1./6 # fraction of data for test
Ntrain = int(N * trainPercent);
Ntest = int(N * testPercent)

# Generate data
x = dataGenerator(catg_lvls = catg_lvls, n_beta = n_beta, n_uniform = n_uniform)
df1, df2 = x.generate(100000)
f = x.genSigmoidTransform(df2, loc = responseMean, width = p_width)
df1, df2 = x.generate_more(N)
df = pd.concat([df1,pd.DataFrame(x.genBernoulliVariates(df2,f),columns=['Response'])],axis=1)

df.loc[:,'Feature: 0'] = 'A' + df.loc[:,'Feature: 0'].astype(str)
df.loc[:,'Feature: 1'] = 'B' + df.loc[:,'Feature: 1'].astype(str)
df.loc[:,'Feature: 2'] = 'C' + df.loc[:,'Feature: 2'].astype(str)

df.to_csv('BinaryClassifierData.csv',index=False)