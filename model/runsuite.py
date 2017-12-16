from experiments import doExperiment
import numpy as np


params={
    'logging':True,
    'epochs':128,
    'batch_size':16,
    'ongpu':False,
    'train_ratio':0.5,
    'feature_dropout' : 0.0,
    'walkers':None,
    'shuffleEx':True,
    'shuffleNodes':True,
    'experiment':'sentiment'}

results=np.zeros((5,1,4,1,1))
losses=[]
for trial in range(5):
    # i=0
    # for network in ['qw1c']:#,'dc']:
    #     params['network']=network
        j=0
        params['qw_network']='qw1c'
        for walk_length in [1,2,3,4]:
            params['walk_length']=walk_length
            k=0
            for learn_amps in [False]:
                params['learn_amps']=learn_amps
                l=0
                for learn_coin in [True]:#,False]:
                    if learn_coin==learn_amps and learn_amps==False:
                        break
                    params['learn_coin']=learn_coin
                    results[trial,i,j,k,l],los=doExperiment(**params)
                    losses.append(los)
                    np.save('results1c',results)
                    # if network!='qw':
                    #     break
                    l+=1
                # if network!='qw':
                #     break
                k+=1
            j+=1
        # i+=1