#------------------------------------------------------------------------------------#
# IMPORTS
#------------------------------------------------------------------------------------#
from torch.nn.functional import softmax, cross_entropy
from torch import sigmoid, sum
# relative
from ileed.utils.helpers import noisy_rational_logloss, IRT_logloss
#------------------------------------------------------------------------------------#
# Main Loss Function
#------------------------------------------------------------------------------------#
def loss(data_tup, state_featurizer_network, action_network, omega, no_sigmoid, loss_type):
    '''
    Main loss function for training our networks used in learn.py
    Inputs:
        data_tup             
        state_featurizer_network 
        action_network    
        omega       
        no_sigmoid        
        loss_type  
    
    Outputs:
        log likelihood of loss 
    '''
    # breakpoint()
    states, actions, _ = data_tup
    prob_a_vec = action_network.prob_forward(states) 
    if loss_type == 'irt':
        out = state_featurizer_network(states) # difficulty of state
        sigma = sigmoid(sum(out * omega.unsqueeze(1), dim=-1))
        # prob_a_vec[i][j][k] = prob of action k being optimal at state[i][j] 
        log_likelihood = IRT_logloss(sigma=sigma, actions=actions, prob_a_vec=prob_a_vec)

    elif loss_type == 'bc':
        log_likelihood = cross_entropy(prob_a_vec.reshape(actions.shape[0]*actions.shape[1],-1),actions.reshape(-1))
    
    return log_likelihood