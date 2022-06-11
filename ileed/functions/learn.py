#------------------------------------------------------------------------------------#
# IMPORTS
#------------------------------------------------------------------------------------#
from numpy import log, inf
from torch import randn, no_grad
from torch.optim import Adam
from tqdm import trange
# internal
from ileed.functions.loss import loss
from ileed.utils.models import MLP
from ileed.utils.helpers import evaluate_model
from ileed.utils.helpers import state_featurizer_loss
#------------------------------------------------------------------------------------#
# Main Learning Function
#------------------------------------------------------------------------------------#
def learn(data_tup, env, network_args, no_sigmoid, loss_type, num_iters, n_eval, num_restarts):
    '''
    Main loss function for training our networks used in learn.py
    Inputs:
        data_tup             
        networks 
        embedding_dim           
        no_sigmoid        
        loss_type  
    
    Outputs:
        log likelihood of loss 

    TODO: Some of these variables should be generalized (learning rates and number of iterations)
    '''
    # init constants needed
    device = network_args['device']
    M = len(data_tup[0])
    # init where we store data
    best_log = inf
    best_eval = None
    best_omega = None
    all_logs = []
    all_evals = []
    # iterate searching for smallest NLL
    for n in trange(num_restarts):
        # init the networks
        state_featurizer_network = MLP(input_size=network_args['indim'], 
                                    hidden_size=network_args['hidden_size'], 
                                    output_size=network_args['embed_dim'],
                                    device=device,
                                    seed=network_args['seed']+n).to(device)
        # define action network 
        action_network = MLP(input_size=network_args['indim'], 
                            hidden_size=network_args['hidden_size'], 
                            output_size=network_args['outdim'],
                            device=device,
                            seed=network_args['seed']+n).to(device)  
        # this network maps from latent embedding to next state (in embedded space)
        latent_transition_network = MLP(input_size=network_args['embed_dim'] + 1, 
                                        hidden_size=network_args['hidden_size'], 
                                        output_size=network_args['embed_dim'],
                                        device=device,
                                        seed=network_args['seed']+n).to(device) if network_args['use_latent'] else None    

       
        
        # the omegas to be learned for the M experts
        omega = randn(M, network_args['embed_dim'], requires_grad=True, device=device)
        # one optim for the networks, one for the omegas
        optim1 = Adam(list(state_featurizer_network.parameters()) + list(action_network.parameters()), lr=1e-3)
        optim2 = Adam([omega], lr=1e-2)
        # also learn the latent_transition_network if provided
        if latent_transition_network is not None: optim3 = Adam(list(state_featurizer_network.parameters()) + list(latent_transition_network.parameters()), lr=1e-3)
        # go over iterations taking derivative of the NLL
        for i in range(num_iters):
            log_likelihood = loss(data_tup, state_featurizer_network, action_network, omega, no_sigmoid, loss_type)
            optim1.zero_grad()
            optim2.zero_grad()
            (log_likelihood).backward()
            optim1.step()
            optim2.step()
            if latent_transition_network is not None:
                aux_loss = state_featurizer_loss(data_tup, state_featurizer_network, latent_transition_network)
                optim3.zero_grad()
                aux_loss.backward()
                optim3.step()
        # once you are done store the log loss if it is better
        with no_grad():
            # compute log
            final_log = loss(data_tup, state_featurizer_network, action_network, omega, no_sigmoid, loss_type)
            all_logs.append(final_log.item())
            # eval net
            if final_log.item() < best_log:
                best_log = final_log.item()
                networks = {
                    'state_featurizer_network': state_featurizer_network,
                    'action_network': action_network,
                    'latent_transition_network': latent_transition_network,
                }
                best_eval = [0,0]
                best_omega = omega
    return best_omega, networks, best_eval, all_logs, all_evals


