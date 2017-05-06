n_input = 256
n_output = 10
def model_to_theta(model):
    return np.concatenate((model[0].T.flatten(), model[1].flatten()))

def theta_to_model(theta):
    n_hid = int(len(theta) / (n_input+n_output))
    input_to_hid = theta[:n_input*n_hid].reshape(n_input,n_hid).T
    hid_to_class = theta[n_input*n_hid:].reshape(n_hid, n_output)
    return input_to_hid, hid_to_class
    
def initial_model(n_hid):
    # n_hid: number of hidden logistic units
    n_params = (n_input+n_output) * n_hid
    # No initialized ramdomly to get always the same result
    as_row_vector = np.cos(range(0,(n_params)))* 0.1
    #print((model_to_theta(theta_to_model(as_row_vector)) - as_row_vector).sum())
    return theta_to_model(as_row_vector)