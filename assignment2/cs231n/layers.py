from builtins import range
from operator import itemgetter
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N, *dims = x.shape
    D, M = w.shape
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_reshape = x.reshape(N, D)
    out = x_reshape.dot(w) + b.reshape((1,M))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N, *dims = x.shape
    D, M = w.shape
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout.dot(w.T)  # (N, D)
    dx = dx.reshape(x.shape) # (N, d1, ..., dk)
    
    x_reshape = x.reshape(N, D)      # (N,D)
    dw = ((dout.T).dot(x_reshape)).T # (D,M)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout 
    dx[x<0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        numer = x - sample_mean
        denom = np.sqrt(sample_var + eps)   # std
        x_hat = numer / denom # normalized x
        out = gamma * x_hat + beta

        cache = {}
        cache['x_hat'] = x_hat
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        cache['numer'] = numer
        cache['denom'] = denom
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['x'] = x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = (x - running_mean) / running_var**0.5
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, numer, denom, gamma, beta, sample_mean, sample_var, x_hat = \
            itemgetter('x', 'numer', 'denom', 'gamma', 'beta', 
                       'sample_mean', 'sample_var', 'x_hat')(cache)

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)

    d_xhat = gamma * dout
    d_denom = np.sum(-(d_xhat * numer) / (denom**2), axis=0)
    d_sample_var = 0.5 * d_denom / denom

    d_numer = d_xhat / denom
    d_sample_mean = -np.sum(d_numer, axis=0)

    # x contributes to sample_mean, sample_var, and numerator
    dsample_mean_dx = 1.0 / N
    dsample_var_dx = 2.0 * (x - sample_mean) / N
    dx = d_numer + d_sample_mean * dsample_mean_dx + d_sample_var * dsample_var_dx

    assert(dgamma.shape == gamma.shape)
    assert(d_xhat.shape == x_hat.shape)
    assert(d_denom.shape == denom.shape)
    assert(dx.shape == x.shape)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, numer, denom, gamma, beta, sample_mean, sample_var, x_hat = \
            itemgetter('x', 'numer', 'denom', 'gamma', 'beta', 
                       'sample_mean', 'sample_var', 'x_hat')(cache)

    dgamma = np.sum(np.multiply(dout, cache['x_hat']), axis=0)
    dbeta = np.sum(dout, axis=0)

    d_xhat = dout * gamma
    d_var = np.sum(d_xhat * (x-sample_mean) * (-0.5) * denom**(-3), axis=0)
    d_mean = np.sum(-d_xhat / denom, axis=0) + \
             d_var * np.mean(-2 * (x - sample_mean), axis=0)

    dx = d_xhat / denom + \
         d_var * 2 * (x - sample_mean) / N +\
         d_mean / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    layer_mean = np.mean(x, axis=1)
    layer_var = np.var(x, axis=1)
    numer = colsub(x, layer_mean)
    denom = np.sqrt(layer_var + eps)
    xhat = coldiv(numer, denom)
    out = gamma * xhat + beta

    cache = {}
    cache['layer_mean'] = layer_mean
    cache['layer_var'] = layer_var
    cache['numer'] = numer
    cache['denom'] = denom
    cache['xhat'] = xhat 
    cache['gamma'] = gamma
    cache['beta'] = beta
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    layer_mean, layer_var, numer, denom, xhat, gamma, beta = \
            itemgetter('layer_mean', 'layer_var', 'numer', 'denom', 'xhat',
                       'gamma', 'beta')(cache)
    dgamma = np.sum(dout * xhat, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    d_xhat = dout * gamma
    d_layer_var = np.sum(colmul(d_xhat * numer * (-0.5), denom**(-3)), axis=1)
    d_layer_mean = np.sum(coldiv(-d_xhat, denom), axis=1) + \
                   (-2) * d_layer_var * np.mean(numer, axis=1)

    #import pdb; pdb.set_trace()
    dx = coladd(coldiv(d_xhat, denom), d_layer_mean / D) + \
         colmul(numer, d_layer_var * (2/D))
                

    assert(dgamma.shape == gamma.shape)
    assert(dbeta.shape == beta.shape)
    assert(dx.shape == xhat.shape)
    assert(d_layer_var.shape == (N,))
    assert(d_layer_mean.shape == (N,))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

def coladd(matrix, colvec):
    return f_mat_colvec(matrix, colvec, np.add)

def colsub(matrix, colvec):
    return f_mat_colvec(matrix, colvec, np.subtract)

def colmul(matrix, colvec):
    return f_mat_colvec(matrix, colvec, np.multiply)

def coldiv(matrix, colvec):
    return f_mat_colvec(matrix, colvec, np.divide)

def f_mat_colvec(matrix, colvec, operation):
    """
    Given (M, N) matrix, and (M,) vec, apply operation to 
    matrix and column vector built from concatenating colvec
    N times horizontally
    """
    return operation(matrix.T, colvec).T

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None


    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    p = dropout_param['p']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout / p
        dx[mask==0] = 0
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    Hout = int(1 + (H + 2 * pad - HH) / stride)
    Wout = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, Hout, Wout))
    
    for i in np.arange(N):              # for each data point (image)
        # pad the data point
        padded = np.zeros((1, C, H+2*pad, W+2*pad))
        padded[:, :, pad:-pad, pad:-pad] = x[i, :, :, :]
        
        idx_height = 0 
        for vs in np.arange(Hout):          # for each height stride
            # starting height idx for this stride
            idx_height = stride * vs
            idx_width = 0
            for hs in np.arange(Wout):      # for each width stride
                idx_width = stride * hs

                #print(f'idx_height={idx_height}, idx_width={idx_width}')
                for f in np.arange(F):      # for each filter
                    out[i,f,vs,hs] = np.sum(padded[:, :, idx_height:(idx_height+HH), idx_width:(idx_width+WW)] * w[f, :, :, :]) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, F, Hout, Wout = dout.shape

    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    # initial empty dx, dw, db
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant') # pad x as before
    
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f, :, :])    # sum of each depth-slice of dout
            for h_out in range(Hout):
                for w_out in range(Wout):
                    h_start = h_out*stride
                    w_start = w_out*stride
                    dx_pad[n, :, h_start:h_start+HH, w_start:w_start+WW] += w[f, :, :, :] * dout[n, f, h_out, w_out]
                    dw[f, :, :, :] += x_pad[n, :, h_start:h_start+HH, w_start:w_start+WW] * dout[n, f, h_out, w_out]

    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    ph, pw, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    Hout = int(1 + (H - ph) / stride)
    Wout = int(1 + (W - pw) / stride)
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    out = np.zeros((N, C, Hout, Wout))
    for n in range(N):
        for c in range(C):
            for h_out in range(Hout):
                for w_out in range(Wout):
                    h_start = h_out*stride
                    w_start = w_out*stride
                    out[n, c, h_out, w_out] = np.max(x[n, c, h_start:h_start+ph, w_start:w_start+pw])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    ph, pw, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, Hout, Wout = dout.shape
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for h_out in range(Hout):
                for w_out in range(Wout):
                    h_start = h_out*stride
                    w_start = w_out*stride
                    idx = np.unravel_index(np.argmax(x[n, c, h_start:h_start+ph, w_start:w_start+pw]), (ph, pw))
                    dx[n, c, h_start:h_start+ph, w_start:w_start+pw][idx] += dout[n, c, h_out, w_out]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape

    # reshape x to (N*H*W)*C to call batch normalization, this way
    # C becomes the D in batchnorm_forward
    x_new = np.reshape(np.transpose(x, (0, 2, 3, 1)), (-1, C))
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)

    # reshape out
    out = np.transpose(np.reshape(out, (N, H, W, C)), (0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape

    # reshape dout to (N*H*W)*C to call batch normalization, this way
    # C becomes the D in batchnorm_forward
    dout_new = np.reshape(np.transpose(dout, (0, 2, 3, 1)), (-1, C))
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)

    # reshape dout
    dx = np.transpose(np.reshape(dx, (N, H, W, C)), (0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    N, C, H, W = x.shape
    # Transform to (N*G)-by-(C//G)*H*W data to use layer_norm
    x_new = np.reshape(x, (N*G, (C//G)*H*W))

    # copy from layernorm_forward -- normalize by group
    layer_mean = np.mean(x_new, axis=1)
    layer_var = np.var(x_new, axis=1)
    numer = colsub(x_new, layer_mean)
    denom = np.sqrt(layer_var + eps)
    xhat = coldiv(numer, denom)

    # reshape the result and transform
    # apply the same transform for all groups!
    xhat = np.reshape(xhat, (N, C, H, W))
    out = gamma * xhat + beta

    cache = {}
    cache['layer_mean'] = layer_mean
    cache['layer_var'] = layer_var
    cache['numer'] = numer
    cache['denom'] = denom
    cache['xhat'] = xhat 
    cache['gamma'] = gamma
    cache['beta'] = beta
    cache['G'] = G
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N, C, H, W = dout.shape
    layer_mean, layer_var, numer, denom, xhat, gamma, beta = \
            itemgetter('layer_mean', 'layer_var', 'numer', 'denom', 'xhat',
                       'gamma', 'beta')(cache)
    G = cache['G']
            
    # sum over everything except for C
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3), keepdims=True) 
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    # reshape back to get stuff back, use layernorm code
    D = (C//G)*H*W
    d_xhat = dout * gamma
    d_xhat = np.reshape(d_xhat, (N*G, D))
    d_layer_var = np.sum(colmul(d_xhat * numer * (-0.5), denom**(-3)), axis=1)
    d_layer_mean = np.sum(coldiv(-d_xhat, denom), axis=1) + \
                    (-2) * d_layer_var * np.mean(numer, axis=1)

    dx = coladd(coldiv(d_xhat, denom), d_layer_mean / D) + \
         colmul(numer, d_layer_var * (2/D))
    # reshape result back
    dx = np.reshape(dx, (N, C, H, W))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
