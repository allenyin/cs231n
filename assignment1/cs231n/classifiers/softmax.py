import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  D, C = W.shape
  N, D = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in np.arange(N):
      scores = X[i, :].dot(W)   # Gives a 1xC vector of scores for this sample
      scores -= np.max(scores)  # numerical stability -- shift values of score so that highest number is 0
      prob = np.exp(scores) / np.sum(np.exp(scores))
      loss -= np.log(prob[y[i]])

      cur_w = X[i,:].reshape((-1,1)).dot(prob.reshape((1,-1)))
      #import pdb; pdb.set_trace()
      cur_w[:, y[i]] -= X[i,:]
      dW += cur_w

  loss /= N
  dW /= N

  # regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  D, C = W.shape
  N, D = X.shape


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) - np.max(X.dot(W), axis=1).reshape((-1,1))  # xw - logC
  probs = np.exp(scores)
  loss = np.log(np.sum(probs, axis=1)) - scores[np.arange(N), y]
  loss = np.sum(loss)/N

  # dW
  norm_prob = probs/np.sum(probs, axis=1).reshape((-1,1))
  norm_prob[np.arange(N), y] -= 1
  dW = X.T.dot(norm_prob)/N


  # regularization
  loss += reg * np.sum(W * W) 
  dW +  2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

