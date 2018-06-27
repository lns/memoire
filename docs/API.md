## API

In the python module `memoire`, we provide 3 primary classes for implementing a distributed replay memory,
namely `ReplayMemory`, `ReplayMemoryClient`, and `ReplayMemoryServer`. The class `ReplayMemory` is the main
class for managing generated replay data, and the class `ReplayMemoryClient` and `ReplayMemoryServer` wrap
over `ReplayMemory` to handle network connections.

### ReplayMemory

The class of `ReplayMemory` can be seen as a local memory structure for storing the episode data.
In this class, we define the following **readonly** properties
```python
class ReplayMemory:
  # Read only properties
  state_size    # size of state  (dtype = np.uint8)
  action_size   # size of action (dtype = np.float32)
  reward_size   # size of reward (dtype = np.float32)
  prob_size     # size of prob   (dtype = np.float32)
  value_size    # size of value  (dtype = np.float32)
  entry_size    # byte size of (s,a,r,p,v)
  capacity      # max number of sample can be stored in this ReplayMemory
  uuid          # universally unique identifier for this instance
```
and **readwrite** properties
```python
class ReplayMemory:
  # Read write properties
  discount_factor     # \gamma: the discount factor used for cumulate reward
  priority_exponent   # \beta: the coefficient for prioritized sampling 
  td_lambda           # \lambda: mixture coefficient for computing multi-step return
  frame_stack         # number of frames stacked for each state (default 1)
  multi_step          # number of steps between prev and next (default 1)
  cache_size          # number of samples in a cache
  max_episode         # max number of episodes allowed in this ReplayMemory
  reuse_cache         # whether to discard used cache or to reuse them
  rwd_coeff           # mixture coefficient for multi-dimensional reward (should match `reward_size`)
  cache_flags         # whether previous (s,a,r,p,v) and next (s,a,r,p,v) should be cached in caches
```
The `ReplayMemory` supports the following methods as
```python
class ReplayMemory:

  def __init__(self, state_size, action_size, reward_size, prob_size, value_size, capacity):
    """ Constructe a ReplayMemory with these properties """
    pass

  def print_info(self):
    """ Print various info of self. (const method) """
    pass

  def num_episode(self):
    """ Number of episodes currently stored in this ReplayMemory. (const method) """
    pass

  def new_episode(self):
    """ Create a new episode

    This API call will find a place in internal storage for a new episode.
    The number of `num_episode()` won't change on calling this method,
    until the current episode is closed by `close_episode()` (number of episode +1),
    or `add_entry()` (may cause number of episode -1 due to limited space).
    A second call of `new_episode()` will clear the added samples
    in previously opened episode, if the previous episode is not closed by a call of `close_episode()`. """
    pass

  def close_episode(self):
    """ Close previously opened episode

    This API will close the episode opened previously by a call of `new_episode()`.
    The samples added to this episode by `add_entry()` will be saved and post-processed to compute
    multi-step return R_t and priority weight w_t for sampling.
    The behaviour of closing a closed episode is undefined and should be avoided.
    The number of `num_episode()` will increase by 1 after calling this method. """
    pass

  def add_entry(self, s, a, r, p, v, init_w):
    """ Add an entry to currently opened episode

    This API will add an entry (s,a,r,p,v) to current opened episode.
    The behaviour of adding entries to a closed episode is undefined and should be avoided.
    The number of `num_episode()` may decrease by 1 if space is insufficient.

    :param  s: state   (np.array of np.uint8)
    :param  a: action  (np.array of np.float32)
    :param  r: reward  (np.array of np.float32)
    :param  p: prob    (np.array of np.float32)
    :param  v: value   (np.array of np.float32)
    :param  init_w: initial sample weight (normally 1.0) """
    pass
    
```

### ReplayMemoryServer
A `ReplayMemoryServer` is usually used with learner worker. The server receives sampled data from
clients and prepares batches of samples for training. The communication is built on facilities provided by [ZeroMQ](http://zeromq.org/).
For the meaning of "REQ/REP", "PULL/PUSH" protocal, as well as "endpoint" and "proxy", please refer to [ZeroMQ - The Guide](http://zguide.zeromq.org/page:all).
The class `ReplayMemoryServer` supports following methods
```python
class ReplayMemoryServer

  def __init__(self, replay_memory, n_caches):
    """ Initialize a ReplayMemoryServer

    :param  replay_memory:  a configured ReplayMemory instance
    :param  n_caches:       number of caches kept at the server side """
    pass

  def rep_worker_main(self, endpoint, mode):
    """ Mainloop for a REP worker 

    a REP worker is responsible for receiving counter update (`n_episode` and `n_steps`)
    and also responsing `GetSizes` requests.

    :param  endpoint:  endpoint as in zeromq format
    :param  mode:      'Bind' for binding the endpoint to a port, or 'Conn' for connecting to the endpoint """
    pass

  def pull_worker_main(self, endpoint, mode):
    """ Mainloop for a PULL worker 

    a PULL worker is responsible for receiving caches from clients.

    :param  endpoint:  endpoint as in zeromq format
    :param  mode:      'Bind' for binding the endpoint to a port, or 'Conn' for connecting to the endpoint """
    pass

  def rep_proxy_main(self, front_ep, back_ep):
    """ Mainloop for a REP Proxy

    :param  front_ep:  front endpoint
    :param  back_ep:   back endpoint """
    pass

  def pull_proxy_main(self, front_ep, back_ep):
    """ Mainloop for a PULL Proxy

    :param  front_ep:  front endpoint
    :param  back_ep:   back endpoint """
    pass

  def get_batch(self, batch_size):
    """ Get a batch from the distributed replay memory

    This call can be used to prepare a batch of samples for the neural network learner.
    We define a transition as the pair of a previous states (s,a,r,p,v) and the next state (s,a,r,p,v).
    The `get_batch()` call will return a batch of transitions, as well as their prioritized weight of sampling.
    Please see our vignette for the details of prioritized sampling. 
    
    :param  batch_size: Batch size
    
    :rtype: tuple(prev, next, weight) """
    pass
```

### ReplayMemoryClient
A `ReplayMemoryClient` is usually used in an actor worker to store generated experience locally, and communicate
with the remote `ReplayMemoryServer`. The class supports the following methods
```python
class ReplayMemoryClient

  def __init__(self, req_endpoint, push_endpoint, capacity):
    """ Initialize a ReplayMemoryClient

    This function call will construct a ReplayMemory, with sizes synchronized from the remote ReplayMemoryServer.

    :param  req_endpoint: endpoint for REP/REQ protocal
    :param  push_endpoint: endpoint for PUSH/PULL protocal
    :param  capacity: ReplayMemory's capacity """
    pass

  def sync_sizes(self):
    """ Manually synchonize sizes from the server """
    pass

  def update_counter(self):
    """ Update local counter to the server. Should be called after each finished episode. """
    pass

  def push_cache(self):
    """ Construct and push a cache to the server. Should be called periodically. """
    pass
```

