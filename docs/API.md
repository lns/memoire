## API

In the python module `memoire`, we provide 3 primary classes for implementing a distributed replay memory,
namely `ReplayMemory`, `ReplayMemoryClient`, and `ReplayMemoryServer`. The class `ReplayMemory` is the main
class for managing generated replay data, and the class `ReplayMemoryClient` and `ReplayMemoryServer` wrap
over `ReplayMemory` to handle network connections.

### ReplayMemory

The class of `ReplayMemory` can be seen as a local memory structure for storing the episode data.
In this class, we define the following properties
```python
class ReplayMemory:

  # Read only properties
  state_buf     # buffer template for state
  action_buf    # buffer template for action
  reward_buf    # buffer template for reward, must have dtype = float32
  prob_buf      # buffer template for prob, must have shape = []
  value_buf     # buffer template for value, must have dtype = float32 and the same shape as reward_buf
  qvest_buf     # buffer template for qvest, must have dtype = float32 and the same shape as reward_buf
  info_buf      # buffer template for info
  entry_size    # byte size of (s,a,r,p,v,q,i)
  max_step      # max number of samples can be stored in this ReplayMemory
  uuid          # universally unique identifier for this instance

  # Read write properties
  priority_exponent   # \beta: the exponent coefficient for prioritized sampling 
  mix_lambda          # \lambda: mixture coefficient for computing multi-step return
  frame_stack         # number of frames stacked for each state (default 1)
  multi_step          # number of steps between prev and next (default 1)
  cache_size          # number of samples in a cache
  max_episode         # max number of episodes allowed in this ReplayMemory
  reuse_cache         # whether to discard used cache or to reuse them
  discount_factor     # \gamma: the (multidimensional) discount factor used for cumulating reward
  reward_coeff        # mixture coefficient for multi-dimensional reward
  cache_flags         # whether previous (s,a,r,p,v,q,i) and next (s,a,r,p,v,q,i) should be cached in caches
```
The `ReplayMemory` supports the following methods as
```python
class ReplayMemory:

  def __init__(self, state_buf, action_buf, reward_buf, prob_buf, value_buf, qvest_buf, info_buf, max_step):
    """ Construct a ReplayMemory with these properties """
    pass

  def print_info(self):
    """ Print various info of self. (const method) """
    pass

  def num_episode(self):
    """ Number of episodes currently stored in this ReplayMemory. (const method) """
    pass

  def pub_bytes(self, topic, message):
    """ Publish a message to the topic, can be received by sub_bytes(topic) """
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

  def close_episode(self, episodic_weight_multiplier=1.0, do_update_value=True, do_update_weight=True):
    """ Close previously opened episode

    This API will close the episode opened previously by a call of `new_episode()`.
    The samples added to this episode by `add_entry()` will be saved and post-processed to compute
    multi-step return R_t and priority weight w_t for sampling.
    The behaviour of closing a closed episode is undefined and should be avoided.
    The number of `num_episode()` will increase by 1 after calling this method.

    :param  episodic_weight_multiplier   Episodic weight multiplier. This will scale the sampling weight for
                                         all samples in an episode.
    :param  do_update_value              Whether we should update q-value estimation value (qvest) for samples
                                         in this episode.
    :param  do_update_weight             Whether we should update the prioritized sampling weight for samples
                                         in this episode.
    """
    pass

  def add_entry(self, s, a, r, p, v, i, init_w):
    """ Add an entry to currently opened episode

    This API will add an entry (s,a,r,p,v,i) to current opened episode.
    The behaviour of adding entries to a closed episode is undefined.
    The number of `num_episode()` may decrease by 1 if space is insufficient.

    :param  s: state   (should match state_buf)
    :param  a: action  (should match action_buf)
    :param  r: reward  (should match reward_buf)
    :param  p: prob    (should match prob_buf)
    :param  v: value   (should match value_buf)
    :param  i: info    (should match info_buf)
    :param  init_w:    initial sample weight (normally 1.0) """
    pass
 
```

### ReplayMemoryServer
A `ReplayMemoryServer` is usually used with learner worker. The server receives sampled data from
clients and prepares batches of samples for training. The communication is built on facilities provided by [ZeroMQ](http://zeromq.org/).
For the meaning of "REQ/REP", "PULL/PUSH" protocal, as well as "endpoint" and "proxy", please refer to [ZeroMQ - The Guide](http://zguide.zeromq.org/page:all).
The class `ReplayMemoryServer` supports following methods
```python
class ReplayMemoryServer
  # Read-only properties
  rem                # Replay Memory
  logfile            # Path of logfile
  total_episodes
  total_caches
  total_steps

  def __init__(self, state_buf, action_buf, reward_buf, prob_buf, value_buf, qvest_buf, info_buf, \
              max_step, pub_endpoint, n_caches):
    """ Initialize a ReplayMemoryServer

    :param  pub_endpoint:   endpoint for PUB/SUB protocal
    :param  n_caches:       number of caches kept at the server side """
    pass

  def set_logfile(self, logfile_path, mode):
    """ Set the path of logfile

    Messages sent from actors will be written to this file. Each message contains the received time,
    the uuid of the sender, and the actual message. The format is `'%s,%08x,%s\n' % (timestamp, uuid, message)`.
    For performance issue, the messages are saved to disk with buffer.

    :param  logfile_path:   path of logfile
    :param  mode:           open mode (e.g. 'w', 'a') as in `man fopen`. """
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

    a PULL worker is responsible for receiving caches from clients. (See `push_cache()` for the definition of cache).
    The server stores recent caches received from multiple clients in a FIFO queue. The length of queue is
    configurable with `n_caches`.

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

    This call can be used to prepare a batch of samples for the neural network learner. The samples are sampled
    without replacement (if `reuse_cache` is set to False) from caches stored in server's cache queue.
    We define a transition as the pair of previous stacked states {(s,a,r,p,v,q,i)} and the next state (s,a,r,p,v,q,i).
    The length of stacked states is configurable with `frame_stack`, while the number of steps between the latest
    state in previous stacked states and the next state is configured by `multi_step`.
    The `get_batch()` call will return a batch of transitions, as well as their prioritized weight of sampling.
    Please see our vignette for the details of prioritized sampling. 

    This function returns on successfully constructing a batch, or raises exception on failure.
    
    :param  batch_size: Batch size

    :raise  RuntimeError: A RuntimeError will be raised if get_batch() failed. This is usually caused by insufficient
                          caches at the ReplayMemoryServer side. See the error info printed for detail reasons.
    
    :rtype: tuple(prev, next, weight) """
    pass
```

### ReplayMemoryClient
A `ReplayMemoryClient` is usually used in an actor worker to store generated experience locally, and communicate
with the remote `ReplayMemoryServer`. The class supports the following methods
```python
class ReplayMemoryClient
  # Read-only properties
  rem                # Replay Memory

  def __init__(self, sub_endpoint, req_endpoint, push_endpoint):
    """ Initialize a ReplayMemoryClient

    This function call will construct a ReplayMemory, with sizes synchronized from the remote ReplayMemoryServer.

    :param  sub_endpoint:  endpoint for PUB/SUB protocal
    :param  req_endpoint:  endpoint for REP/REQ protocal
    :param  push_endpoint: endpoint for PUSH/PULL protocal
    :param  input_uuid:    this can be used to specify actor IP address. 0 for random.
                           See examples/client_mt.py for an example."""
    pass

  def sync_sizes(self, max_step):
    """ Manually synchonize sizes from the server

    This function will send a request to the server to query about *_sizes, and reconstruct the local ReplayMemory
    with the new parameters.

    The function returns on response received.

    :param  max_step: local ReplayMemory's max_step """
    pass

  def update_counter(self):
    """ Update local counter to the server. Should be called after each finished episode.

    The function returns on response received.
    """
    pass

  def push_cache(self):
    """ Construct and push a cache to the server. Should be called periodically.

    A cache is a batch of transitions sampled (with replacement) from the client's local replay memory.
    The sampling weight is proportional to the prioritized weight calculated in `close_episode()`.
    The size of cache (how number samples in a cahce) is configured by `cache_size`.

    This function may fail when there is no samples with positive sampling weight in local replay memory.
    It returns 0 on cache sent successfully, or -1 on failure.

    :rtype: int """
    pass

  def write_log(self, log_message):
    """ Send a log message to be saved to logfile at the server.

    :param  log_message: a string of message to be sent """
    pass

  def sub_bytes(self, topic):
    """ Subscribe to messages of topic

    This function call will block until a message is received.

    :rtype: str """
    pass
```

