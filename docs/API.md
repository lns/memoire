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
  reward_buf    # buffer template for reward, must have dtype = float32
  prob_buf      # buffer template for prob, must have shape = []
  value_buf     # buffer template for value, must have dtype = float32 and the same shape as reward_buf
  qvest_buf     # buffer template for qvest, must have dtype = float32 and the same shape as reward_buf
  num_slot      # number of slots
  entry_size    # byte size of (s,a,r,p,v,q,i)
  max_step      # max number of samples can be stored in this ReplayMemory
  uuid          # universally unique identifier for this instance

  # Read write properties
  priority_exponent   # \beta: the exponent coefficient for prioritized sampling 
  mix_lambda          # \lambda: mixture coefficient for computing multi-step return
  rollout_len         # length of rollout
  do_padding          # whether to do padding, this is mainly for frame stacking
  priority_decay      # decay priority for later stats (normally 1.0)
  traceback_threshold # threshold to stop traceback computation (default 1e-4)
  discount_factor     # \gamma: the (multidimensional) discount factor used for cumulating reward
  reward_coeff        # mixture coefficient for multi-dimensional reward
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
  # Read-write properties
  pub_endpoint       # Endpoint for PUB
  pub_hwm            # high water mark (HWM) for PUB
  rep_hwm            # HWM for REP
  pull_hwm           # HWM for PULL
  pull_buf_size      # Buffer size for PULL

  def __init__(self, entry, max_step, n_slot):
    """ Construct a ReplayMemory with these properties

    :param  entry:          This is used to define the shape and dtypes. We require entry[-3] is reward, entry[-2] is prob, and entry[-1] is value. See examples for usage.
    :param  max_step:       Max number of steps stored in memory for data from each client.
    :param  n_slot:         Number of slots, each client occupies a slot.
    """
    pass

  def close(self):
    """ Close the server """
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

    REP worker is responsible for answering GET_INFO messages. See `proto/msg.proto` for details.

    :param  endpoint:  endpoint as in zeromq format
    :param  mode:      'Bind' for binding the endpoint to a port, or 'Conn' for connecting to the endpoint """
    pass

  def pull_worker_main(self, endpoint, mode):
    """ Mainloop for a PULL worker 

    PULL worker is responsible for receiving PUSH_DATA messages. See `proto/msg.proto` for details.

    :param  endpoint:  endpoint as in zeromq format
    :param  mode:      'Bind' for binding the endpoint to a port, or 'Conn' for connecting to the endpoint """
    pass

  def rep_proxy_main(self, front_ep, front_mode, back_ep, back_mode):
    """ Mainloop for a REP Proxy

    :param  front_ep:  front endpoint
    :param  front_mode:Bind or Conn
    :param  back_ep:   back endpoint
    :param  back_mode: Bind or Conn """
    pass

  def pull_proxy_main(self, front_ep, front_mode, back_ep, back_mode):
    """ Mainloop for a PULL Proxy

    :param  front_ep:  front endpoint
    :param  front_mode:Bind or Conn
    :param  back_ep:   back endpoint
    :param  back_mode: Bind or Conn """
    pass

  def pub_proxy_main(self, front_ep, front_mode, back_ep, back_mode):
    """ Mainloop for a PUB Proxy

    :param  front_ep:  front endpoint
    :param  front_mode:Bind or Conn
    :param  back_ep:   back endpoint
    :param  back_mode: Bind or Conn """
    pass

  def get_data(self, batch_size):
    """ Get data from the distributed replay memory

    This call can be used to prepare a batch of samples for the neural network learner.
    The `get_data()` call will return a batch of transitions, as well as their prioritized weight of sampling.
    This function may block if we do not have data in memory.
    
    :param  batch_size: Batch size
    
    :rtype: tuple(prev, next, weight) """
    pass

  def pub_bytes(self, s):
    """ Pub a string to clients.

    This call will pub a string to the subscribed clients.
    Usually this function is used to publish latest model to the actors.

    :param  s:         bytes of data
    """
    pass
```

### ReplayMemoryClient
A `ReplayMemoryClient` is usually used in an actor worker to push data to and receive model from the remote `ReplayMemoryServer`. The class supports the following methods
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
                           See examples/client_mt.py for an example."""
    pass

  def close(self):
    """ Close a client """
    pass

  def add_entry(self, entry, is_term):
    """ Add an entry to the replay memory. The data well be locally cached before
    sent to the remote server. Please see examples for usage.

    :param  entry:         data entry
    :param  is_term:       whether this is a terminal state (end of episode)
    """
    pass

  def push_log(self, log_message):
    """ Send a log message to be saved to logfile at the server.

    :param  log_message: a string of message to be sent """
    pass

  def sub_bytes(self, topic):
    """ Subscribe to messages of topic

    This function call will block until a message is received.

    :rtype: str """
    pass
```

