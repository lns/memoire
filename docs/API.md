## API

In the python module `memoire`, we provide 3 primary classes for implementing a distributed replay memory,
namely `ReplayMemory`, `ReplayMemoryClient`, and `ReplayMemoryServer`. The class `ReplayMemory` is the main
class for managing generated replay data, and the class `ReplayMemoryClient` and `ReplayMemoryServer` wrap
over `ReplayMemory` to handle network connections.

In the class of `ReplayMemory`, we define the following **readonly** properties
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
