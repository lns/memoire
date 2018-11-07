
Distributed data collection
---------------------------

Example 1:
```bash
python server.py &
python client.py
```

Example 2:
set `use_proxy = True` in client.py
```bash
python proxy.py &
python server.py &
python client.py
```

Publish/Subscribe
-----------------

Example 1:
```bash
python sub.py &
python pub.py
```

Example 2:
set `use_proxy = True` in sub.py
```bash
python proxy.py &
python pub.py &
python sub.py
```

