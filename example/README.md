
Experiment 1:
```bash
python server.py &
python client.py
```

Experiment 2:
set `use_proxy = True` in client.py
```bash
python server.py &
python proxy.py &
python client.py
```

