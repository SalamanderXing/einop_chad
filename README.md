# Einop Chad

_One chad op to rule them all_


Almost exactly like [Einop](https://github.com/cgarciae/einop) but it also supports `einsum` and `pack`.


## Installation
```

git clone https://github.com/SalamanderXing/einop_chad

cd einop_chad

pip install einop_chad
```

## Usage

#### Einsum

```python
import numpy as np
from einop import einop

a = np.random.uniform(size=(10, 20))
b = np.random.uniform(size=(20, 15))

y = einop(a, b, "a b, b c -> a c") # matrix multiplication

assert y.shape == (10, 15)
```

#### Pack

```python
import numpy as np
from einop import einop

a = np.random.uniform(size=(10, 15))
b = np.random.uniform(size=(20, 15))
c = np.random.uniform(size=(30, 15))

y = einop((a, b, c) "* a") # concatenate along axis 0

assert y.shape == (10 + 20 + 30, 15)
```



#### Rearrange
```python
x = np.random.randn(100, 5, 3)

einop(x, 'i j k -> k i j').shape
>>> (3, 100, 5)
```

#### Reduction
```python
x = np.random.randn(100, 5, 3)

einop(x, 'i j k -> i j', reduction='sum').shape
>>> (100, 5)
```

#### Repeat
```python
x = np.random.randn(100, 5, 3)

einop(x, 'i j k -> i j k l', l=10).shape
>>> (100, 5, 3, 10)
```
