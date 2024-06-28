from epistemon import Tensor
from epistemon import Value
import torch

a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
c = a*b

c_sum = c.sum()
c_sum.backward()

print(a)
print(b)

# Step 1: Create tensors a and b
a = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
b = torch.tensor([4, 5, 6], dtype=torch.float32, requires_grad=True)

# Step 2: Multiply tensors a and b to get tensor c
c = a * b

# Step 3: Perform a backward pass on c. To do this, we need a scalar value, so sum c first.
c_sum = c.sum()
c_sum.backward()

# Step 4: Display the gradients of tensors a and b
print("Gradients of tensor a:", a.grad)
print("Gradients of tensor b:", b.grad)