import torch
from torchdiffeq import odeint

# Define the ODE function function only depends on y but not t
class ODEFunc(torch.nn.Module):
    def forward(self, t, y):
        return -2 * y

# Initial state and time points
y0 = torch.tensor([1.0])  # Initial value y(0) = 1
t = torch.linspace(0, 5, 100)  # Time points from 0 to 5

# Instantiate the ODE function
ode_func = ODEFunc()

# Solve the ODE
y = odeint(ode_func, y0, t)

# Print the result
print("y values:", y)

# Plot the solution
import matplotlib.pyplot as plt
plt.plot(t.numpy(), torch.exp(-2 * t).numpy(), label="Analytical Solution", linestyle='dashed')
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.savefig('current.png')