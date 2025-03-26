import torch
import matplotlib.pyplot as plt
losses = []

# 摰𡁶儔閮梶毀鞈   
sizes = torch.tensor([1500, 2000, 1200], dtype=torch.float32)
bedrooms = torch.tensor([3, 4, 2], dtype=torch.float32)
prices = torch.tensor([300000, 400000, 250000], dtype=torch.float32)

#   嘥 见 𡝗 𢠃  
w1 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)

# 摰𡁶儔閮梶毀餈游  
learning_rate = 0.0000001
epochs = 10000

for epoch in range(epochs):
    #   鞉葫    
    predictions = w1 * sizes + w2 * bedrooms
    
    # 閮  埈 滚仃
    loss = torch.mean((predictions - prices) ** 2)
    
    # 雿輻鍂  滚 穃 單偘閮  埈０摨 
    loss.backward()
    losses.append(loss.item())

    #  凒 鰵甈𢠃  
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        # 撠 ０摨行飛 妟
        w1.grad.zero_()
        w2.grad.zero_()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Weighting
print(f'weighting嚗鯱1 = {w1.item()}, w2 = {w2.item()}')