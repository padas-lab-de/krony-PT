from transformers import MistralModel, MistralConfig

conf1 = MistralConfig()  # original conf
conf2 = MistralConfig(num_hidden_layers=16)  # new conf, num_hidden_layer split by 2

model1 = MistralModel(conf1)
model2 = MistralModel(conf2)

print(f"Number of params: {sum(p.numel() for p in model1.parameters()):_}")
print(f"Number of params: {sum(p.numel() for p in model2.parameters()):_}")

#configuration = model.cfig
