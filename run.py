from nexus import Nexus
model=Nexus()
outs=model.generate("hey there tell me something about AI",max_new_tokens=50)
print(outs)