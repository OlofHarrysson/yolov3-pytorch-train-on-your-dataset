with open('label_index.txt') as f:
  lines = f.read().splitlines() # or f.read()

n_in_sample_index = 50
lines = lines[:n_in_sample_index]

with open('sample_label_index.txt', 'w') as f:
  f.writelines("\n".join(lines)) # or f.write()