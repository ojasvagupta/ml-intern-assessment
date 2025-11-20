for i in range(5):
      print(f"> sample {i+1}: {model.generate(max_tokens=30, seed=i)}")
