import model as model

#initializing the model
generator = model.get_model()

cats = model.getCategories()
intLabel = cats.index('ayn')

print(intLabel)