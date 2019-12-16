# Neural Iterated Learning

The structure of the project is illustrated as follows:

1. **Evolutions**: Evolutions control the training of all generations.
2. **Models**: take the inputs and produce losses for updating params.
   1. **Encoders**: sub-module of models to encode different inputs.
   2. **Decoders**: sub-module of models to generate messages based on the 
        representation from Encoders.
   3. **Losses**: sub-module of models to gain loss for training models.
3. **DataIterators**: provide data to models under evolution.
   1. **Prepocesses**: sub-module of DataIterator to provide preprocessing 
    functions.
   2. **Voc**: sub-module of DataIterator to provide dictionaries.
4. **Utils**: Other functions to support evolutions.