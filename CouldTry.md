
### Things we could try

- Changing Model Set up
  - Kernel size
  - Filters
  - Strides
  - Number of convolutional vs dense
  - Adding drop outs
  - Adding batch normization
  - Adding leaky relu
  - Loss function
  - Activtion
- Changing input
  - Given as 254 x 254 x 3
  - Could make grey scale 254 x 254 x 1
  - Could try and downsample image 127 x 127 x 3
- Fit parameters
  - Epochs
  - Batch size
  - Train Validation split
    - Currenly 0.2 on the training data
    - Could try and split before they're combined
    - Could use the training data as the training and test as test


