import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

cnn = load_model('model/', custom_objects=None)

test_image = image.load_img('yalefaces/check/subject02/subject02.wink.bmp', target_size = (64, 64))
#test_image = image.load_img('yalefaces/check/random/1.bmp', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

# print(training_set.class_indices)

maxprob = 0
bestmatch = -1
for i in range(0,15):
    if maxprob < result[0][i]:
        maxprob = result[0][i]
        bestmatch = i+1

print('\n\n',result)

print('The person detected is Subject #{}'.format(bestmatch))