import tensorflow as tf



class ImageTripletNeuralNet():
    """This class is for a CNN that will operate on a triplet
    of images (template,search,diff) as the atomic unit of data.

    """

    def __init__(self):
        """Initialize the CNN"""
        self.dataset = None

    def preprocess_input_images(self, datadir):
        """Run the preprocessing steps on a set of input images.

        Input images are all in datadir, in sub-directories sorted
        by class.   The name of each subdirectory indicates the class
        (we use labels='inferred').
        """

        self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
            datadir, labels='inferred', label_mode='int',
            class_names=None, color_mode='rgb', batch_size=32,
            image_size=(256,256), shuffle=True, seed=None,
            validation_split=None, subset=None,
            interpolation='bilinear', follow_links=False
            )
        return

