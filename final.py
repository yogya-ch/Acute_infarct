
# # DATA AUGMENTATION


def augment_dataset(dir_list):
    """
    Arguments:
        file_dir: A string representing the directory where files containg images that we want to augment are found.
    """
    
    #from keras.preprocessing.image import ImageDataGenerator
    #from os import listdir
    
    data_gen = ImageDataGenerator(rotation_range=7, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1,
                                  brightness_range=(0.001, 0.9),
                                  horizontal_flip=False, 
                                  vertical_flip=False, 
                                  fill_mode='nearest'
                                 )
    try:
        os.mkdir('AUGMENTED_DATA')
    except:
        print("AUGMENTED_DATA is already created")
        
    for directory in listdir(dir_list):
        save_to_dir="AUGMENTED_DATA/"+ directory
        try:
            os.mkdir(save_to_dir)
        except:
            print(save_to_dir+"is already created")
        filename="DWI.jpg"
        # load the image
        image = cv2.imread(dir_list + '\\'+directory + '\\' + "DWI.jpg")
        
        image = image.reshape((1,) + image.shape)
        # prefix of the names for the generated sampels.
        save_prefix = 'aug_' + filename[:-4]
            
        i=0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,save_prefix=save_prefix,
                                       save_format='jpg'):
                i += 1
                if i > 20:
                    break
    return 0



augment_dataset("CLEANED_DATA")


# # LOAD THE DATA


def load_data(dir_list, image_size):
    """
    Read images, resize and normalize them. 
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """
    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    target_names={}
    i=0
    for directory in os.listdir(dir_list):
        target_names[i]=directory
        for filename in os.listdir(dir_list+'\\'+directory):
            # load the image
            image = cv2.imread(dir_list + '\\'+directory + '\\' + filename)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            y.append(i)
        i=i+1
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    return X, y,target_names


X,y,target_names = load_data("AUGMENTED_DATA_1",[32,32])

target_names.keys()


# # PLOT SAMPLES


def plot_sample_images(X, y,target_names,n=10):
    """
    Plots n sample images for both values of y (labels).
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
        target_names: Dictonary with key value is no assigned to respective brain region.
    """
    
    for label in target_names.keys():
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]
        
        columns_n = 10
        rows_n = int(n/ columns_n)

        plt.figure(figsize=(10, 1))
        
        i = 1 # current plot        
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])
            plt.xticks([])
            plt.yticks([])
            i += 1
        
        label_to_str = lambda label: target_names[label]
        plt.suptitle(f"Brain inffract: {label_to_str(label)}")
        plt.show()


plot_sample_images(X, y,target_names)


# # SPLIT THE DATASET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
print("X_train =",X_train.shape)
print("X_test =",X_test.shape)
print("y_train =",y_train.shape)
print("y_test =",y_test.shape)



# # BUILDING MODEL 

IMG_WIDTH, IMG_HEIGHT = (32, 32)
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_CLASSES = 46


def build_model(input_shape,num_classes=46):
    """
    Arugments:
        input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)
        num_classes: number of output neurons
    Returns:
        model: A Model object.
    """ 
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(4, 4), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

model = build_model(IMG_SHAPE,NUM_CLASSES)

model.summary()


# ### COMPILE THE MODEL

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test1 = lb.transform(y_test)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### CHECKPOINT


#callbacks
es = EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience=6
)


# ### TRAINING THE MODEL

model.fit(x=X_train, y=y_train, batch_size=21, epochs=80,validation_split=0.2,callbacks=[es])

loss, acc = model.evaluate(x=X_test, y=y_test1)

# ### TESTING THE MODEL

model=model.predict(X_test)
y_predicted=[]
for i in ymodel:
    y_predicted.append(i.argmax())


accuracy = accuracy_score(y_test, y_predicted)
print('Test Accuracy = %.2f' % accuracy)


# ## PLOT OF ACCURACY AND LOSS

history = model.history.history
def plot_metrics(history):
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['acc']
    val_acc = history['val_acc']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
plot_metrics(history) 


# ## inference folder should be in same format as cleaned data 

X1,y1,target1 = load_data("inference",[32,32])

r=model.predict(X1)
y_predicted=[]
predicted_names={ }
i=0
for i in range(len(r)):
    print("inf = " ,y1[i])
    print("predicted=",target_names[r[i].argmax()])
    print("target=",target1[y1[i]])
    y_predicted.append(r[i].argmax())
    predicted_names[y1[i]]=(target_names[r[i].argmax()])


# # Converting to .pb

# 
#     Freezes the state of a session into a pruned computation graph.
# 
#     Creates a new computation graph where variable nodes are replaced by
#     constants taking their current value in the session. The new graph will be
#     pruned so subgraphs that are not necessary to compute the requested
#     outputs are removed.
#     @param session The TensorFlow session to be frozen.
#     @param keep_var_names A list of variable names that should not be frozen,
#                           or None to freeze all the variables in the graph.
#     @param output_names Names of the relevant graph outputs.
#     @param clear_devices Remove the device directives from the graph for better portability.
#     @return The frozen graph definition.

from keras import backend as K
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
   
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])


tf.train.write_graph(frozen_graph, os.getcwd(), 'pro.pb', as_text=False)



