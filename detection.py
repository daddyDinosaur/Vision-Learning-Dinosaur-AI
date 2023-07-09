import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


def clear_console():
    os.system('cls')


def setup_tensorflow_config():
    tf.config.run_functions_eagerly(False)


def create_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint('modal_a.h5', monitor='val_loss', save_best_only=True)
    return [early_stopping, model_checkpoint]


def create_data_generators(train_dir, validation_dir, target_size=(100, 80), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')
    
    return train_generator, validation_generator


def create_model(input_shape=(100, 80, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(loss='binary_crossentropy',  
                  optimizer='adam', 
                  metrics=['accuracy'])

    return model


def main():
    clear_console()
    setup_tensorflow_config()

    train_dir = 'ai/train'
    validation_dir = 'ai/validation'
    batch_size = 32

    callbacks = create_callbacks()
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir, batch_size=batch_size)

    model = create_model()
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks  
    )

    model.load_weights('modal_a.h5')

    loss, accuracy = model.evaluate(validation_generator)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')


if __name__ == '__main__':
    main()
