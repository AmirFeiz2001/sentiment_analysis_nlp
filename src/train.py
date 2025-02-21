from keras.callbacks import EarlyStopping
import os

def train_model(model, train_padded, train_labels, test_padded, test_labels, epochs=20, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    earlystop = EarlyStopping(patience=4, monitor='val_loss', mode='min', verbose=1)
    history = model.fit(
        train_padded, train_labels,
        epochs=epochs,
        validation_data=(test_padded, test_labels),
        callbacks=[earlystop],
        verbose=1
    )
    model.save(os.path.join(output_dir, "sentiment_model.h5"))
    print(f"Model saved to {output_dir}/sentiment_model.h5")
    return history
