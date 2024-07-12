import h5py

model_path = 'eye_Model.h5'
with h5py.File(model_path, 'r') as f:
    if 'keras_version' in f.attrs:
        print(f.attrs['keras_version'])

