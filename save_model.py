import ann_config_3D as ann

CNN = ann.convolutional_residual()
CNN.load_model()
CNN.save_model('./')
