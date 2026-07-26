from app.detector.ae import ae
from app.detector.cnn import CNN
from app.detector.lstm import LSTM
from app.detector.hbos import hbos
from app.detector.lof import lof
from app.detector.isolation_forest import iforest
from app.detector.isolation_forest_raw import iforest1
from app.detector.matrix_profile import mp
from app.detector.norma import norma
from app.detector.ocsvm import ocsvm
from app.detector.pca import pca
from app.detector.poly import poly
import tensorflow as tf
def run_detector(pred_detector, sequence):
    tf.config.set_visible_devices([], 'GPU')
    if pred_detector.upper() == "AE":
        return ae(sequence)
    elif pred_detector.upper() == "CNN":
        return CNN(sequence)
    elif pred_detector.upper() == "LSTM":
        return LSTM(sequence)
    elif pred_detector.upper() == "HBOS":
        return hbos(sequence)
    elif pred_detector.upper() == "LOF":
        return lof(sequence)
    elif pred_detector.upper() == "IFOREST":
        return iforest(sequence)
    elif pred_detector.upper() == "IFOREST1":
        return iforest1(sequence)
    elif pred_detector.upper() == "MP":
        return mp(sequence)
    elif pred_detector.upper() == "NORMA":
        return norma(sequence)
    elif pred_detector.upper() == "OCSVM":
        return ocsvm(sequence)
    elif pred_detector.upper() == "PCA":
        return pca(sequence)
    elif pred_detector.upper() == "POLY":
        return poly(sequence)
