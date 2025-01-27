from app.detector.AE import ae
from app.detector.CNN import CNN
from app.detector.LSTM import LSTM
from app.detector.HBOS import hbos
from app.detector.LOF import lof
from app.detector.IForest import iforest
from app.detector.IForest1 import iforest1
from app.detector.MP import mp
from app.detector.NORMA import norma
from app.detector.OCSVM import ocsvm
from app.detector.PCA import pca
from app.detector.POLY import poly
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