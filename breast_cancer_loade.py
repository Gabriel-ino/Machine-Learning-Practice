#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:53:13 2021

@author: gabriel
"""


import numpy as np
from keras.models import model_from_json


archive = open('classificador_breast.json', 'r')
neural_structure = archive.read()
archive.close()

classificador = model_from_json(neural_structure)
classificador.load_weights('classificador_breast.h5')

new = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08,
                 0.134, 0.178, 0.2, 0.05, 1098, 0.87, 4500,
                 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007,
                 23.15, 16.64, 178.5, 2021, 0.14, 0.185, 0.84,
                 158, 0.363]])

previsao = classificador.predict(new)
previsao = (previsao > 0.7)