#!/bin/bash

f2py -c geodesiclm.pyf -L../geodesicLM/ -lgeodesiclm  -llapack -lblas -lgfortran
