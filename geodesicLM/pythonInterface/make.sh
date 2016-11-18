#!/bin/bash

f2py -c geodesiclm.pyf -L/home/sean/GitRepos/SloppyJoes/geodesicLMv1.1/geodesicLM/ -lgeodesiclm  -llapack -lblas -lgfortran
