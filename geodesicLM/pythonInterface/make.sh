#!/bin/bash

f2py -c geodesiclm.pyf -L/home/sean/GitRepos/SloppyJoes/geodesicLM/geodesicLM/ -lgeodesiclm  -llapack -lblas -lgfortran
