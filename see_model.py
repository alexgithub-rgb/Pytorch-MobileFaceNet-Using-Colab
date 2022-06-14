import os
import re
import shutil
import time
from datetime import datetime, timedelta
import argparse
import functools
import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.reader import Dataset
from models.arcmargin import ArcNet
from models.mobilefacenet_reduce import MobileFaceNet
from utils.utils import add_arguments, print_arguments, get_lfw_list
from utils.utils import get_features, get_feature_dict, test_performance
from thop import profile
print("new1")
model = MobileFaceNet()
summary(model, (3, 112, 112))
input1=torch.randn(1,3, 112, 112)
flops,params=profile(model,inputs=(input1,))
print("FLOPS="+str(flops/1000**3)+"G")
print("PARAMS="+str(params/1000**2)+"M")