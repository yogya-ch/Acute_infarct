####Inference Engine

from openvino.inference_engine import IENetwork, IEPlugin
import os
import time
import cv2
import argparse
import numpy as np
import tkinter as tk

root= tk.Tk()
canvas1 = tk.Canvas(root, width = 300, height = 300)
canvas1.pack()

target_names = {1: 'Bilateral cerebellar hemispheres', 2: 'Bilateral cerebellar hemispheres and vermis', 3: 'Bilateral frontal lobes', 4: 'Bilateral occipital lobes', 5: 'Brainstem', 6: 'Lacunar infarct in dorsal aspect of pons', 7: 'Lacunar infarct in left parietal lobe', 8: 'Lacunar infarct in medulla oblongata on the left', 9: 'Lacunar infarct in pons on the left', 10: 'Lacunar infarct in posterior limb of left internal capsule', 11: 'Lacunar infarct in right corona radiata', 12: 'Lacunar infarct in right putamen', 13: 'Lacunar infarcts in bilateral occipital lobes', 14: 'Lacunar infarcts in left corona radiata', 15: 'Lacunar infarcts in the right parietal lobe', 16: 'Left centrum semi ovale and right parietal lobe', 17: 'Left cerebellar hemisphere', 18: 'Left cerebellar lacunar infarcts', 19: 'Left frontal lobe', 20: 'Left frontal lobe in precentral gyral location', 21: 'Left fronto-parietal lobe', 22: 'Left fronto-temporo-parietal region', 23: 'Left insula', 24: 'Left occipital and temporal lobes', 25: 'Left occipital lobe', 26: 'Left parietal lobe', 27: 'Left thalamic lacunar infarct', 28: 'Medial part of right frontal and parietal lobes', 29: 'Mid brain on right side', 30: 'Pontine infarct on the right', 31: 'Right anterior thalamic infarct', 32: 'Right cerebellar hemisphere', 33: 'Right cerebellar hemisphere infarct', 34: 'Right corona radiata', 35: 'Right frontal lobe', 36: 'Right fronto-parietal lobe', 37: 'Right fronto-parieto-temporo- occipital lobes', 38: 'Right ganglio-capsular region', 39: 'Right insula', 40: 'Right lentiform nucleus', 41: 'Right occipital lobe', 42: 'Right parietal lacunar infarct', 43: 'Right parietal lobe', 44: 'Right temporal lobe', 45: 'Right thalamus', 46: 'Splenium of the corpus callosum'}

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input", help="path to input", required=True)
ap.add_argument("-d","--device", help="Device", required=True)
ap.add_argument("-m","--model", help="path to xml file", required=True)
args = ap.parse_args()
#reading the model----

model_xml = str(args.model)
model_bin = str(os.path.splitext(model_xml)[0])+ ".bin"

#Setup Devices----
plugin = IEPlugin(device=str(args.device)) #MYRIAD
net = IENetwork(model=model_xml, weights=model_bin)

#Allocating input and output blobs----
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))


#load model to plugin----
exec_net = plugin.load(network=net, num_requests=2)

#read and preprocess input images---
n, c, h, w = net.inputs[input_blob].shape
#print(n,c,h,w)

"""Note: compile OpenCV with JPEG file support enabled."""


#print(args.input)
image = cv2.imread(str(args.input))
if(image[:-1] != (w,h)):
	image = cv2.resize(image, (w, h))
image = image.transpose((2,0,1)) #as openvino expects in this format HWC to CHW
	
res = exec_net.infer(inputs={input_blob:image})
res = res[out_blob]
a = np.amax(res)
result = np.where(res == a)
out = target_names[(result[1][0])+1]
print("-----------------------------")
print(out)
print("-----------------------------")

label1 = tk.Label(root, text= out, fg='green', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=label1)
root.mainloop()