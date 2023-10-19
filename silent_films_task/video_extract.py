import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

#define function to extract frames from the raw clips.
def get_frames(filename,n_frames=32):
	frames = [] 
	v_cap = cv2.VideoCapture(filename)
	v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_list = np.linspace(0, v_len - 1, n_frames + 1, dtype=np.int16)
	for fn in range(v_len):
		success, frame = v_cap.read()
		if success is False:
			continue
		if (fn in frame_list):
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frames.append(frame)
	v_cap.release()
	return frames

#we need to transform the frames into a tensor
def transform_frames(frames):
	h, w = 112, 112
	mean = [0.43216, 0.394666, 0.37645]
	std = [0.22803, 0.22145, 0.216989]
	test_transformer = transforms.Compose([
		transforms.Resize((h, w)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])
	frames_tr = []
	for frame in frames:
		frame = Image.fromarray(frame)
		frame_tr = test_transformer(frame)
		frames_tr.append(frame_tr)
	imgs_tensor = torch.stack(frames_tr)
	imgs_tensor = torch.transpose(imgs_tensor, 1, 0)
	# imgs_tensor = imgs_tensor.unsqueeze(0)
	return imgs_tensor

#store the frames on your disk if you want.
def store_frames(frames,path2store):
	for ii,frame in enumerate(frames):
		frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
		path2img=os.path.join(path2store,"frame"+str(ii)+".jpg")
		cv2.imwrite(path2img, frame)


def transform_newframes1(frames):
	h, w = 112, 112
	mean = [0.43216, 0.394666, 0.37645]
	std = [0.22803, 0.22145, 0.216989]
	mean1 = [0.5, 0.5, 0.5]
	std1 = [0.5, 0.5, 0.5]
	resize = transforms.Resize((h,w))
	norm = transforms.Normalize(mean,std)
	trans1 = transforms.ToTensor()
	rotation = transforms.RandomRotation(90)
	norm2 = transforms.Normalize(mean1,std1)
	colorjitter = transforms.ColorJitter(brightness=(0.5, 0.9), contrast=(0.1, 0.9), saturation=(0.1, 0.9), hue=(-0.5, 0.5))  #改变亮度、对比度饱和度
	randomaffine = transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=66)  #随机仿射变换
	frames_tr = []
	for ii,frame in enumerate(frames):
		frame = Image.fromarray(frame)
		frame = resize(frame)
		prob_rotation = np.random.rand()
		prob_norm2 = np.random.rand()
		prob_colorjitter = np.random.rand()
		prob_affine = np.random.rand()
		if prob_rotation > 0.8:
			frame = rotation(frame)
		if prob_colorjitter > 0.8:
			frame = colorjitter(frame)
		if prob_affine > 0.8:
			frame = randomaffine(frame)

		frame = trans1(frame)
		if prob_norm2 > 0.8:
			frame = norm2(frame)
		else:
			frame = norm(frame)
		frames_tr.append(frame)
	imgs_tensor = torch.stack(frames_tr)
	imgs_tensor = torch.transpose(imgs_tensor, 1, 0)
	# imgs_tensor = imgs_tensor.unsqueeze(0)
	return imgs_tensor