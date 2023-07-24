from torchvision import models
import torch
from torchvision import transforms
import sys
from PIL import Image
from imagenet_classes import classes
import os
from torchvision.datasets import ImageNet
from tqdm import tqdm
from typing import Tuple
import json
from imagenet_x import load_annotations
import pandas as pd

if len(sys.argv) != 2:
	print("Usage: python classify.py path/to/image/or/imagenet/root/directory")
	exit(1)

class MyImageNet(ImageNet):
	def __init__(self, image_dir, split='val', transform=None):
		super().__init__(image_dir, split=split, transform=transform)
		self.annotations = load_annotations(partition=split)
		meta_class_ls = list(self.annotations["metaclass"].unique())
		meta_class_ls.sort()
		self.meta_class_mappings = dict()
		for meta_class_idx in range(len(meta_class_ls)):
			self.meta_class_mappings[meta_class_ls[meta_class_idx]] = meta_class_idx


	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
		"""
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
		path, _ = self.samples[index]

		file_name = path.split("/")[-1]

		annotation_df = self.annotations[self.annotations["file_name"] == file_name]

		target = list(annotation_df["metaclass"])[0]
		target = self.meta_class_mappings[target]

		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, path, annotation_df

	@staticmethod
	def collate_fn(data):
		sample_ls = [data[idx][0] for idx in range(len(data))]
		target_ls = [data[idx][1] for idx in range(len(data))]
		path_ls = [data[idx][2] for idx in range(len(data))]
		annotation_ls = [data[idx][3] for idx in range(len(data))]
		sample_tensor = torch.stack(sample_ls, dim = 0)
		target_tensor = torch.tensor(target_ls)
		annotation_tensor = pd.concat(annotation_ls)
		return sample_tensor, target_tensor, path_ls, annotation_tensor

model = models.resnet152(pretrained=True)

transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
	    mean=[0.485, 0.456, 0.406],
	    std=[0.229, 0.224, 0.225]
	)])

if os.path.isfile(sys.argv[1]):
	image_path = sys.argv[1]
	print(f"Inferring label for image: {image_path}")

	img = Image.open(image_path)

	img_t = transform(img)
	batch_t = torch.unsqueeze(img_t, 0)

	model.eval()

	res = model(batch_t)
	_, index = torch.max(res, 1)
	print(index.item())
	percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
	print(classes[index.item()], percentage[index.item()].item())
	

else:
	image_dir = sys.argv[1]
	print(f"Inferring labels for validation set in ImageNet dataset: {image_dir}")

	imagenet_dataset = MyImageNet(image_dir, split='val', transform=transform)
	data_loader = torch.utils.data.DataLoader(imagenet_dataset, collate_fn = MyImageNet.collate_fn,
                                          batch_size=16,
                                          shuffle=False)

	model.eval()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	wnids = imagenet_dataset.wnids
	model.to(device)
	reslist = []
	for image, target, path in tqdm(data_loader):
		image = image.to(device)
		# target = target.to(device, non_blocking=True)
		output = model(image)

		# _, index = torch.max(output, 1)
		# percentage = torch.nn.functional.softmax(output, dim=1) * 100
		# percentages = torch.take(percentage, index)

		# index = index.tolist()
		# percentages = percentages.tolist()
		# target = target.tolist()
		# for p, i, perc, t in zip(path, index, percentages, target):
		# 	reslist.append({
		# 		"path": p,
		# 		"target": t,
		# 		"tname": classes[t],
		# 		"twnid": wnids[t],
		# 		"results": [{
		# 			"pred": i,
		# 			"confidence": perc,
		# 			"predname": classes[i],
		# 			"pwnid": wnids[i]
		# 		}]
		# 	})

		_, indices = torch.sort(output, descending=True)
		indices = indices.tolist()
		percentage = (torch.nn.functional.softmax(output, dim=1) * 100).tolist()
		target = target.tolist()

		for p, i, t, perc in zip(path, indices, target, percentage):
			reslist.append({
				"path": p,
				"target": t,
				"tname": classes[t],
				"twnid": wnids[t],
				"results": [{
					"pred": i[idx],
					"confidence": perc[i[idx]],
					"predname": classes[i[idx]],
					"pwnid": wnids[i[idx]]
				} for idx in range(10)]
			})

	print("Dumping results in imagenet_results.json")

	json.dump(reslist, open('imagenet_results.json', 'w'), indent=2)


	

