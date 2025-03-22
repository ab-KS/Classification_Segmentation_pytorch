class dataset(Dataset):
  transformReqByQuestion = transforms.Compose([
    transforms.Resize((360, 480)),  # Resizing the image
    transforms.ToTensor(),  # Convert to torch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing it
    ])
  

  resizeTransform = transforms.Compose([
    transforms.Resize((360, 480))  # Resizing the image
,transforms.ToTensor()])
    
  dictLabelsValues ={('64', '128', '64'): 0, ('192', '0', '128'): 1, ('0', '128', '192'): 2, ('0', '128', '64'): 3, ('128', '0', '0'): 4, ('64', '0', '128'): 5, ('64', '0', '192'): 6, ('192', '128', '64'): 7, ('192', '192', '128'): 8, ('64', '64', '128'): 9, ('128', '0', '192'): 10, ('192', '0', '64'): 11, ('128', '128', '64'): 12, ('192', '0', '192'): 13, ('128', '64', '64'): 14, ('64', '192', '128'): 15, ('64', '64', '0'): 16, ('128', '64', '128'): 17, ('128', '128', '192'): 18, ('0', '0', '192'): 19, ('192', '128', '128'): 20, ('128', '128', '128'): 21, ('64', '128', '192'): 22, ('0', '0', '64'): 23, ('0', '64', '64'): 24, ('192', '64', '128'): 25, ('128', '128', '0'): 26, ('192', '128', '192'): 27, ('64', '0', '64'): 28, ('192', '192', '0'): 29, ('0', '0', '0'): 30, ('64', '192', '0'): 31}
     
  def __init__(self,images_path, labels_path):
    self.images_path = images_path
    self.labels_path = labels_path
    self.images_list = os.listdir(images_path)#name of all images
    self.labels_list = os.listdir(labels_path)#name of all masks
    self.images_list.sort()
    self.labels_list.sort()
  def __len__(self):
    return len(self.images_list)
  def _labelAssigningFunction(self, label):
    label_np = label.numpy().astype(int)#label tensor to label numpy; faster indexing
    # Reshaping label to (H*W, 3) for efficient lookup
    h, w = label_np.shape[1], label_np.shape[2]
    label_reshaped = label_np.reshape(3, -1).T  # Shape (H*W, 3)
    # Convert RGB values to string format (keys for dict lookup)
    label_strings = [tuple(map(str, rgb)) for rgb in label_reshaped]
    # Use NumPy's vectorized lookup for fast dictionary mapping
    labelAssigned = np.array([dataset.dictLabelsValues.get(rgb, 30) for rgb in label_strings])
    # Reshape back to (H, W) and convert to PyTorch tensor
    mask_Image = torch.tensor(labelAssigned.reshape(h, w))
    return mask_Image
      
  def __getitem__(self,index):
    image_name = self.images_list[index]
    image_path = os.path.join(self.images_path,image_name)
    #print(image_name)
    #print(image_path)
    label_name = self.labels_list[index]
    label_path = os.path.join(self.labels_path,label_name)
    #print(label_name)
    #print(label_path)
    image = Image.open(image_path)
    image_tensor = dataset.transformReqByQuestion(image.convert("RGB"))
    #image_tensor = (image_tensor*255).round().to(torch.uint8)
    #image_array = np.array(image_tensor, dtype=np.uint8)  # Convert to NumPy array (int)
    #print(image_tensor)
    label = Image.open(label_path)
    label_tensor = dataset.resizeTransform(label.convert("RGB"))
    label_tensor = (label_tensor*255).round().to(torch.uint8)

    #label_array = np.array(label_tensor, dtype=np.uint8)  # Convert to NumPy array (int)
    #print(label_tensor)
    mask_Image = self._labelAssigningFunction(label_tensor)
    return image_tensor, label_tensor, mask_Image, image_name, label_name#, image_array, label_array
