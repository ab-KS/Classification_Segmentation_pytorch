class dataset(Dataset):
    #defining class transform. Resizes in 256x256 and PIL image type to tensor; dtype becomes float32
    transform = transforms.Compose([transforms.Resize((256, 256)),
    transforms.ToTensor()])
    
    def __init__(self,pathOfFolder,dictionaryLabel):
        #intended to store directories of all animals
        self.animal_directories = []
        #stores the path of the folder which we want to access
        self.pathOfFolder = pathOfFolder
        #Stores tuples of image and the label assigned to it
        self.Timage_directories = []
        
        for animal_name, label in dictionaryLabel.items():
            animal_path = os.path.join(pathOfFolder, animal_name)         
            # List all files in the animal folder
            animal_files = os.listdir(animal_path)
                # Store the directory of this animal
            self.animal_directories.append(animal_path)

            # Filter only .jpg images and store them with their label
            for file in animal_files:
                if file.endswith(".jpg") or file.endswith(".jpeg"):
                    image_path = os.path.join(animal_path, file)
                    self.Timage_directories.append((image_path, label))
        
    def __len__(self):
        return(len(self.Timage_directories))
            
    def __getitem__(self,index):
        #Getting the image and the label assigned to the image
        image_path,label = self.Timage_directories[index][0],self.Timage_directories[index][1]
        #We open the image using the Image Library
        image = Image.open(image_path)
        #We convert the image to tensor and resize it as well
        tensorImage = dataset.transform(image)
        #returning the image
        return (tensorImage , label)
        
    # def callClassTensors(self,wantedClass):
    #     return self.classTensors[wantedClass]
    def callImNameLabel(self,imageName):
        return self.imNameLabel[imageName]
    def getAllImagesNames(self):
        return self.allImagesNames
    
