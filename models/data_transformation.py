import torchvision.transforms as transforms
def create_transformation():
  train_transform=transforms.Compose([
        transforms.Resize((256,512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
        #RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
        transforms.Pad(10),
        transforms.RandomCrop((256,512)),
        transforms.ToTensor(),
        transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  parts_transform=transforms.Compose([
          transforms.Resize((64,64)),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(10),
          transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
          #RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
          transforms.Pad(10),
          transforms.RandomCrop((64,64)),
          transforms.ToTensor(),
          transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

  trunk_transform=transforms.Compose([
          transforms.Resize((64,128)),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(10),
          transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
          #RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
          transforms.Pad(10),
          transforms.RandomCrop((64,128)),
          transforms.ToTensor(),
          transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

  val_transform=transforms.Compose([
          transforms.Resize((256,512)),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  val_parts_transform=transforms.Compose([
          transforms.Resize((64,64)),
          transforms.ToTensor(),
          transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

  val_trunk_transform=transforms.Compose([
          transforms.Resize((64,128)),
          transforms.ToTensor(),
          transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      
  return train_transform,parts_transform,trunk_transform,val_transform,val_parts_transform,val_trunk_transform