from common_imports import torch, nn, torchvision, transforms, optim, os
import resnet
from trainer_v2 import train_model
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from data_loader import get_splits
from model_bed import get_models, get_simple_cnn
import pandas as pd
import numpy as np
import torch.nn.functional as F

if __name__ == "__main__":

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    dataset_paths = {'fixed_color':'./Data/cmnist/fixed/test/', 
                    'rand_color':'./Data/cmnist/random/test/'}

    num_classes = 10

    # models_dict = get_models(num_classes, False)

    models_dict = get_simple_cnn(num_classes)

    for dataset_type, dataset_path in  dataset_paths.items():

        dataset = ImageFolder(dataset_path, transform=transforms.Compose([transforms.Resize((224,224)), 
                                                                    transforms.ToTensor(),    
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

        for model_name, model in models_dict.items():
            
            for model_flavor in dataset_paths.keys():
                
                cur_model_name = model_name + "_" + model_flavor + ".pth"

                model_path = "output/models/best_" + cur_model_name

                print("Infer Model: {0}".format(model_path))

                model.load_state_dict(torch.load(model_path))
                model.eval()

                y_pred = []
                y_true = []
                y_softmax = []

                with torch.no_grad():

                    for images, labels in data_loader:
                        model.to(device)
                        images = images.to(device)
                        outputs = model(images)

                        softmax_scores = F.softmax(outputs, dim=1)
                        
                        max_values, preds = torch.max(softmax_scores, dim=1)
                        
                        preds = torch.argmax(outputs, dim=1)
                        y_pred.extend(preds.cpu().numpy())
                        y_true.extend(labels.cpu().numpy())
                        y_softmax.extend(max_values.cpu().numpy())
            
                df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred, 'score':y_softmax})
                filtered_df = df[df.y_true == df.y_pred]
                filtered_df.groupby('y_true')['score'].agg(['mean', 'std']).reset_index()

                output_path = "output/logs/" + cur_model_name.split('.')[0] + "_" + dataset_type + ".csv"

                print("Logging: {0}".format(output_path))
                filtered_df.to_csv(output_path, index=False)


                                                        
