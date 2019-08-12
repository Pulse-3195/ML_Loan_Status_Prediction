# Problem Statement :-
# 1. Develop a statistical model for predicting bad loans
# 2. Use the model to identify the most important drivers of bad loans
# 3. With these new insights, make recommendations to avoid funding bad loans

# Required Packages & Libraries:-

# importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import libraries for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# import libraries for model evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

# importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import libraries for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# import libraries for model evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

# import libraries for KNN
from sklearn.neighbors import KNeighborsClassifier

# import library for ignoring warnings in the output
import warnings
warnings.filterwarnings('ignore')

# import libraries for KNN
from sklearn.neighbors import KNeighborsClassifier

# import library for ignoring warnings in the output
import warnings
warnings.filterwarnings('ignore')

#defining working directory
os.chdir('C:\\Users\\hp\\Desktop\\p\\Data_Science_Course\\Piyush\\Excercises')
