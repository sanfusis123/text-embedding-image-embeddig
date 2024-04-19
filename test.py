import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
df = pd.read_csv('image_classes.csv')
plt.figure(figsize=(15,10))
for i in range((n:=len(df))):
    plt.subplot((n//4) + 1,4,i+1)
    img = mpimg.imread(df.iloc[i]['file_name'])
    plt.imshow(img)
    plt.title(df.iloc[i]['cluster'])
plt.show()