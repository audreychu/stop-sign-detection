import cv2
import glob as glob
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure


# Compute gradient for Stop Present (Logical = 1)
df = pd.DataFrame()
for name in glob.glob('/Users/audreychu/Documents/4th Year/STA160/StopYes/*jpg')[0:1]:
    jpg = cv2.imread(name)
    image = color.rgb2gray(jpg)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    start = '/Users/audreychu/Documents/4th Year/STA160/StopYes/'
    end = '.jpg'
    name = name[len(start):-len(end)]
    fd = np.append(fd, name)
    fd = pd.DataFrame(fd).T
    df = df.append(fd)

df['Stop'] = [1] * len(df)


# Compute gradient for Stop Absent (Logical = 0)
df2 = pd.DataFrame()
for name in glob.glob('/Users/audreychu/Documents/4th Year/STA160/StopNo/*jpg'):
    jpg = cv2.imread(name)
    image = color.rgb2gray(jpg)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    start = '/Users/audreychu/Documents/4th Year/STA160/StopNo/'
    end = '.jpg'
    name = name[len(start):-len(end)]
    fd = np.append(fd, name)
    fd = pd.DataFrame(fd).T
    df2 = df2.append(fd)

df2['Stop'] = [0] * len(df2)


# Combine dataframes
df_ = pd.concat([df,df2])
df_ = df_.reset_index()
df_.to_csv('hog_data.csv', index=False, encoding='utf-8')