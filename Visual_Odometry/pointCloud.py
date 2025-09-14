import pandas as pd
import numpy as np

class PointCloud:
    def __init__(self):
        self.points = pd.DataFrame(columns=['imgID', 'xy', 'XYZ', 'descriptor'])
    
    def addPoints(self, imgID, xy, descriptor, XYZ):
        df = pd.DataFrame()
        df['xy'] = list(xy)
        df['XYZ'] = list(XYZ)
        df['descriptor'] = list(descriptor)
        df['imgID'] = [imgID] * df.shape[0]
        self.points = pd.concat([self.points, df], ignore_index=True)
    
    def findImgPoints(self, imgID):
        return self.points.loc[self.points['imgID'].isin(imgID)]['XYZ'].to_list(), \
               np.vstack(self.points.loc[self.points['imgID'].isin(imgID)]['descriptor'].to_list())
